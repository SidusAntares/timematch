from torch.utils.data.sampler import WeightedRandomSampler
import sklearn.metrics
from collections import Counter
from copy import deepcopy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from ideas.source_phase_compactness import (
    SourceSegmentWeightTracker,
    build_source_segment_partition_spec,
    compute_source_structure_loss,
    describe_source_segment_partition_spec,
)
from ideas.source_feature_reshaper import (
    build_source_feature_reshaper,
    compute_dual_path_relation_regularization,
    compute_source_feature_reshaper_regularization,
)
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    RandomTemporalShift,
    Identity,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, to_cuda, cycle


def _check_temporal_index_range(model, positions, applied_shift, tag):
    temporal_encoder = model.temporal_encoder
    min_pos = int(positions.min().item())
    max_pos = int(positions.max().item())
    min_idx = min_pos + applied_shift + temporal_encoder.max_temporal_shift
    max_idx = max_pos + applied_shift + temporal_encoder.max_temporal_shift
    table_size = temporal_encoder.positional_enc.num_embeddings

    if min_idx < 0 or max_idx >= table_size:
        raise ValueError(
            f"{tag} temporal indices out of range: "
            f"positions=[{min_pos}, {max_pos}], shift={applied_shift}, "
            f"embedding_indices=[{min_idx}, {max_idx}], table_size={table_size}. "
            "This usually means an extra temporal shift was applied on top of TimeMatch "
            "alignment or the positional encoding range is inconsistent with the dataset dates."
        )


def train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    assert not getattr(config, "with_shift_aug", False), (
        "TimeMatch with source phase compactness / v2.3 phase-aware partition must not enable "
        "RandomTemporalShift-style source-side augmentation, because compactness is defined on the original source time axis."
    )
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(splits, config, config.balance_source)

    # Setup model
    pretrained_path = f"{config.weights}/fold_{fold_num}"
    weights_checkpoint = getattr(config, "weights_checkpoint", "model.pt")
    if not weights_checkpoint:
        weights_checkpoint = "model.pt"
    if not weights_checkpoint.endswith(".pt"):
        weights_checkpoint = f"{weights_checkpoint}.pt"
    if os.path.isabs(weights_checkpoint):
        checkpoint_path = weights_checkpoint
    else:
        checkpoint_path = os.path.join(pretrained_path, weights_checkpoint)
    print(f"Loading source weights from {checkpoint_path}")
    pretrained_checkpoint = torch.load(checkpoint_path, weights_only=False)
    pretrained_weights = pretrained_checkpoint["state_dict"]
    student.load_state_dict(pretrained_weights)
    teacher = deepcopy(student)
    student.to(device)
    teacher.to(device)

    source_feature_reshaper = build_source_feature_reshaper(
        kind=getattr(config, "source_feature_reshaper", "none"),
        feature_dim=student.spatial_encoder.output_dim,
        strength=getattr(config, "source_feature_reshaper_strength", 0.10),
        kernel_size=getattr(config, "source_feature_reshaper_kernel_size", 3),
    )
    if source_feature_reshaper is not None:
        source_feature_reshaper.to(device)
        if "source_feature_reshaper_state_dict" in pretrained_checkpoint:
            source_feature_reshaper.load_state_dict(pretrained_checkpoint["source_feature_reshaper_state_dict"])

    # Training setup
    global_step, best_f1 = 0, 0
    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_loss_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    steps_per_epoch = config.steps_per_epoch

    params = list(student.parameters())
    if source_feature_reshaper is not None:
        params += list(source_feature_reshaper.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)
    source_phase_partition_spec = build_source_segment_partition_spec(
        source_loader.dataset.date_positions,
        dataset=source_loader.dataset,
        mode=getattr(config, "source_segment_partition_mode", getattr(config, "source_phase_partition_mode", "uniform")),
        segment_count=getattr(config, "source_segment_count", getattr(config, "source_phase_count", 5)),
        gap_threshold=getattr(config, "source_phase_gap_threshold", 45),
        min_points=getattr(config, "source_phase_min_points", 3),
        max_points=getattr(config, "source_phase_max_points", 8),
        max_span=getattr(config, "source_phase_max_span", 120),
        semantic_quantile=getattr(config, "source_segment_semantic_quantile", 0.75),
        semantic_max_samples_per_class=getattr(config, "source_segment_semantic_max_samples_per_class", 128),
        semantic_curvature_trade_off=getattr(config, "source_segment_semantic_curvature_trade_off", 0.5),
        semantic_energy_trade_off=getattr(config, "source_segment_semantic_energy_trade_off", 0.25),
        semantic_similarity_trade_off=getattr(config, "source_segment_semantic_similarity_trade_off", 0.25),
        semantic_max_extra_cuts_per_base=getattr(config, "source_segment_semantic_max_extra_cuts_per_base", 2),
        semantic_merge_boundary_trade_off=getattr(config, "source_segment_semantic_merge_boundary_trade_off", 0.5),
        semantic_aggl_min_points=getattr(config, "source_segment_semantic_aggl_min_points", 3),
        semantic_aggl_target_slack=getattr(config, "source_segment_semantic_aggl_target_slack", 1),
        semantic_aggl_merge_cost_tolerance=getattr(config, "source_segment_semantic_aggl_merge_cost_tolerance", 1.15),
        semantic_aggl_dynamics_trade_off=getattr(config, "source_segment_semantic_aggl_dynamics_trade_off", 0.35),
    )
    print("source segment partition:", describe_source_segment_partition_spec(source_phase_partition_spec))
    phase_weight_tracker = SourceSegmentWeightTracker(
        phase_count=source_phase_partition_spec["phase_count"],
        phase_partition_spec=source_phase_partition_spec,
        min_sample_points_per_phase=getattr(config, "source_phase_min_sample_points", 2),
    )

    source_iter = iter(cycle(source_loader))
    target_iter = iter(cycle(target_loader))
    min_shift, max_shift = -config.max_temporal_shift, config.max_temporal_shift
    target_to_source_shift = 0
    # To evaluate how well we estimate class distribution
    target_labels = target_loader_no_aug.dataset.get_labels()
    actual_class_distr = estimate_class_distribution(target_labels, config.num_classes)
    source_class_distr = estimate_class_distribution(source_loader.dataset.get_labels(), config.num_classes)

    # estimate an initial guess for shift using Inception Score
    if config.estimate_shift:
        shift_estimator = 'IS' if config.shift_estimator == 'AM' else config.shift_estimator
        target_to_source_shift = estimate_temporal_shift(teacher, target_loader_no_aug, device, min_shift=min_shift, max_shift=max_shift, sample_size=config.sample_size, shift_estimator=shift_estimator)
        if target_to_source_shift >= 0:
            min_shift = 0
        else:
            max_shift = 0

        # Use estimated shift to get initial pseudo labels
        pseudo_softmaxes = get_pseudo_labels(teacher, target_loader_no_aug, device, target_to_source_shift, n=None)
        all_pseudo_labels = torch.max(pseudo_softmaxes, dim=1)[1]

    source_to_target_shift = 0
    shift_history = []
    selection_metric_history = []
    for epoch in range(config.epochs):
        progress_bar = tqdm(range(steps_per_epoch), desc=f"TimeMatch Epoch {epoch + 1}/{config.epochs}")
        loss_meter = AverageMeter()

        if config.estimate_shift:
            estimated_class_distr = estimate_class_distribution(all_pseudo_labels, config.num_classes)
            writer.add_scalar("train/kl_d", kl_divergence(actual_class_distr, estimated_class_distr), epoch)
            target_to_source_shift = estimate_temporal_shift(teacher,
                    target_loader_no_aug, device, estimated_class_distr,
                    min_shift=min_shift, max_shift=max_shift, sample_size=config.sample_size,
                    shift_estimator=config.shift_estimator)
            if epoch == 0:
                if config.shift_source:
                    source_to_target_shift = -target_to_source_shift
                else:
                    source_to_target_shift = 0
                min_shift, max_shift = min(target_to_source_shift, 0), max(0, target_to_source_shift)
            writer.add_scalar("train/temporal_shift", target_to_source_shift, epoch)
            shift_history.append(int(target_to_source_shift))

        student.train()
        teacher.eval()  # don't update BN or use dropout for teacher
        if source_feature_reshaper is not None:
            source_feature_reshaper.train()

        all_labels, all_pseudo_labels, all_pseudo_mask = [], [], []
        for step in progress_bar:
            sample_source, (sample_target_weak, sample_target_strong) = next(source_iter), next(target_iter)

            # Get pseudo labels from teacher
            pixels_t_weak, mask_t_weak, position_t_weak, extra_t_weak = to_cuda(sample_target_weak, device)
            with torch.no_grad():
                teacher_preds = F.softmax(teacher.forward(pixels_t_weak, mask_t_weak, position_t_weak + target_to_source_shift, extra_t_weak), dim=1)
            pseudo_conf, pseudo_targets = torch.max(teacher_preds, dim=1)
            pseudo_mask = pseudo_conf > config.pseudo_threshold
            target_update_count = int(pseudo_mask.sum().item())

            # Update student on shifted source data and pseudo-labeled target data
            pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
            source_labels = sample_source['label'].cuda(device, non_blocking=True)
            pixels_t, mask_t, position_t, extra_t = to_cuda(sample_target_strong, device)
            logits_target = None
            loss_target = 0.0
            reshaper_loss = pixels_s.sum() * 0.0
            reshaper_logs = {}
            compact_loss = pixels_s.sum() * 0.0
            compact_logs = {}
            dual_relation_loss = pixels_s.sum() * 0.0
            dual_relation_logs = {}
            loss_source_reshaped = pixels_s.sum() * 0.0
            if config.domain_specific_bn:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                spatial_feats_source_raw = student.spatial_encoder(pixels_s, mask_s, extra_s)
                temporal_feats_source_raw = student.temporal_encoder(
                    spatial_feats_source_raw,
                    position_s + source_to_target_shift,
                )
                logits_source_raw = student.decoder(temporal_feats_source_raw)
                logits_source = logits_source_raw
                if source_feature_reshaper is not None:
                    spatial_feats_source = source_feature_reshaper(
                        spatial_feats_source_raw.detach(),
                        positions=position_s,
                        labels=source_labels,
                    )
                    reshaper_loss, reshaper_logs = compute_source_feature_reshaper_regularization(
                        spatial_feats_source_raw.detach(),
                        spatial_feats_source,
                    )
                    compact_loss, compact_logs = compute_source_structure_loss(
                        spatial_feats_source,
                        position_s,
                        source_labels,
                        weight_tracker=phase_weight_tracker,
                        version=getattr(config, "source_structure_loss_version", "compactness"),
                        intra_trade_off=getattr(config, "source_structure_intra_trade_off", 1.0),
                        amplitude_trade_off=getattr(config, "source_structure_amplitude_trade_off", 0.25),
                        interphase_trade_off=getattr(config, "source_structure_interphase_trade_off", 0.25),
                        shape_trade_off=getattr(config, "source_structure_shape_trade_off", 0.15),
                        trend_trade_off=getattr(config, "source_structure_trend_trade_off", 0.05),
                        season_trade_off=getattr(config, "source_structure_season_trade_off", 0.02),
                        segment_inter_trade_off=getattr(config, "source_structure_segment_inter_trade_off", 0.02),
                        boundary_window_trade_off=getattr(config, "source_structure_boundary_window_trade_off", 0.02),
                        boundary_window_size=getattr(config, "source_structure_boundary_window_size", 2),
                        anchor_spatial_feats=spatial_feats_source_raw.detach(),
                        anchor_positions=position_s,
                    )
                    temporal_feats_source = student.temporal_encoder(
                        spatial_feats_source.detach(),
                        position_s + source_to_target_shift,
                    )
                    logits_source = student.decoder(temporal_feats_source)
                    loss_source_reshaped = criterion(logits_source, source_labels)
                    if getattr(config, "source_feature_dual_path", False):
                        dual_relation_loss, dual_relation_logs = compute_dual_path_relation_regularization(
                            logits_source_raw,
                            logits_source,
                            raw_temporal_feats=temporal_feats_source_raw,
                            reshaped_temporal_feats=temporal_feats_source,
                        )
                if target_update_count >= 2:  # at least 2 examples required for BN
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                    logits_target = student.forward(pixels_t[pseudo_mask], mask_t[pseudo_mask], position_t[pseudo_mask], extra_t[pseudo_mask])
            else:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                spatial_feats_source_raw = student.spatial_encoder(pixels_s, mask_s, extra_s)
                temporal_feats_source_raw = student.temporal_encoder(
                    spatial_feats_source_raw,
                    position_s + source_to_target_shift,
                )
                logits_source_raw = student.decoder(temporal_feats_source_raw)
                spatial_feats_source = spatial_feats_source_raw
                temporal_feats_source = temporal_feats_source_raw
                logits_source = logits_source_raw
                if source_feature_reshaper is not None:
                    spatial_feats_source = source_feature_reshaper(
                        spatial_feats_source_raw.detach(),
                        positions=position_s,
                        labels=source_labels,
                    )
                    reshaper_loss, reshaper_logs = compute_source_feature_reshaper_regularization(
                        spatial_feats_source_raw.detach(),
                        spatial_feats_source,
                    )
                    compact_loss, compact_logs = compute_source_structure_loss(
                        spatial_feats_source,
                        position_s,
                        source_labels,
                        weight_tracker=phase_weight_tracker,
                        version=getattr(config, "source_structure_loss_version", "compactness"),
                        intra_trade_off=getattr(config, "source_structure_intra_trade_off", 1.0),
                        amplitude_trade_off=getattr(config, "source_structure_amplitude_trade_off", 0.25),
                        interphase_trade_off=getattr(config, "source_structure_interphase_trade_off", 0.25),
                        shape_trade_off=getattr(config, "source_structure_shape_trade_off", 0.15),
                        trend_trade_off=getattr(config, "source_structure_trend_trade_off", 0.05),
                        season_trade_off=getattr(config, "source_structure_season_trade_off", 0.02),
                        segment_inter_trade_off=getattr(config, "source_structure_segment_inter_trade_off", 0.02),
                        boundary_window_trade_off=getattr(config, "source_structure_boundary_window_trade_off", 0.02),
                        boundary_window_size=getattr(config, "source_structure_boundary_window_size", 2),
                        anchor_spatial_feats=spatial_feats_source_raw.detach(),
                        anchor_positions=position_s,
                    )
                    temporal_feats_source = student.temporal_encoder(
                        spatial_feats_source.detach(),
                        position_s + source_to_target_shift,
                    )
                    logits_source = student.decoder(temporal_feats_source)
                    loss_source_reshaped = criterion(logits_source, source_labels)
                    if getattr(config, "source_feature_dual_path", False):
                        dual_relation_loss, dual_relation_logs = compute_dual_path_relation_regularization(
                            logits_source_raw,
                            logits_source,
                            raw_temporal_feats=temporal_feats_source_raw,
                            reshaped_temporal_feats=temporal_feats_source,
                        )

                if target_update_count > 0:
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                    logits_target = student.forward(
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        position_t[pseudo_mask],
                        extra_t[pseudo_mask],
                    )

            loss_source_raw = criterion(logits_source_raw, source_labels)
            if source_feature_reshaper is not None and getattr(config, "source_feature_dual_path", False):
                loss_source = (
                    loss_source_raw
                    + getattr(config, "source_feature_dual_cls_trade_off", 1.0) * loss_source_reshaped
                    + getattr(config, "source_feature_dual_relation_trade_off", 0.05) * dual_relation_loss
                )
            else:
                loss_source = criterion(logits_source, source_labels)
            if logits_target is not None:
                loss_target = criterion(logits_target, pseudo_targets[pseudo_mask])
            loss = loss_source + config.trade_off * loss_target
            if source_feature_reshaper is not None:
                loss = loss + compact_loss + getattr(config, "source_feature_reshaper_reg_trade_off", 0.0) * reshaper_loss

            # compute loss and backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)

            # Metrics
            loss_meter.update(loss.item())
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.3f}")
            all_labels.extend(sample_target_weak['label'].tolist())
            all_pseudo_labels.extend(pseudo_targets.tolist())
            all_pseudo_mask.extend(pseudo_mask.tolist())

            if step % config.log_step == 0:
                writer.add_scalar("train/loss", loss_meter.val, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/target_updates", target_update_count, global_step)
                if source_feature_reshaper is not None:
                    writer.add_scalar(
                        "train/source_feature_reshaper_reg_loss",
                        float(reshaper_logs.get("source_reshaper_reg_loss", 0.0)),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_phase_compactness_loss",
                        float(compact_logs.get("compactness_loss", 0.0)),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_structure_loss",
                        float(compact_logs.get("structure_loss", compact_logs.get("compactness_loss", 0.0))),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_cls_loss_raw",
                        float(loss_source_raw.detach().item()),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_cls_loss_reshaped",
                        float(loss_source_reshaped.detach().item()),
                        global_step,
                    )
                    if getattr(config, "source_feature_dual_path", False):
                        writer.add_scalar(
                            "train/source_dual_relation_loss",
                            float(dual_relation_logs.get("source_dual_relation_loss", 0.0)),
                            global_step,
                        )
                for name, value in reshaper_logs.items():
                    writer.add_scalar(f"train/{name}", value, global_step)
                for name, value in compact_logs.items():
                    if name != "compactness_loss":
                        writer.add_scalar(f"train/{name}", value, global_step)
                for name, value in dual_relation_logs.items():
                    writer.add_scalar(f"train/{name}", value, global_step)

            global_step += 1

        progress_bar.close()

        # Evaluate pseudo labels
        all_labels, all_pseudo_labels, all_pseudo_mask = np.array(all_labels), np.array(all_pseudo_labels), np.array(all_pseudo_mask)
        pseudo_count = all_pseudo_mask.sum()
        conf_pseudo_f1 = sklearn.metrics.f1_score(all_labels[all_pseudo_mask], all_pseudo_labels[all_pseudo_mask], average='macro', zero_division=0)
        print(f"Teacher pseudo label F1 {conf_pseudo_f1:.3f} (n={pseudo_count})")
        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        if config.run_validation and not getattr(config, "disable_validation_in_timematch", False):
            if config.output_student:
                student.eval()
                if source_feature_reshaper is not None:
                    source_feature_reshaper.eval()
                best_f1 = validation(
                    best_f1,
                    None,
                    config,
                    criterion,
                    device,
                    epoch,
                    student,
                    val_loader,
                    writer,
                    source_feature_reshaper=source_feature_reshaper,
                    apply_source_feature_reshaper=False,
                )
            else:
                teacher.eval()
                best_f1 = validation(
                    best_f1,
                    None,
                    config,
                    criterion,
                    device,
                    epoch,
                    teacher,
                    val_loader,
                    writer,
                    source_feature_reshaper=None,
                    apply_source_feature_reshaper=False,
                )

        if (
            getattr(config, "selection_metrics_out", None)
            and getattr(config, "selection_score_mode", "temporal_perturbation")
            in (
                "temporal_perturbation_trajectory",
                "temporal_perturbation_late_filter",
                "pure_perturbation_late_reject",
                "pure_perturbation_margin_tiebreak",
            )
        ):
            epoch_metrics = compute_selection_metrics(
                teacher=teacher,
                student=student,
                target_loader=target_loader_no_aug,
                device=device,
                target_to_source_shift=target_to_source_shift,
                num_classes=config.num_classes,
                pseudo_threshold=config.pseudo_threshold,
                source_class_distr=source_class_distr,
                shift_history=shift_history,
                max_temporal_shift=config.max_temporal_shift,
                config=config,
            )
            epoch_metrics["selection_epoch"] = int(epoch + 1)
            selection_metric_history.append(epoch_metrics)
            writer.add_scalar("selection/temporal_perturbation_score", epoch_metrics["selection_temporal_perturbation_score"], epoch)
            writer.add_scalar("selection/perturbation_score", epoch_metrics["selection_perturbation_score"], epoch)

    if getattr(config, "selection_metrics_out", None):
        if (
            getattr(config, "selection_score_mode", "temporal_perturbation")
            in (
                "temporal_perturbation_trajectory",
                "temporal_perturbation_late_filter",
                "pure_perturbation_late_reject",
                "pure_perturbation_margin_tiebreak",
            )
            and selection_metric_history
        ):
            metrics = dict(selection_metric_history[-1])
            metrics["selection_score_history"] = selection_metric_history
            _apply_trajectory_selection_score(metrics, selection_metric_history, config)
        else:
            metrics = compute_selection_metrics(
                teacher=teacher,
                student=student,
                target_loader=target_loader_no_aug,
                device=device,
                target_to_source_shift=target_to_source_shift,
                num_classes=config.num_classes,
                pseudo_threshold=config.pseudo_threshold,
                source_class_distr=source_class_distr,
                shift_history=shift_history,
                max_temporal_shift=config.max_temporal_shift,
                config=config,
            )
        metrics["selected_weights_checkpoint"] = getattr(config, "weights_checkpoint", "model.pt")
        metrics["target_to_source_shift"] = int(target_to_source_shift)
        metrics["epochs_ran"] = int(config.epochs)
        out_dir = os.path.dirname(config.selection_metrics_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(config.selection_metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(
            "Target selection metrics:",
            ", ".join(f"{key}={value}" for key, value in metrics.items() if key != "selected_weights_checkpoint"),
        )

    # Save model final model
    if config.output_student:
        checkpoint = {'state_dict': student.state_dict()}
        if source_feature_reshaper is not None:
            checkpoint['source_feature_reshaper_state_dict'] = source_feature_reshaper.state_dict()
        torch.save(checkpoint, best_model_path)
    else:
        checkpoint = {'state_dict': teacher.state_dict()}
        if source_feature_reshaper is not None:
            checkpoint['source_feature_reshaper_state_dict'] = source_feature_reshaper.state_dict()
        torch.save(checkpoint, best_model_path)


@torch.no_grad()
def compute_selection_metrics(
    teacher,
    student,
    target_loader,
    device,
    target_to_source_shift,
    num_classes,
    pseudo_threshold,
    source_class_distr,
    shift_history,
    max_temporal_shift,
    config,
):
    teacher.eval()
    student.eval()
    teacher_probs, student_probs = [], []
    time_mask_probs, temporal_jitter_probs, value_noise_probs = [], [], []
    indices = []
    max_batches = getattr(config, "selection_metric_batches", 200)

    for batch_idx, sample in enumerate(tqdm(target_loader, desc="computing selection metrics")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        indices.extend(sample["index"].tolist())
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        shifted_positions = _clamp_positions_for_model(teacher, positions + target_to_source_shift)
        teacher_logits = teacher.forward(pixels, valid_pixels, shifted_positions, extra)
        student_logits = student.forward(pixels, valid_pixels, positions, extra)
        teacher_probs.append(F.softmax(teacher_logits, dim=1).cpu())
        student_probs.append(F.softmax(student_logits, dim=1).cpu())

        perturbed_pixels = _apply_time_mean_mask(
            pixels,
            mask_p=getattr(config, "selection_time_mask_p", 0.15),
        )
        perturbed_logits = teacher.forward(perturbed_pixels, valid_pixels, shifted_positions, extra)
        time_mask_probs.append(F.softmax(perturbed_logits, dim=1).cpu())

        jittered_positions = _apply_temporal_jitter(
            teacher,
            shifted_positions,
            max_jitter=getattr(config, "selection_temporal_jitter", 3),
        )
        perturbed_logits = teacher.forward(pixels, valid_pixels, jittered_positions, extra)
        temporal_jitter_probs.append(F.softmax(perturbed_logits, dim=1).cpu())

        noisy_pixels = _apply_value_noise(
            pixels,
            noise_std=getattr(config, "selection_value_noise_std", 0.03),
        )
        perturbed_logits = teacher.forward(noisy_pixels, valid_pixels, shifted_positions, extra)
        value_noise_probs.append(F.softmax(perturbed_logits, dim=1).cpu())

    if not teacher_probs:
        raise RuntimeError("No target batches were available for selection metrics")

    indices = torch.as_tensor(indices)
    order = torch.argsort(indices)
    teacher_probs = torch.cat(teacher_probs, dim=0)[order]
    student_probs = torch.cat(student_probs, dim=0)[order]
    time_mask_probs = torch.cat(time_mask_probs, dim=0)[order]
    temporal_jitter_probs = torch.cat(temporal_jitter_probs, dim=0)[order]
    value_noise_probs = torch.cat(value_noise_probs, dim=0)[order]

    teacher_conf, pseudo_labels = teacher_probs.max(dim=1)
    student_labels = student_probs.argmax(dim=1)
    high_conf_mask = teacher_conf >= pseudo_threshold

    pred_entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-8))).sum(dim=1)
    normalized_entropy = pred_entropy.mean().item() / max(np.log(num_classes), 1e-8)
    coverage = high_conf_mask.float().mean().item()
    mean_confidence = teacher_conf.mean().item()
    agreement = (pseudo_labels == student_labels).float().mean().item()

    pseudo_counts = torch.bincount(pseudo_labels, minlength=num_classes).float()
    pseudo_dist = (pseudo_counts / pseudo_counts.sum().clamp_min(1.0)).numpy()
    class_entropy = float(-(pseudo_dist * np.log(pseudo_dist + 1e-8)).sum() / max(np.log(num_classes), 1e-8))
    max_class_fraction = float(pseudo_dist.max()) if len(pseudo_dist) else 1.0
    effective_class_fraction = float(np.exp(-(pseudo_dist * np.log(pseudo_dist + 1e-8)).sum()) / max(num_classes, 1))

    if high_conf_mask.any():
        high_conf_counts = torch.bincount(pseudo_labels[high_conf_mask], minlength=num_classes).float()
        high_conf_dist = (high_conf_counts / high_conf_counts.sum().clamp_min(1.0)).numpy()
        high_conf_class_entropy = float(
            -(high_conf_dist * np.log(high_conf_dist + 1e-8)).sum() / max(np.log(num_classes), 1e-8)
        )
        high_conf_max_class_fraction = float(high_conf_dist.max())
        high_conf_effective_class_fraction = float(
            np.exp(-(high_conf_dist * np.log(high_conf_dist + 1e-8)).sum()) / max(num_classes, 1)
        )
    else:
        high_conf_class_entropy = 0.0
        high_conf_max_class_fraction = 1.0
        high_conf_effective_class_fraction = 0.0

    source_dist = np.asarray(source_class_distr, dtype=np.float64)
    source_dist = source_dist / max(source_dist.sum(), 1e-8)
    pseudo_dist = np.asarray(pseudo_dist, dtype=np.float64)
    mixture = 0.5 * (source_dist + pseudo_dist)
    js_div = 0.5 * np.sum(source_dist * (np.log(source_dist + 1e-8) - np.log(mixture + 1e-8)))
    js_div += 0.5 * np.sum(pseudo_dist * (np.log(pseudo_dist + 1e-8) - np.log(mixture + 1e-8)))
    js_div = float(js_div / max(np.log(2.0), 1e-8))

    if shift_history:
        shifts = np.asarray(shift_history, dtype=np.float64)
        denom = max(float(max_temporal_shift), 1.0)
        shift_std_norm = float(np.std(shifts) / denom)
        shift_last_delta_norm = float(abs(shifts[-1] - shifts[-2]) / denom) if len(shifts) >= 2 else 0.0
        shift_stability = float(max(0.0, 1.0 - min(1.0, 0.5 * shift_std_norm + 0.5 * shift_last_delta_norm)))
    else:
        shift_std_norm = 0.0
        shift_last_delta_norm = 0.0
        shift_stability = 0.5

    class_balance_score = 0.60 * high_conf_class_entropy + 0.40 * class_entropy
    collapse_penalty = 0.50 * max_class_fraction + 0.50 * high_conf_max_class_fraction

    legacy_score = (
        getattr(config, "selection_score_coverage_weight", 0.25) * coverage
        + getattr(config, "selection_score_confidence_weight", 0.15) * mean_confidence
        + getattr(config, "selection_score_agreement_weight", 0.15) * agreement
        + getattr(config, "selection_score_class_balance_weight", 0.45) * class_balance_score
        + getattr(config, "selection_score_shift_stability_weight", 0.20) * shift_stability
        - getattr(config, "selection_score_entropy_weight", 0.10) * normalized_entropy
        - getattr(config, "selection_score_source_prior_weight", 0.25) * js_div
        - 0.20 * collapse_penalty
    )
    perturbation_metrics = _compute_perturbation_consistency(
        teacher_probs,
        {
            "time_mask": time_mask_probs,
            "temporal_jitter": temporal_jitter_probs,
            "value_noise": value_noise_probs,
        },
        num_classes,
    )
    perturbation_score = (
        0.50 * perturbation_metrics["selection_perturbation_prob_consistency"]
        + 0.50 * perturbation_metrics["selection_perturbation_label_agreement"]
    )
    robust_score = (
        getattr(config, "selection_perturbation_weight", 1.0) * perturbation_score
        + 0.15 * class_balance_score
        + 0.10 * shift_stability
        - getattr(config, "selection_collapse_penalty_weight", 0.35) * collapse_penalty
        - 0.10 * js_div
    )
    score_mode = getattr(config, "selection_score_mode", "temporal_perturbation")
    if score_mode.startswith("pure_perturbation"):
        score = perturbation_score
    elif score_mode == "legacy":
        score = legacy_score
    else:
        score = robust_score

    metrics = {
        "selection_score": float(score),
        "selection_score_mode": score_mode,
        "selection_legacy_score": float(legacy_score),
        "selection_temporal_perturbation_score": float(robust_score),
        "selection_perturbation_score": float(perturbation_score),
        "selection_coverage": float(coverage),
        "selection_mean_confidence": float(mean_confidence),
        "selection_teacher_student_agreement": float(agreement),
        "selection_prediction_entropy": float(normalized_entropy),
        "selection_class_entropy": float(class_entropy),
        "selection_high_conf_class_entropy": float(high_conf_class_entropy),
        "selection_effective_class_fraction": float(effective_class_fraction),
        "selection_high_conf_effective_class_fraction": float(high_conf_effective_class_fraction),
        "selection_max_class_fraction": float(max_class_fraction),
        "selection_high_conf_max_class_fraction": float(high_conf_max_class_fraction),
        "selection_source_prior_js": float(js_div),
        "selection_shift_stability": float(shift_stability),
        "selection_shift_std_norm": float(shift_std_norm),
        "selection_shift_last_delta_norm": float(shift_last_delta_norm),
    }
    metrics.update({key: float(value) for key, value in perturbation_metrics.items()})
    return metrics


def _clamp_positions_for_model(model, positions):
    temporal_encoder = model.temporal_encoder
    min_position = -temporal_encoder.max_temporal_shift
    max_position = temporal_encoder.positional_enc.num_embeddings - temporal_encoder.max_temporal_shift - 1
    return positions.clamp(min=min_position, max=max_position)


def _apply_time_mean_mask(pixels, mask_p):
    if mask_p <= 0:
        return pixels
    time_mean = pixels.mean(dim=1, keepdim=True)
    mask_shape = [pixels.shape[0], pixels.shape[1]] + [1] * (pixels.dim() - 2)
    time_mask = torch.rand(mask_shape, device=pixels.device) < mask_p
    return torch.where(time_mask, time_mean, pixels)


def _apply_temporal_jitter(model, positions, max_jitter):
    if max_jitter <= 0:
        return positions
    jitter = torch.randint(
        low=-int(max_jitter),
        high=int(max_jitter) + 1,
        size=positions.shape,
        device=positions.device,
        dtype=positions.dtype,
    )
    return _clamp_positions_for_model(model, positions + jitter)


def _apply_value_noise(pixels, noise_std):
    if noise_std <= 0:
        return pixels
    reduce_dims = tuple(range(1, pixels.dim()))
    scale = pixels.float().std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
    return pixels + torch.randn_like(pixels) * scale * float(noise_std)


def _normalized_js_divergence(probs_a, probs_b, num_classes):
    mixture = 0.5 * (probs_a + probs_b)
    kl_a = (probs_a * (torch.log(probs_a.clamp_min(1e-8)) - torch.log(mixture.clamp_min(1e-8)))).sum(dim=1)
    kl_b = (probs_b * (torch.log(probs_b.clamp_min(1e-8)) - torch.log(mixture.clamp_min(1e-8)))).sum(dim=1)
    return (0.5 * (kl_a + kl_b)).mean().item() / max(np.log(2.0), 1e-8)


def _compute_perturbation_consistency(clean_probs, perturbed_probs_by_name, num_classes):
    clean_labels = clean_probs.argmax(dim=1)
    label_agreements = []
    prob_consistencies = []
    metrics = {}
    for name, perturbed_probs in perturbed_probs_by_name.items():
        perturbed_labels = perturbed_probs.argmax(dim=1)
        label_agreement = (clean_labels == perturbed_labels).float().mean().item()
        prob_consistency = 1.0 - min(1.0, _normalized_js_divergence(clean_probs, perturbed_probs, num_classes))
        label_agreements.append(label_agreement)
        prob_consistencies.append(prob_consistency)
        metrics[f"selection_{name}_label_agreement"] = label_agreement
        metrics[f"selection_{name}_prob_consistency"] = prob_consistency
    metrics["selection_perturbation_label_agreement"] = float(np.mean(label_agreements))
    metrics["selection_perturbation_prob_consistency"] = float(np.mean(prob_consistencies))
    return metrics


def _apply_trajectory_selection_score(metrics, history, config):
    score_mode = getattr(config, "selection_score_mode", "temporal_perturbation_trajectory")
    if score_mode.startswith("pure_perturbation"):
        base_key = "selection_perturbation_score"
    else:
        base_key = "selection_temporal_perturbation_score"
    first_score = float(history[0].get(base_key, metrics.get(base_key, 0.0)))
    final_score = float(metrics.get(base_key, 0.0))
    penultimate_score = float(history[-2].get(base_key, final_score)) if len(history) >= 2 else first_score
    total_gain = max(final_score - first_score, 0.0)
    late_gain = max(final_score - penultimate_score, 0.0)
    if total_gain <= 1e-8:
        late_gain_ratio = 0.0
    else:
        late_gain_ratio = min(1.0, late_gain / (total_gain + 1e-8))
    alpha = float(getattr(config, "selection_trajectory_alpha", 0.30))
    late_gain_threshold = float(getattr(config, "selection_late_gain_threshold", 0.20))
    if score_mode == "pure_perturbation_late_reject":
        late_reject_threshold = float(getattr(config, "selection_late_reject_threshold", 0.80))
        is_rejected = late_gain_ratio > late_reject_threshold
        trajectory_score = final_score - (1.0 if is_rejected else 0.0)
        trajectory_multiplier = 1.0
        excess_late_gain = max(0.0, late_gain_ratio - late_reject_threshold)
        metrics["selection_late_reject_threshold"] = float(late_reject_threshold)
        metrics["selection_late_reject_applied"] = bool(is_rejected)
    elif score_mode == "pure_perturbation_margin_tiebreak":
        margin = float(getattr(config, "selection_margin_tiebreak", 0.01))
        trajectory_score = final_score + margin * float(metrics.get("selection_legacy_score", 0.0))
        trajectory_multiplier = 1.0
        excess_late_gain = 0.0
        metrics["selection_margin_tiebreak"] = float(margin)
    elif score_mode == "temporal_perturbation_late_filter":
        excess_late_gain = max(0.0, late_gain_ratio - late_gain_threshold)
        trajectory_multiplier = max(0.0, 1.0 - alpha * excess_late_gain)
        trajectory_score = final_score * trajectory_multiplier
    else:
        excess_late_gain = late_gain_ratio
        trajectory_multiplier = max(0.0, 1.0 - alpha * late_gain_ratio)
        trajectory_score = final_score * trajectory_multiplier
    metrics["selection_score_mode"] = score_mode
    metrics["selection_score"] = float(trajectory_score)
    metrics["selection_trajectory_base_score"] = float(final_score)
    metrics["selection_trajectory_first_score"] = float(first_score)
    metrics["selection_trajectory_penultimate_score"] = float(penultimate_score)
    metrics["selection_trajectory_total_gain"] = float(total_gain)
    metrics["selection_trajectory_late_gain"] = float(late_gain)
    metrics["selection_trajectory_late_gain_ratio"] = float(late_gain_ratio)
    metrics["selection_trajectory_late_gain_threshold"] = float(late_gain_threshold)
    metrics["selection_trajectory_excess_late_gain"] = float(excess_late_gain)
    metrics["selection_trajectory_multiplier"] = float(trajectory_multiplier)
    metrics["selection_trajectory_alpha"] = float(alpha)


def estimate_class_distribution(labels, num_classes):
    return np.bincount(labels, minlength=num_classes) / len(labels)

def kl_divergence(actual, estimated):
    return np.sum(actual * (np.log(actual + 1e-5) - np.log(estimated + 1e-5)))

@torch.no_grad()
def update_ema_variables(model, ema, decay=0.99):
    for ema_v, model_v in zip(ema.state_dict().values(), model.state_dict().values()):
        ema_v.copy_(decay * ema_v + (1. - decay) * model_v)


def get_data_loaders(splits, config, balance_source=True):
    weak_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        Normalize(),
        ToTensor(),
    ])

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, strong_aug,
            indices=splits[config.source]['train'],
            closed_set=getattr(config, 'closed_set', False),)

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    target_dataset = PixelSetData(config.data_root, config.target,
            config.classes, None,
            indices=splits[config.target]['train'],
            closed_set=getattr(config, 'closed_set', False))

    strong_dataset = deepcopy(target_dataset)
    strong_dataset.transform = strong_aug
    weak_dataset = deepcopy(target_dataset)
    weak_dataset.transform = weak_aug
    target_dataset_weak_strong = TupleDataset(weak_dataset, strong_dataset)

    no_aug_dataset = deepcopy(target_dataset)
    no_aug_dataset.transform = weak_aug
    # For shift estimation
    target_loader_no_aug = data.DataLoader(
        no_aug_dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # For mean teacher training
    target_loader_weak_strong = data.DataLoader(
        target_dataset_weak_strong,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')
    print(f'size of target dataset: {len(target_dataset)} ({len(target_loader_weak_strong)} batches)')

    return source_loader, target_loader_no_aug, target_loader_weak_strong


class TupleDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.weak = dataset1
        self.strong = dataset2
        assert len(dataset1) == len(dataset2)
        self.len = len(dataset1)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.weak[index], self.strong[index])


@torch.no_grad()
def estimate_temporal_shift(model, target_loader, device, class_distribution=None, min_shift=-60, max_shift=60, sample_size=100, shift_estimator='IS'):
    shifts = list(range(min_shift, max_shift + 1))
    model.eval()
    if sample_size is None:
        sample_size = len(target_loader)

    target_iter = iter(target_loader)
    shift_softmaxes, labels = [], []
    for _ in tqdm(range(sample_size), desc=f'Estimating shift between [{min_shift}, {max_shift}]'):
        sample = next(target_iter)
        labels.extend(sample['label'].tolist())
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        spatial_feats = model.spatial_encoder.forward(pixels, valid_pixels, extra)
        shift_logits = torch.stack([model.decoder(model.temporal_encoder(spatial_feats, positions + shift)) for shift in shifts], dim=1)
        shift_probs = F.softmax(shift_logits, dim=2)
        shift_softmaxes.append(shift_probs)
    shift_softmaxes = torch.cat(shift_softmaxes).cpu().numpy()  # (N, n_shifts, n_classes)
    labels = np.array(labels)
    shift_predictions = np.argmax(shift_softmaxes, axis=2)  # (N, n_shifts)

    # shift_f1_scores = [f1_score(labels, shift_predictions, num_classes) for shift_predictions in all_shift_predictions]
    shift_acc_scores = [(labels == predictions).mean() for predictions in np.moveaxis(shift_predictions, 0, 1)]
    print(f"Most accurate shift {shifts[np.argmax(shift_acc_scores)]} with {np.max(shift_acc_scores):.3f}")

    p_yx = shift_softmaxes # (N, n_shifts, n_classes)
    p_y = shift_softmaxes.mean(axis=0)  # (n_shifts, n_classes)


    if shift_estimator == 'IS':
        inception_score = np.mean(np.sum(p_yx * (np.log(p_yx + 1e-5) - np.log(p_y[np.newaxis] + 1e-5)), axis=2), axis=0)  # (n_shifts)

        shift_indices_ranked = np.argsort(inception_score)[::-1]  # max is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best Inception Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")
        return best_shift

    elif shift_estimator == 'ENT':
        entropy_score = -np.mean(np.sum(p_yx * np.log(p_yx + 1e-5), axis=2), axis=0)  # (n_shifts)
        shift_indices_ranked = np.argsort(entropy_score)  # min is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best Entropy Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")
        return best_shift

    elif shift_estimator == 'AM':
        assert class_distribution is not None, 'Target class distribution required to compute AM score'

        # estimate class distribution
        one_hot_p_y = np.zeros_like(p_y)
        for i in range(len(shifts)):
            one_hot = np.zeros((shift_softmaxes.shape[0], shift_softmaxes.shape[-1]))  # (n, classes)
            one_hot[np.arange(one_hot.shape[0]), shift_predictions[:, i]] = 1
            one_hot_p_y[i] = one_hot.mean(axis=0)

        c_train = class_distribution
        # kl_d = np.sum(c_train * (np.log(c_train + 1e-5) - np.log(p_y + 1e-5)), axis=1) # soft class distr
        kl_d = np.sum(c_train * (np.log(c_train + 1e-5) - np.log(one_hot_p_y + 1e-5)), axis=1)
        entropy = np.mean(np.sum(-p_yx * np.log(p_yx + 1e-5), axis=2), axis=0)
        am = kl_d + entropy
        shift_indices_ranked = np.argsort(am)  # min is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best AM Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")

        return best_shift
    elif shift_estimator == 'ACC':  # for upperbound comparison
        shift_indices_ranked = np.argsort(shift_acc_scores)[::-1]  # max is best
        return shifts[np.argmax(shift_acc_scores)]
    else:
        raise NotImplementedError




@torch.no_grad()
def get_pseudo_labels(model, data_loader, device, best_shift, n=500):
    model.eval()
    pseudo_softmaxes = []
    indices = []
    for i, sample in enumerate(tqdm(data_loader, "computing pseudo labels")):
        if n is not None and i == n:
            break
        indices.extend(sample["index"].tolist())

        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        logits = model.forward(pixels, valid_pixels, positions + best_shift, extra)
        probs = F.softmax(logits, dim=1).cpu()
        pseudo_softmaxes.extend(probs.tolist())

    indices = torch.as_tensor(indices)
    pseudo_softmaxes = torch.as_tensor(pseudo_softmaxes)
    pseudo_softmaxes = pseudo_softmaxes[torch.argsort(indices)]

    return pseudo_softmaxes
