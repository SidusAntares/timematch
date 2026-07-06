from torch.utils.data.sampler import WeightedRandomSampler
import csv
import json
import os
import sys
import time
import sklearn.metrics
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    RandomTemporalShift,
    Identity,
)
from ideas.adaptive_stage_contrast import compute_adaptive_stage_contrast_loss
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


def _format_diag_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return ""
        return f"{float(value):.6f}"
    return str(value)


def _append_diag_tsv(path, row, fields):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: _format_diag_value(row.get(field)) for field in fields})


def _pseudo_metrics(labels, pseudo_labels, pseudo_conf, pseudo_mask, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
    pseudo_conf = np.asarray(pseudo_conf, dtype=np.float64)
    pseudo_mask = np.asarray(pseudo_mask, dtype=bool)
    if labels.size == 0:
        return {
            "teacher_pseudo_coverage": 0.0,
            "teacher_pseudo_confidence_mean": 0.0,
            "teacher_pseudo_class_entropy": 0.0,
            "teacher_pseudo_effective_class_count": 0.0,
            "teacher_pseudo_class_distribution_json": "[]",
            "teacher_target_macro_f1_offline": 0.0,
        }
    if pseudo_mask.any():
        masked_labels = pseudo_labels[pseudo_mask]
        masked_conf = pseudo_conf[pseudo_mask]
    else:
        masked_labels = np.asarray([], dtype=np.int64)
        masked_conf = np.asarray([], dtype=np.float64)
    counts, probs, entropy, effective = _distribution_summary(masked_labels, num_classes)
    return {
        "teacher_pseudo_coverage": float(pseudo_mask.mean()),
        "teacher_pseudo_confidence_mean": float(masked_conf.mean()) if masked_conf.size else 0.0,
        "teacher_pseudo_class_entropy": entropy,
        "teacher_pseudo_effective_class_count": effective,
        "teacher_pseudo_class_distribution_json": json.dumps(probs.tolist(), ensure_ascii=True),
        "teacher_target_macro_f1_offline": float(
            sklearn.metrics.f1_score(labels, pseudo_labels, average="macro", zero_division=0)
        ),
    }


@torch.no_grad()
def _forward_shift_ensemble(model, pixels, mask, positions, extra, shifts):
    probs = []
    for shift in shifts:
        _check_temporal_index_range(model, positions, int(shift), "target")
        logits = model.forward(pixels, mask, positions + int(shift), extra)
        probs.append(F.softmax(logits, dim=1))
    return torch.stack(probs, dim=0).mean(dim=0)


def train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(splits, config, config.balance_source)

    # Setup model
    pretrained_path = f"{config.weights}/fold_{fold_num}"
    pretrained_weights = torch.load(f"{pretrained_path}/model.pt", weights_only=False)["state_dict"]
    student.load_state_dict(pretrained_weights)
    teacher = deepcopy(student)
    student.to(device)
    teacher.to(device)

    # Training setup
    global_step, best_f1 = 0, 0
    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_loss_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    steps_per_epoch = config.steps_per_epoch

    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)

    source_iter = iter(cycle(source_loader))
    target_iter = iter(cycle(target_loader))
    min_shift, max_shift = -config.max_temporal_shift, config.max_temporal_shift
    target_to_source_shift = 0
    target_to_source_topk_shifts = [0]
    last_shift_details = None
    shift_policy = getattr(config, "timematch_shift_policy", "original_timematch")
    if shift_policy == "original":
        shift_policy = "original_timematch"
    diagnostic_fields = [
        "task",
        "source",
        "target",
        "seed",
        "config",
        "epoch",
        "estimated_shift_t_to_s",
        "estimated_shift_s_to_t",
        "topk_shifts_json",
        "is_score_top1",
        "am_score_top1",
        "is_top1_top2_margin",
        "am_top1_top2_margin",
        "is_curve_sharpness",
        "am_curve_sharpness",
        "teacher_pseudo_coverage",
        "teacher_pseudo_confidence_mean",
        "teacher_pseudo_class_entropy",
        "teacher_pseudo_effective_class_count",
        "teacher_pseudo_class_distribution_json",
        "source_loss",
        "target_loss",
        "total_loss",
        "teacher_target_macro_f1_offline",
        "student_target_macro_f1_offline",
    ]
    stage_contrast_enabled = float(getattr(config, "stage_contrast_trade_off", 0.0)) > 0.0
    stage_contrast_fields = [
        "task",
        "source",
        "target",
        "seed",
        "epoch",
        "step",
        "stage_loss",
        "stage_valid_queries",
        "stage_skipped_no_positive",
        "stage_skipped_no_negative",
        "stage_positive_count_mean",
        "stage_negative_class_count_mean",
        "stage_pseudo_coverage",
        "stage_target_conf_mean",
        "stage_lengths_mean",
        "stage_lengths_std",
        "stage_dp_cost_mean",
        "correspondence_entropy",
        "correspondence_valid_candidate_mean",
        "correspondence_mean_time_gap",
        "correspondence_fallback_count",
        "correspondence_max_weight_mean",
        "stage_trade_off",
        "stage_count",
        "temperature",
        "stage_time_radius",
        "stage_time_temperature",
        "stage_contrast_backend",
        "stage_loss_compute_time_ms",
        "source_valid_class_count",
        "source_valid_class_stage_count",
        "correspondence_fallback_ratio",
        "correspondence_valid_class_mean",
    ]

    # To evaluate how well we estimate class distribution
    target_labels = target_loader_no_aug.dataset.get_labels()
    actual_class_distr = estimate_class_distribution(target_labels, config.num_classes)

    # estimate an initial guess for shift using Inception Score unless the diagnostic policy disables it
    if config.estimate_shift and shift_policy != "no_shift":
        shift_estimator = 'IS' if config.shift_estimator == 'AM' else config.shift_estimator
        if shift_policy == "oracle_scalar_shift_diagnostic":
            shift_estimator = "F1"
        last_shift_details = estimate_temporal_shift_details(
            teacher,
            target_loader_no_aug,
            device,
            min_shift=min_shift,
            max_shift=max_shift,
            sample_size=config.sample_size,
            shift_estimator=shift_estimator,
            num_classes=config.num_classes,
            pseudo_threshold=config.pseudo_threshold,
            topk=getattr(config, "timematch_topk_shifts", 3),
        )
        target_to_source_shift = int(last_shift_details["best_shift"])
        target_to_source_topk_shifts = [int(x) for x in last_shift_details["topk_shifts"]]
        print(
            f"Initial {shift_estimator} shift {target_to_source_shift}; "
            f"topk={target_to_source_topk_shifts}"
        )
        if target_to_source_shift >= 0:
            min_shift = 0
        else:
            max_shift = 0

        # Use estimated shift to get initial pseudo labels
        pseudo_softmaxes = get_pseudo_labels(teacher, target_loader_no_aug, device, target_to_source_shift, n=None)
        all_pseudo_labels = torch.max(pseudo_softmaxes, dim=1)[1]
    else:
        pseudo_softmaxes = get_pseudo_labels(teacher, target_loader_no_aug, device, 0, n=None)
        all_pseudo_labels = torch.max(pseudo_softmaxes, dim=1)[1]

    source_to_target_shift = 0
    for epoch in range(config.epochs):
        progress_bar = tqdm(
            range(steps_per_epoch),
            desc=f"TimeMatch Epoch {epoch + 1}/{config.epochs}",
            disable=not sys.stderr.isatty(),
        )
        loss_meter = AverageMeter()

        if config.estimate_shift and shift_policy not in {"no_shift", "fixed_initial_shift", "oracle_scalar_shift_diagnostic"}:
            estimated_class_distr = estimate_class_distribution(all_pseudo_labels, config.num_classes)
            writer.add_scalar("train/kl_d", kl_divergence(actual_class_distr, estimated_class_distr), epoch)
            last_shift_details = estimate_temporal_shift_details(
                teacher,
                target_loader_no_aug,
                device,
                estimated_class_distr,
                min_shift=min_shift,
                max_shift=max_shift,
                sample_size=config.sample_size,
                shift_estimator=config.shift_estimator,
                num_classes=config.num_classes,
                pseudo_threshold=config.pseudo_threshold,
                topk=getattr(config, "timematch_topk_shifts", 3),
            )
            target_to_source_shift = int(last_shift_details["best_shift"])
            target_to_source_topk_shifts = [int(x) for x in last_shift_details["topk_shifts"]]
            if epoch == 0:
                if config.shift_source:
                    source_to_target_shift = -target_to_source_shift
                else:
                    source_to_target_shift = 0
                min_shift, max_shift = min(target_to_source_shift, 0), max(0, target_to_source_shift)
            writer.add_scalar("train/temporal_shift", target_to_source_shift, epoch)
        elif epoch == 0:
            if shift_policy == "no_shift":
                target_to_source_shift = 0
                target_to_source_topk_shifts = [0]
                source_to_target_shift = 0
                min_shift, max_shift = 0, 0
            elif config.shift_source:
                source_to_target_shift = -target_to_source_shift
                min_shift, max_shift = min(target_to_source_shift, 0), max(0, target_to_source_shift)
            else:
                source_to_target_shift = 0

        student.train()
        teacher.eval()  # don't update BN or use dropout for teacher

        all_labels, all_pseudo_labels, all_pseudo_conf, all_pseudo_mask = [], [], [], []
        stage_loss_meter = AverageMeter()
        stage_valid_queries_meter = AverageMeter()
        stage_pseudo_coverage_meter = AverageMeter()
        stage_target_conf_meter = AverageMeter()
        stage_lengths_meter = AverageMeter()
        stage_corr_entropy_meter = AverageMeter()
        stage_corr_gap_meter = AverageMeter()
        stage_corr_fallback_meter = AverageMeter()
        for step in progress_bar:
            sample_source, (sample_target_weak, sample_target_strong) = next(source_iter), next(target_iter)

            # Get pseudo labels from teacher
            pixels_t_weak, mask_t_weak, position_t_weak, extra_t_weak = to_cuda(sample_target_weak, device)
            with torch.no_grad():
                if shift_policy == "topk_shift_ensemble_diagnostic":
                    teacher_preds = _forward_shift_ensemble(
                        teacher,
                        pixels_t_weak,
                        mask_t_weak,
                        position_t_weak,
                        extra_t_weak,
                        target_to_source_topk_shifts,
                    )
                else:
                    teacher_preds = F.softmax(
                        teacher.forward(
                            pixels_t_weak,
                            mask_t_weak,
                            position_t_weak + target_to_source_shift,
                            extra_t_weak,
                        ),
                        dim=1,
                    )
            pseudo_conf, pseudo_targets = torch.max(teacher_preds, dim=1)
            pseudo_mask = pseudo_conf > config.pseudo_threshold

            # Update student on shifted source data and pseudo-labeled target data
            pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
            source_labels = sample_source['label'].cuda(device, non_blocking=True)
            pixels_t, mask_t, position_t, extra_t = to_cuda(sample_target_strong, device)
            logits_target = None
            loss_target = 0.0
            if config.domain_specific_bn:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                logits_source = student.forward(pixels_s, mask_s, position_s + source_to_target_shift, extra_s)
                if len(torch.nonzero(pseudo_mask)) >= 2:  # at least 2 examples required for BN
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                    logits_target = student.forward(pixels_t[pseudo_mask], mask_t[pseudo_mask], position_t[pseudo_mask], extra_t[pseudo_mask])
            else:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                pixels = torch.cat([pixels_s, pixels_t[pseudo_mask]])
                mask = torch.cat([mask_s, mask_t[pseudo_mask]])
                position = torch.cat([position_s + source_to_target_shift, position_t[pseudo_mask]])
                extra = torch.cat([extra_s, extra_t[pseudo_mask]])
                logits = student.forward(pixels, mask, position, extra)
                logits_source, logits_target = logits[:config.batch_size], logits[config.batch_size:]

            loss_source = criterion(logits_source, source_labels)
            if logits_target is not None:
                loss_target = criterion(logits_target, pseudo_targets[pseudo_mask])
            loss = loss_source + config.trade_off * loss_target
            stage_loss = None
            stage_logs = {}
            if stage_contrast_enabled:
                if str(getattr(config, "stage_contrast_feature_kind", "spatial")).lower() != "spatial":
                    raise ValueError(
                        "v31_adaptive_stage_shift_contrast currently only supports "
                        "stage_contrast_feature_kind='spatial'"
                    )
                stage_threshold = getattr(config, "stage_contrast_pseudo_threshold", None)
                if stage_threshold is None:
                    stage_target_mask = pseudo_mask
                else:
                    stage_target_mask = pseudo_conf >= float(stage_threshold)
                source_stage_features = student.spatial_encoder(pixels_s, mask_s, extra_s)
                target_stage_features = student.spatial_encoder(pixels_t, mask_t, extra_t)
                if bool(getattr(config, "stage_contrast_debug", False)) and source_stage_features.is_cuda:
                    torch.cuda.synchronize(source_stage_features.device)
                stage_start_time = time.perf_counter()
                stage_loss, stage_logs = compute_adaptive_stage_contrast_loss(
                    source_stage_features,
                    position_s,
                    source_labels,
                    target_stage_features,
                    position_t,
                    pseudo_targets,
                    pseudo_conf,
                    stage_target_mask,
                    target_to_source_shift,
                    num_stages=getattr(config, "stage_contrast_stage_count", 6),
                    stage_min_len=getattr(config, "stage_min_len", 2),
                    stage_partition_mode=getattr(config, "stage_partition_mode", "feature_change_dp"),
                    stage_time_radius=getattr(config, "stage_time_radius", 30.0),
                    stage_time_temperature=getattr(config, "stage_time_temperature", 10.0),
                    temperature=getattr(config, "stage_contrast_temperature", 0.1),
                    normalize=True,
                    num_classes=getattr(config, "num_classes", None),
                    backend=getattr(config, "stage_contrast_backend", "class_prototype_fast"),
                )
                if bool(getattr(config, "stage_contrast_debug", False)) and source_stage_features.is_cuda:
                    torch.cuda.synchronize(source_stage_features.device)
                stage_logs["stage_loss_compute_time_ms"] = (time.perf_counter() - stage_start_time) * 1000.0
                loss = loss + float(config.stage_contrast_trade_off) * stage_loss
                stage_loss_meter.update(stage_logs.get("stage_contrast_loss", 0.0))
                stage_valid_queries_meter.update(stage_logs.get("stage_valid_queries", 0.0))
                stage_pseudo_coverage_meter.update(stage_logs.get("stage_pseudo_coverage", 0.0))
                stage_target_conf_meter.update(stage_logs.get("stage_target_conf_mean", 0.0))
                stage_lengths_meter.update(stage_logs.get("source_stage_lengths_mean", 0.0))
                stage_corr_entropy_meter.update(stage_logs.get("correspondence_entropy", 0.0))
                stage_corr_gap_meter.update(stage_logs.get("correspondence_mean_time_gap", 0.0))
                stage_corr_fallback_meter.update(stage_logs.get("correspondence_fallback_count", 0.0))

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
            all_pseudo_conf.extend(pseudo_conf.tolist())
            all_pseudo_mask.extend(pseudo_mask.tolist())

            if step % config.log_step == 0:
                writer.add_scalar("train/loss", loss_meter.val, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/target_updates", len(torch.nonzero(pseudo_mask)), global_step)
                if stage_contrast_enabled:
                    writer.add_scalar("train/stage_contrast_loss", stage_loss_meter.val, global_step)
                    writer.add_scalar("train/stage_valid_queries", stage_valid_queries_meter.val, global_step)
                    writer.add_scalar("train/stage_pseudo_coverage", stage_pseudo_coverage_meter.val, global_step)
                    writer.add_scalar("train/stage_target_conf_mean", stage_target_conf_meter.val, global_step)
                    writer.add_scalar("train/stage_lengths_mean", stage_lengths_meter.val, global_step)
                    writer.add_scalar("train/correspondence_entropy", stage_corr_entropy_meter.val, global_step)
                    writer.add_scalar("train/correspondence_mean_time_gap", stage_corr_gap_meter.val, global_step)
                    writer.add_scalar("train/correspondence_fallback_count", stage_corr_fallback_meter.val, global_step)
                    stage_row = {
                        "task": getattr(config, "timematch_diagnostic_task", "")
                        or f"{config.source.split('/')[1]}_to_{config.target.split('/')[1]}",
                        "source": config.source,
                        "target": config.target,
                        "seed": config.seed,
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "stage_loss": stage_logs.get("stage_contrast_loss", 0.0),
                        "stage_valid_queries": stage_logs.get("stage_valid_queries", 0.0),
                        "stage_skipped_no_positive": stage_logs.get("stage_skipped_no_positive", 0.0),
                        "stage_skipped_no_negative": stage_logs.get("stage_skipped_no_negative", 0.0),
                        "stage_positive_count_mean": stage_logs.get("stage_positive_count_mean", 0.0),
                        "stage_negative_class_count_mean": stage_logs.get("stage_negative_class_count_mean", 0.0),
                        "stage_pseudo_coverage": stage_logs.get("stage_pseudo_coverage", 0.0),
                        "stage_target_conf_mean": stage_logs.get("stage_target_conf_mean", 0.0),
                        "stage_lengths_mean": stage_logs.get("source_stage_lengths_mean", 0.0),
                        "stage_lengths_std": stage_logs.get("source_stage_lengths_std", 0.0),
                        "stage_dp_cost_mean": stage_logs.get("source_stage_dp_cost_mean", 0.0),
                        "correspondence_entropy": stage_logs.get("correspondence_entropy", 0.0),
                        "correspondence_valid_candidate_mean": stage_logs.get("correspondence_valid_candidate_mean", 0.0),
                        "correspondence_mean_time_gap": stage_logs.get("correspondence_mean_time_gap", 0.0),
                        "correspondence_fallback_count": stage_logs.get("correspondence_fallback_count", 0.0),
                        "correspondence_max_weight_mean": stage_logs.get("correspondence_max_weight_mean", 0.0),
                        "stage_trade_off": getattr(config, "stage_contrast_trade_off", 0.0),
                        "stage_count": getattr(config, "stage_contrast_stage_count", 6),
                        "temperature": getattr(config, "stage_contrast_temperature", 0.1),
                        "stage_time_radius": getattr(config, "stage_time_radius", 30.0),
                        "stage_time_temperature": getattr(config, "stage_time_temperature", 10.0),
                        "stage_contrast_backend": getattr(config, "stage_contrast_backend", "class_prototype_fast"),
                        "stage_loss_compute_time_ms": stage_logs.get("stage_loss_compute_time_ms", 0.0),
                        "source_valid_class_count": stage_logs.get("source_valid_class_count", 0.0),
                        "source_valid_class_stage_count": stage_logs.get("source_valid_class_stage_count", 0.0),
                        "correspondence_fallback_ratio": stage_logs.get("correspondence_fallback_ratio", 0.0),
                        "correspondence_valid_class_mean": stage_logs.get("correspondence_valid_class_mean", 0.0),
                    }
                    _append_diag_tsv(
                        getattr(config, "stage_contrast_log_path", ""),
                        stage_row,
                        stage_contrast_fields,
                    )

            global_step += 1

        progress_bar.close()
        if stage_contrast_enabled:
            print(
                "ADAPTIVE_STAGE_CONTRAST|"
                f"epoch={epoch + 1}|"
                f"loss={stage_loss_meter.avg:.6f}|"
                f"valid_queries={stage_valid_queries_meter.avg:.3f}|"
                f"pseudo_coverage={stage_pseudo_coverage_meter.avg:.6f}|"
                f"target_conf={stage_target_conf_meter.avg:.6f}|"
                f"stage_len_mean={stage_lengths_meter.avg:.6f}|"
                f"corr_entropy={stage_corr_entropy_meter.avg:.6f}|"
                f"corr_time_gap={stage_corr_gap_meter.avg:.6f}|"
                f"fallback={stage_corr_fallback_meter.avg:.3f}|"
                f"backend={getattr(config, 'stage_contrast_backend', 'class_prototype_fast')}"
            )

        # Evaluate pseudo labels
        all_labels, all_pseudo_labels = np.array(all_labels), np.array(all_pseudo_labels)
        all_pseudo_conf, all_pseudo_mask = np.array(all_pseudo_conf), np.array(all_pseudo_mask)
        pseudo_count = all_pseudo_mask.sum()
        if pseudo_count > 0:
            conf_pseudo_f1 = sklearn.metrics.f1_score(
                all_labels[all_pseudo_mask],
                all_pseudo_labels[all_pseudo_mask],
                average='macro',
                zero_division=0,
            )
        else:
            conf_pseudo_f1 = 0.0
        print(f"Teacher pseudo label F1 {conf_pseudo_f1:.3f} (n={pseudo_count})")
        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        pseudo_summary = _pseudo_metrics(
            all_labels,
            all_pseudo_labels,
            all_pseudo_conf,
            all_pseudo_mask,
            config.num_classes,
        )
        if last_shift_details is None:
            shift_diag = {}
        else:
            best_idx = int(last_shift_details.get("best_shift_idx", 0))
            shift_diag = {
                "is_score_top1": float(last_shift_details["is_scores"][last_shift_details["best_is_idx"]]),
                "am_score_top1": float(last_shift_details["am_scores"][last_shift_details["best_am_idx"]]),
                "is_top1_top2_margin": float(last_shift_details.get("is_top1_top2_margin", 0.0)),
                "am_top1_top2_margin": float(last_shift_details.get("am_top1_top2_margin", 0.0)),
                "is_curve_sharpness": float(last_shift_details.get("is_curve_sharpness", 0.0)),
                "am_curve_sharpness": float(last_shift_details.get("am_curve_sharpness", 0.0)),
            }
        diag_row = {
            "task": getattr(config, "timematch_diagnostic_task", "")
            or f"{config.source.split('/')[1]}_to_{config.target.split('/')[1]}",
            "source": config.source,
            "target": config.target,
            "seed": config.seed,
            "config": shift_policy,
            "epoch": epoch + 1,
            "estimated_shift_t_to_s": target_to_source_shift,
            "estimated_shift_s_to_t": source_to_target_shift,
            "topk_shifts_json": json.dumps(target_to_source_topk_shifts, ensure_ascii=True),
            "source_loss": loss_source.item() if hasattr(loss_source, "item") else float(loss_source),
            "target_loss": loss_target.item() if hasattr(loss_target, "item") else float(loss_target),
            "total_loss": loss_meter.avg,
            "student_target_macro_f1_offline": "",
            **shift_diag,
            **pseudo_summary,
        }
        print(
            "TIMEMATCH_SHIFT_TRAJECTORY|"
            + "|".join(f"{field}={_format_diag_value(diag_row.get(field))}" for field in diagnostic_fields)
        )
        _append_diag_tsv(
            getattr(config, "timematch_diagnostic_log_path", ""),
            diag_row,
            diagnostic_fields,
        )

        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        if config.run_validation:
            if config.output_student:
                student.eval()
                best_f1 = validation(best_f1, None, config, criterion, device, epoch, student, val_loader, writer)
            else:
                teacher.eval()
                best_f1 = validation(best_f1, None, config, criterion, device, epoch, teacher, val_loader, writer)

    # Save model final model 
    if config.output_student:
        torch.save({'state_dict': student.state_dict()}, best_model_path)
    else:
        torch.save({'state_dict': teacher.state_dict()}, best_model_path)

def estimate_class_distribution(labels, num_classes):
    return np.bincount(labels, minlength=num_classes) / len(labels)

def kl_divergence(actual, estimated):
    return np.sum(actual * (np.log(actual + 1e-5) - np.log(estimated + 1e-5)))


def _safe_entropy(probs, axis=-1):
    return -np.sum(probs * np.log(probs + 1e-12), axis=axis)


def _distribution_summary(labels, num_classes):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        probs = np.zeros(num_classes, dtype=np.float64)
    else:
        probs = counts / total
    entropy = float(_safe_entropy(probs, axis=0))
    norm_entropy = entropy / float(np.log(num_classes + 1e-12)) if num_classes > 1 else 0.0
    effective = float(np.exp(entropy))
    return counts, probs, norm_entropy, effective


def _top_margin(scores, higher_is_better=True):
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 2:
        return 0.0
    order = np.argsort(values)
    if higher_is_better:
        order = order[::-1]
        return float(values[order[0]] - values[order[1]])
    return float(values[order[1]] - values[order[0]])


def _curve_sharpness(scores, best_idx, higher_is_better=True):
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if higher_is_better:
        return float(values[best_idx] - values.mean())
    return float(values.mean() - values[best_idx])


@torch.no_grad()
def collect_shift_softmaxes(model, target_loader, device, min_shift=-60, max_shift=60, sample_size=100):
    shifts = list(range(min_shift, max_shift + 1))
    model.eval()
    if sample_size is None:
        sample_size = len(target_loader)

    target_iter = iter(target_loader)
    shift_softmaxes, labels = [], []
    for _ in tqdm(
        range(sample_size),
        desc=f'Estimating shift between [{min_shift}, {max_shift}]',
        disable=not sys.stderr.isatty(),
    ):
        try:
            sample = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            sample = next(target_iter)
        labels.extend(sample['label'].tolist())
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        spatial_feats = model.spatial_encoder.forward(pixels, valid_pixels, extra)
        shift_logits = torch.stack(
            [model.decoder(model.temporal_encoder(spatial_feats, positions + shift)) for shift in shifts],
            dim=1,
        )
        shift_probs = F.softmax(shift_logits, dim=2)
        shift_softmaxes.append(shift_probs)
    shift_softmaxes = torch.cat(shift_softmaxes).cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    return shifts, shift_softmaxes, labels


def score_shift_softmaxes(shifts, shift_softmaxes, labels, num_classes, class_distribution=None, pseudo_threshold=0.9):
    shift_predictions = np.argmax(shift_softmaxes, axis=2)
    p_yx = shift_softmaxes
    p_y = shift_softmaxes.mean(axis=0)

    acc_scores = np.asarray([(labels == predictions).mean() for predictions in np.moveaxis(shift_predictions, 0, 1)])
    f1_scores = np.asarray([
        sklearn.metrics.f1_score(labels, predictions, average='macro', zero_division=0)
        for predictions in np.moveaxis(shift_predictions, 0, 1)
    ])
    inception_score = np.mean(
        np.sum(p_yx * (np.log(p_yx + 1e-12) - np.log(p_y[np.newaxis] + 1e-12)), axis=2),
        axis=0,
    )
    entropy_score = -np.mean(np.sum(p_yx * np.log(p_yx + 1e-12), axis=2), axis=0)

    if class_distribution is None:
        class_distribution = estimate_class_distribution(labels, num_classes)
    one_hot_p_y = np.zeros_like(p_y)
    for i in range(len(shifts)):
        one_hot = np.zeros((shift_softmaxes.shape[0], shift_softmaxes.shape[-1]))
        one_hot[np.arange(one_hot.shape[0]), shift_predictions[:, i]] = 1
        one_hot_p_y[i] = one_hot.mean(axis=0)
    kl_d = np.sum(class_distribution * (np.log(class_distribution + 1e-12) - np.log(one_hot_p_y + 1e-12)), axis=1)
    am_score = kl_d + np.mean(np.sum(-p_yx * np.log(p_yx + 1e-12), axis=2), axis=0)

    best_is_idx = int(np.argsort(inception_score)[::-1][0])
    best_am_idx = int(np.argsort(am_score)[0])
    best_acc_idx = int(np.argsort(acc_scores)[::-1][0])
    best_f1_idx = int(np.argsort(f1_scores)[::-1][0])
    best_ent_idx = int(np.argsort(entropy_score)[0])

    rows = []
    for idx, shift in enumerate(shifts):
        probs = shift_softmaxes[:, idx, :]
        preds = shift_predictions[:, idx]
        conf = probs.max(axis=1)
        pred_counts, pred_probs, pred_entropy, pred_effective = _distribution_summary(preds, num_classes)
        rows.append(
            {
                "shift_delta": shift,
                "is_score": float(inception_score[idx]),
                "am_score": float(am_score[idx]),
                "entropy_mean": float(np.mean(_safe_entropy(probs, axis=1))),
                "prediction_confidence_mean": float(conf.mean()),
                "prediction_confidence_std": float(conf.std()),
                "marginal_entropy": float(_safe_entropy(p_y[idx], axis=0)),
                "marginal_kl_to_uniform": float(np.sum(p_y[idx] * np.log((p_y[idx] + 1e-12) * num_classes))),
                "pseudo_label_coverage_tau": float((conf > pseudo_threshold).mean()),
                "pseudo_label_class_entropy": pred_entropy,
                "pseudo_label_effective_class_count": pred_effective,
                "pseudo_label_class_distribution_json": json.dumps(pred_probs.tolist(), ensure_ascii=True),
                "target_macro_f1_at_shift": float(f1_scores[idx]),
                "target_accuracy_at_shift": float(acc_scores[idx]),
                "target_mean_confidence_at_shift": float(conf.mean()),
            }
        )

    return {
        "rows": rows,
        "shifts": shifts,
        "acc_scores": acc_scores,
        "f1_scores": f1_scores,
        "is_scores": inception_score,
        "am_scores": am_score,
        "entropy_scores": entropy_score,
        "best_is_idx": best_is_idx,
        "best_am_idx": best_am_idx,
        "best_acc_idx": best_acc_idx,
        "best_f1_idx": best_f1_idx,
        "best_ent_idx": best_ent_idx,
        "best_is_shift": shifts[best_is_idx],
        "best_am_shift": shifts[best_am_idx],
        "best_acc_shift": shifts[best_acc_idx],
        "best_f1_shift": shifts[best_f1_idx],
        "best_ent_shift": shifts[best_ent_idx],
        "is_top1_top2_margin": _top_margin(inception_score, True),
        "am_top1_top2_margin": _top_margin(am_score, False),
        "is_curve_sharpness": _curve_sharpness(inception_score, best_is_idx, True),
        "am_curve_sharpness": _curve_sharpness(am_score, best_am_idx, False),
    }


def estimate_temporal_shift_details(
    model,
    target_loader,
    device,
    class_distribution=None,
    min_shift=-60,
    max_shift=60,
    sample_size=100,
    shift_estimator='IS',
    num_classes=None,
    pseudo_threshold=0.9,
    topk=3,
):
    shifts, shift_softmaxes, labels = collect_shift_softmaxes(
        model,
        target_loader,
        device,
        min_shift=min_shift,
        max_shift=max_shift,
        sample_size=sample_size,
    )
    if num_classes is None:
        num_classes = int(shift_softmaxes.shape[-1])
    scores = score_shift_softmaxes(
        shifts,
        shift_softmaxes,
        labels,
        num_classes,
        class_distribution=class_distribution,
        pseudo_threshold=pseudo_threshold,
    )

    estimator = str(shift_estimator).upper()
    if estimator == 'IS':
        best_idx = scores["best_is_idx"]
        ranked = np.argsort(scores["is_scores"])[::-1]
    elif estimator == 'ENT':
        best_idx = scores["best_ent_idx"]
        ranked = np.argsort(scores["entropy_scores"])
    elif estimator == 'AM':
        best_idx = scores["best_am_idx"]
        ranked = np.argsort(scores["am_scores"])
    elif estimator == 'ACC':
        best_idx = scores["best_acc_idx"]
        ranked = np.argsort(scores["acc_scores"])[::-1]
    elif estimator == 'F1':
        best_idx = scores["best_f1_idx"]
        ranked = np.argsort(scores["f1_scores"])[::-1]
    else:
        raise NotImplementedError

    topk = max(1, int(topk))
    topk_indices = [int(idx) for idx in ranked[:topk]]
    scores.update(
        {
            "best_shift": shifts[best_idx],
            "best_shift_idx": int(best_idx),
            "topk_shifts": [shifts[idx] for idx in topk_indices],
            "topk_indices": topk_indices,
            "labels": labels,
            "shift_softmaxes": shift_softmaxes,
        }
    )
    return scores

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
    details = estimate_temporal_shift_details(
        model,
        target_loader,
        device,
        class_distribution=class_distribution,
        min_shift=min_shift,
        max_shift=max_shift,
        sample_size=sample_size,
        shift_estimator=shift_estimator,
    )
    print(f"Most accurate shift {details['best_acc_shift']} with {np.max(details['acc_scores']):.3f}")
    estimator = str(shift_estimator).upper()
    if estimator == 'IS':
        print(
            f"Best Inception Score shift {details['best_shift']} "
            f"with accuracy {details['acc_scores'][details['best_shift_idx']]:.3f}"
        )
    elif estimator == 'ENT':
        print(
            f"Best Entropy Score shift {details['best_shift']} "
            f"with accuracy {details['acc_scores'][details['best_shift_idx']]:.3f}"
        )
    elif estimator == 'AM':
        print(
            f"Best AM Score shift {details['best_shift']} "
            f"with accuracy {details['acc_scores'][details['best_shift_idx']]:.3f}"
        )
    elif estimator == 'F1':
        print(
            f"Best F1 shift {details['best_shift']} "
            f"with macro_f1 {details['f1_scores'][details['best_shift_idx']]:.3f}"
        )
    return details["best_shift"]




@torch.no_grad()
def get_pseudo_labels(model, data_loader, device, best_shift, n=500):
    model.eval()
    pseudo_softmaxes = []
    indices = []
    for i, sample in enumerate(tqdm(data_loader, "computing pseudo labels", disable=not sys.stderr.isatty())):
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
