import os

import torch
from torchvision import transforms
from dataset import PixelSetData, create_train_loader
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
from ideas.source_phase_grid import make_phase_grid_positions, project_to_phase_grid
from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    RandomTemporalShift,
    ToTensor,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, to_cuda


def _parse_checkpoint_epochs(spec):
    if spec is None:
        return set()
    if isinstance(spec, (list, tuple, set)):
        return {int(x) for x in spec}
    text = str(spec).strip()
    if not text:
        return set()
    tokens = [tok.strip() for tok in text.replace(";", ",").replace("\n", ",").split(",")]
    return {int(tok) for tok in tokens if tok}


def _save_source_checkpoint(model, source_feature_reshaper, path):
    checkpoint = {"state_dict": model.state_dict()}
    if source_feature_reshaper is not None:
        checkpoint["source_feature_reshaper_state_dict"] = source_feature_reshaper.state_dict()
    torch.save(checkpoint, path)


def train_supervised_source_phase_compactness(model, config, writer, splits, val_loader, device, best_model_path):
    """
    Source-only training with source-domain phase compactness regularization.

    Design choice:
    - the shared PSE is driven only by the raw-path classification objective
    - structure regularization is attached to the source-only reshaper branch
    - reshaped-path supervision updates downstream temporal/classification heads,
      but does not backpropagate into the shared PSE or the reshaper
    """
    assert not getattr(config, "with_shift_aug", False), (
        "sourcephasecompact / v2.3 phase-aware training must not use RandomTemporalShift-style "
        "source-side augmentation, because the source phase partition is defined on the original source time axis."
    )
    model.to(device)

    train_transform = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
        Normalize(),
        ToTensor(),
    ])
    dataset_name = config.source
    if config.train_on_target:
        dataset_name = config.target

    dataset = PixelSetData(
        config.data_root,
        dataset_name,
        config.classes,
        train_transform,
        splits[dataset_name]['train'],
        closed_set=config.closed_set,
    )
    data_loader = create_train_loader(dataset, config.batch_size, config.num_workers)
    print(f'training dataset: {dataset_name}, n={len(dataset)}, batches={len(data_loader)}')
    phase_partition_spec = build_source_segment_partition_spec(
        dataset.date_positions,
        dataset=dataset,
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
    print("source segment partition:", describe_source_segment_partition_spec(phase_partition_spec))

    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    steps_per_epoch = len(data_loader)
    source_feature_reshaper = build_source_feature_reshaper(
        kind=getattr(config, "source_feature_reshaper", "none"),
        feature_dim=model.spatial_encoder.output_dim,
        strength=getattr(config, "source_feature_reshaper_strength", 0.10),
        kernel_size=getattr(config, "source_feature_reshaper_kernel_size", 3),
    )
    params = list(model.parameters())
    if source_feature_reshaper is not None:
        source_feature_reshaper.to(device)
        params += list(source_feature_reshaper.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)
    phase_weight_tracker = SourceSegmentWeightTracker(
        phase_count=phase_partition_spec["phase_count"],
        phase_partition_spec=phase_partition_spec,
        min_sample_points_per_phase=getattr(config, "source_phase_min_sample_points", 2),
    )
    phase_grid_count = int(getattr(config, "source_phase_grid_count", 5))
    phase_grid_trade_off = float(getattr(config, "source_phase_grid_trade_off", 0.0))
    phase_grid_weight_tracker = None
    if phase_grid_trade_off > 0.0:
        phase_grid_weight_tracker = SourceSegmentWeightTracker(
            phase_count=phase_grid_count,
            phase_partition_spec={
                "mode": "uniform",
                "phase_count": phase_grid_count,
                "date_positions": list(range(phase_grid_count)),
                "intervals": None,
            },
            min_sample_points_per_phase=1,
        )
        print(
            "source phase-grid structure view: "
            f"count={phase_grid_count}, "
            f"trade_off={phase_grid_trade_off:.4f}, "
            f"kernel={getattr(config, 'source_phase_grid_kernel', 'linear')}, "
            f"bandwidth={getattr(config, 'source_phase_grid_bandwidth', 0.0):.4f}"
        )

    best_f1 = 0
    checkpoint_epochs = _parse_checkpoint_epochs(getattr(config, "source_checkpoint_epochs", ""))
    checkpoint_dir = os.path.join(config.fold_dir, getattr(config, "source_checkpoint_dirname", "checkpoints"))
    if checkpoint_epochs:
        os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(config.epochs):
        print(f"====================Epoch {epoch + 1}/{config.epochs}====================")
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        compact_loss_meter = AverageMeter()
        reshaper_loss_meter = AverageMeter()
        dual_cls_loss_meter = AverageMeter()
        dual_relation_loss_meter = AverageMeter()

        global_step = epoch * len(data_loader)
        for step, sample in enumerate(data_loader):
            targets = sample['label'].cuda(device=device, non_blocking=True)
            pixels, mask, positions, extra = to_cuda(sample, device)

            spatial_feats_raw = model.spatial_encoder(pixels, mask, extra)
            temporal_feats_raw = model.temporal_encoder(spatial_feats_raw, positions)
            outputs_raw = model.decoder(temporal_feats_raw)

            spatial_feats = spatial_feats_raw
            reshaper_loss = spatial_feats_raw.sum() * 0.0
            reshaper_logs = {}
            compact_loss = spatial_feats_raw.sum() * 0.0
            compact_logs = {"compactness_loss": 0.0}
            if source_feature_reshaper is not None:
                spatial_feats_anchor = spatial_feats_raw.detach()
                spatial_feats = source_feature_reshaper(spatial_feats_anchor, positions=positions, labels=targets)
                reshaper_loss, reshaper_logs = compute_source_feature_reshaper_regularization(
                    spatial_feats_anchor,
                    spatial_feats,
                )
                compact_loss, compact_logs = compute_source_structure_loss(
                    spatial_feats,
                    positions,
                    targets,
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
                    warp_invariant_trade_off=getattr(config, "source_structure_warp_invariant_trade_off", 0.35),
                    anchor_spatial_feats=spatial_feats_anchor,
                    anchor_positions=positions,
                )
                if phase_grid_weight_tracker is not None:
                    phase_grid_feats, phase_grid_support = project_to_phase_grid(
                        spatial_feats,
                        positions,
                        grid_count=phase_grid_count,
                        kernel=getattr(config, "source_phase_grid_kernel", "linear"),
                        bandwidth=(
                            None
                            if float(getattr(config, "source_phase_grid_bandwidth", 0.0)) <= 0.0
                            else float(getattr(config, "source_phase_grid_bandwidth", 0.0))
                        ),
                        min_support=float(getattr(config, "source_phase_grid_min_support", 0.20)),
                    )
                    phase_grid_positions = make_phase_grid_positions(positions, phase_grid_count)
                    phase_grid_loss, phase_grid_logs = compute_source_structure_loss(
                        phase_grid_feats,
                        phase_grid_positions,
                        targets,
                        weight_tracker=phase_grid_weight_tracker,
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
                        warp_invariant_trade_off=getattr(config, "source_structure_warp_invariant_trade_off", 0.35),
                    )
                    compact_loss = compact_loss + phase_grid_trade_off * phase_grid_loss
                    compact_logs["source_phase_grid_structure_loss"] = float(phase_grid_loss.detach().item())
                    compact_logs["source_phase_grid_weighted_structure_loss"] = float(
                        (phase_grid_trade_off * phase_grid_loss).detach().item()
                    )
                    compact_logs["source_phase_grid_mean_support"] = float(
                        phase_grid_support.mean().detach().item()
                    )
                    compact_logs["source_phase_grid_min_support"] = float(
                        phase_grid_support.min().detach().item()
                    )
                    compact_logs["source_phase_grid_low_support_fraction"] = float(
                        (
                            phase_grid_support
                            < float(getattr(config, "source_phase_grid_min_support", 0.20))
                        )
                        .to(dtype=phase_grid_support.dtype)
                        .mean()
                        .detach()
                        .item()
                    )
                    for key, value in phase_grid_logs.items():
                        if key.startswith("source_structure_"):
                            compact_logs[f"source_phase_grid_{key}"] = value
                temporal_feats = model.temporal_encoder(spatial_feats.detach(), positions)
                outputs = model.decoder(temporal_feats)
            else:
                temporal_feats = temporal_feats_raw
                outputs = outputs_raw

            cls_loss_raw = criterion(outputs_raw, targets)
            cls_loss = criterion(outputs, targets)
            dual_relation_loss = spatial_feats_raw.sum() * 0.0
            dual_relation_logs = {}
            if source_feature_reshaper is not None and getattr(config, "source_feature_dual_path", False):
                dual_relation_loss, dual_relation_logs = compute_dual_path_relation_regularization(
                    outputs_raw,
                    outputs,
                    raw_temporal_feats=temporal_feats_raw,
                    reshaped_temporal_feats=temporal_feats,
                )

            if source_feature_reshaper is not None and getattr(config, "source_feature_dual_path", False):
                loss = (
                    cls_loss_raw
                    + getattr(config, "source_feature_dual_cls_trade_off", 1.0) * cls_loss
                    + getattr(config, "source_feature_dual_relation_trade_off", 0.05) * dual_relation_loss
                )
            else:
                loss = cls_loss
            loss = loss + compact_loss + getattr(config, "source_feature_reshaper_reg_trade_off", 0.0) * reshaper_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)
            cls_loss_meter.update(cls_loss_raw.item(), n=config.batch_size)
            compact_loss_meter.update(compact_logs["compactness_loss"], n=config.batch_size)
            if source_feature_reshaper is not None:
                reshaper_loss_meter.update(reshaper_logs["source_reshaper_reg_loss"], n=config.batch_size)
                dual_cls_loss_meter.update(cls_loss.item(), n=config.batch_size)
                if getattr(config, "source_feature_dual_path", False):
                    dual_relation_loss_meter.update(dual_relation_logs["source_dual_relation_loss"], n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)
                writer.add_scalar("train/source_cls_loss_raw", cls_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_phase_compactness_loss", compact_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_structure_loss", compact_loss_meter.val, global_step + step)
                if source_feature_reshaper is not None:
                    writer.add_scalar("train/source_feature_reshaper_reg_loss", reshaper_loss_meter.val, global_step + step)
                    writer.add_scalar("train/source_cls_loss_reshaped", dual_cls_loss_meter.val, global_step + step)
                    if getattr(config, "source_feature_dual_path", False):
                        writer.add_scalar("train/source_dual_relation_loss", dual_relation_loss_meter.val, global_step + step)
                for key, value in compact_logs.items():
                    if key != "compactness_loss":
                        writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in reshaper_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in dual_relation_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)

        model.eval()
        best_f1 = validation(
            best_f1,
            best_model_path,
            config,
            criterion,
            device,
            epoch,
            model,
            val_loader,
            writer,
            source_feature_reshaper=source_feature_reshaper,
            apply_source_feature_reshaper=False,
        )
        current_epoch = epoch + 1
        if current_epoch in checkpoint_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{current_epoch}.pt")
            _save_source_checkpoint(model, source_feature_reshaper, checkpoint_path)
            print(f"Saved source checkpoint at epoch {current_epoch}: {checkpoint_path}")
