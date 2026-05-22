import json
import os

import torch
from data_adapters.factory import create_train_dataset, create_training_loader
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
from ideas.source_structure_adaptivity import SourceTargetIntraStrengthAdapter
from ideas.source_structure_reliability import compute_svd_structure_reliability_factors
from ideas.source_temporal_window import (
    compute_source_target_soft_support_mask,
    compute_source_target_temporal_mask,
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


def _collect_static_mask_features(
    model,
    source_feature_reshaper,
    loader,
    device,
    max_batches,
    use_reshaper,
):
    features = []
    labels = []
    positions = []
    was_training = model.training
    reshaper_was_training = source_feature_reshaper.training if source_feature_reshaper is not None else False
    model.eval()
    if source_feature_reshaper is not None:
        source_feature_reshaper.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            pixels, mask, batch_positions, extra = to_cuda(sample, device)
            batch_labels = sample.get("label")
            if batch_labels is not None:
                batch_labels = batch_labels.cuda(device=device, non_blocking=True)
            batch_features = model.spatial_encoder(pixels, mask, extra)
            if use_reshaper and source_feature_reshaper is not None and batch_labels is not None:
                batch_features = source_feature_reshaper(batch_features.detach(), positions=batch_positions, labels=batch_labels)
            features.append(batch_features.detach().cpu())
            positions.append(batch_positions.detach().cpu())
            if batch_labels is not None:
                labels.append(batch_labels.detach().cpu())
    if was_training:
        model.train()
    if source_feature_reshaper is not None and reshaper_was_training:
        source_feature_reshaper.train()
    if not features:
        raise RuntimeError("No batches were available for source-target static temporal mask computation.")
    output = {
        "features": torch.cat(features, dim=0).to(device),
        "positions": torch.cat(positions, dim=0).to(device),
    }
    if labels:
        output["labels"] = torch.cat(labels, dim=0).to(device)
    return output


def _compute_static_source_target_temporal_mask(
    model,
    source_feature_reshaper,
    source_loader,
    target_loader,
    config,
    phase_partition_spec,
    phase_count,
    device,
):
    max_batches = int(getattr(config, "source_structure_static_mask_max_batches", 64))
    source_bundle = _collect_static_mask_features(
        model,
        source_feature_reshaper,
        source_loader,
        device,
        max_batches=max_batches,
        use_reshaper=True,
    )
    target_bundle = _collect_static_mask_features(
        model,
        None,
        target_loader,
        device,
        max_batches=max_batches,
        use_reshaper=False,
    )
    temporal_window_mode = str(getattr(config, "source_structure_temporal_window_mode", "none")).lower()
    if temporal_window_mode == "source_target_soft_support":
        weights, logs = compute_source_target_soft_support_mask(
            source_bundle["features"],
            source_bundle["labels"],
            target_bundle["features"],
            source_positions=source_bundle["positions"],
            target_positions=target_bundle["positions"],
            min_weight=getattr(config, "source_structure_temporal_window_min_weight", 0.15),
            smooth_kernel_size=getattr(config, "source_structure_temporal_support_smooth_kernel_size", 5),
            reliability_gate=getattr(config, "source_structure_temporal_window_reliability_gate", True),
            reliability_low=getattr(config, "source_structure_temporal_window_reliability_low", 5e-4),
            reliability_high=getattr(config, "source_structure_temporal_window_reliability_high", 4e-2),
        )
    else:
        weights, logs = compute_source_target_temporal_mask(
            source_bundle["features"],
            source_bundle["labels"],
            target_bundle["features"],
            source_positions=source_bundle["positions"],
            phase_partition_spec=phase_partition_spec,
            phase_count=phase_count,
            min_sample_points=getattr(config, "source_phase_min_sample_points", 2),
            min_weight=getattr(config, "source_structure_temporal_window_min_weight", 0.15),
            reliability_gate=getattr(config, "source_structure_temporal_window_reliability_gate", True),
            reliability_low=getattr(config, "source_structure_temporal_window_reliability_low", 5e-4),
            reliability_high=getattr(config, "source_structure_temporal_window_reliability_high", 4e-2),
        )
    logs["source_structure_static_mask_max_batches"] = float(max_batches)
    return weights.detach(), logs


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

    dataset_name = config.source
    if config.train_on_target:
        dataset_name = config.target

    dataset = create_train_dataset(
        config,
        dataset_name,
        splits,
    )
    data_loader = create_training_loader(dataset, config)
    target_data_loader = None
    temporal_window_mode = str(getattr(config, "source_structure_temporal_window_mode", "none")).lower()
    needs_target_loader = (
        str(getattr(config, "source_structure_adaptivity_mode", "none")).lower() == "target_margin"
        or temporal_window_mode in {"source_target_mask", "source_target_static_mask", "source_target_soft_support"}
    )
    if needs_target_loader:
        target_dataset = create_train_dataset(
            config,
            config.target,
            splits,
        )
        target_data_loader = create_training_loader(target_dataset, config)
        print(f'target adaptivity dataset: {config.target}, n={len(target_dataset)}, batches={len(target_data_loader)}')
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
    adaptive_structure_weights = bool(getattr(config, "source_structure_adaptive_weights", False))
    reliability_mode = str(getattr(config, "source_structure_adaptivity_mode", "none")).lower()
    target_strength_adapter = None
    target_iter = None
    if adaptive_structure_weights and reliability_mode == "target_margin":
        target_strength_adapter = SourceTargetIntraStrengthAdapter(
            min_factor=getattr(config, "source_structure_reliability_min_factor", 0.75),
            max_factor=getattr(config, "source_structure_reliability_max_factor", 1.20),
        )
        target_iter = iter(target_data_loader)
    static_temporal_window_weights = None
    static_temporal_mask_logs = {}
    static_mask_computed = False
    static_mask_warmup_epochs = max(0, int(getattr(config, "source_structure_static_mask_warmup_epochs", 0)))
    if temporal_window_mode in {"source_target_static_mask", "source_target_soft_support"}:
        print(
            "source-target static temporal support: "
            f"mode={temporal_window_mode}, "
            f"warmup_epochs={static_mask_warmup_epochs}, "
            f"max_batches={int(getattr(config, 'source_structure_static_mask_max_batches', 64))}, "
            f"min_weight={float(getattr(config, 'source_structure_temporal_window_min_weight', 0.15)):.3f}"
        )
    if adaptive_structure_weights:
        print(
            "source structure adaptive weights: "
            f"mode={reliability_mode}, "
            f"zeta={getattr(config, 'source_structure_reliability_zeta', 0.90):.3f}, "
            f"strength={getattr(config, 'source_structure_reliability_strength', 0.35):.3f}, "
            f"bounds=[{getattr(config, 'source_structure_reliability_min_factor', 0.70):.3f}, "
            f"{getattr(config, 'source_structure_reliability_max_factor', 1.20):.3f}]"
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
        if (
            temporal_window_mode in {"source_target_static_mask", "source_target_soft_support"}
            and not static_mask_computed
            and epoch >= static_mask_warmup_epochs
        ):
            static_temporal_window_weights, static_temporal_mask_logs = _compute_static_source_target_temporal_mask(
                model,
                source_feature_reshaper,
                data_loader,
                target_data_loader,
                config,
                phase_partition_spec,
                getattr(phase_weight_tracker, "phase_count", None),
                device,
            )
            static_mask_computed = True
            if static_temporal_window_weights.numel() > 16:
                preview_indices = torch.linspace(
                    0,
                    static_temporal_window_weights.numel() - 1,
                    steps=12,
                    device=static_temporal_window_weights.device,
                ).round().long()
                weight_text = ", ".join(
                    f"t{int(idx.item()) + 1}={float(static_temporal_window_weights[idx].detach().item()):.3f}"
                    for idx in preview_indices
                )
            else:
                weight_text = ", ".join(
                    f"p{idx + 1}={float(weight.detach().item()):.3f}"
                    for idx, weight in enumerate(static_temporal_window_weights)
                )
            print(f"Computed source-target static temporal support: {weight_text}")
            mask_path = os.path.join(config.fold_dir, "source_target_static_temporal_support.json")
            with open(mask_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "mode": temporal_window_mode,
                        "warmup_epochs": static_mask_warmup_epochs,
                        "weights": [
                            float(weight.detach().item())
                            for weight in static_temporal_window_weights
                        ],
                        "logs": static_temporal_mask_logs,
                    },
                    fp,
                    indent=2,
                    sort_keys=True,
                )
            print(f"Saved source-target static temporal support: {mask_path}")
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        compact_loss_meter = AverageMeter()
        reshaper_loss_meter = AverageMeter()
        dual_cls_loss_meter = AverageMeter()
        dual_relation_loss_meter = AverageMeter()
        adaptive_factor_meter = AverageMeter()
        adaptive_raw_factor_meter = AverageMeter()
        adaptive_target_margin_meter = AverageMeter()
        adaptive_source_reliability_meter = AverageMeter()
        adaptive_mismatch_cv_meter = AverageMeter()
        mask_score_meter = AverageMeter()
        mask_weight_meters = {}
        support_score_meter = AverageMeter()
        support_gate_meter = AverageMeter()
        support_weight_meters = {}

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
                structure_weight_factors = None
                structure_reliability_logs = {}
                if adaptive_structure_weights and reliability_mode == "svd_reliability":
                    structure_weight_factors, structure_reliability_logs = compute_svd_structure_reliability_factors(
                        spatial_feats,
                        positions,
                        targets,
                        phase_partition_spec=phase_partition_spec,
                        min_sample_points=getattr(config, "source_phase_min_sample_points", 2),
                        zeta=getattr(config, "source_structure_reliability_zeta", 0.90),
                        strength=getattr(config, "source_structure_reliability_strength", 0.35),
                        min_factor=getattr(config, "source_structure_reliability_min_factor", 0.70),
                        max_factor=getattr(config, "source_structure_reliability_max_factor", 1.20),
                    )
                intra_trade_off = getattr(config, "source_structure_intra_trade_off", 1.0)
                trend_trade_off = getattr(config, "source_structure_trend_trade_off", 0.05)
                segment_inter_trade_off = getattr(config, "source_structure_segment_inter_trade_off", 0.02)
                boundary_window_trade_off = getattr(config, "source_structure_boundary_window_trade_off", 0.02)
                temporal_window_weights = None
                temporal_support_weights = None
                if structure_weight_factors is not None:
                    intra_trade_off = intra_trade_off * structure_weight_factors["intra"]
                    trend_trade_off = trend_trade_off * structure_weight_factors["trend"]
                    segment_inter_trade_off = segment_inter_trade_off * structure_weight_factors["segment_inter"]
                    boundary_window_trade_off = boundary_window_trade_off * structure_weight_factors["boundary_window"]
                if temporal_window_mode == "source_target_static_mask" and static_temporal_window_weights is not None:
                    temporal_window_weights = static_temporal_window_weights
                    structure_reliability_logs.update(static_temporal_mask_logs)
                elif temporal_window_mode == "source_target_soft_support" and static_temporal_window_weights is not None:
                    temporal_support_weights = static_temporal_window_weights
                    structure_reliability_logs.update(static_temporal_mask_logs)
                if target_strength_adapter is not None:
                    try:
                        target_sample = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_data_loader)
                        target_sample = next(target_iter)
                    target_pixels, target_mask, target_positions, target_extra = to_cuda(target_sample, device)
                    spatial_encoder_was_training = model.spatial_encoder.training
                    with torch.no_grad():
                        model.spatial_encoder.eval()
                        target_spatial_feats = model.spatial_encoder(target_pixels, target_mask, target_extra)
                    if spatial_encoder_was_training:
                        model.spatial_encoder.train()
                    target_factor, target_logs = target_strength_adapter.update(
                        spatial_feats,
                        targets,
                        target_spatial_feats,
                    )
                    intra_trade_off = intra_trade_off * target_factor
                    structure_reliability_logs.update(target_logs)
                    if temporal_window_mode == "source_target_mask":
                        temporal_window_weights, temporal_mask_logs = compute_source_target_temporal_mask(
                            spatial_feats,
                            targets,
                            target_spatial_feats,
                            source_positions=positions,
                            phase_partition_spec=phase_partition_spec,
                            phase_count=getattr(phase_weight_tracker, "phase_count", None),
                            min_sample_points=getattr(config, "source_phase_min_sample_points", 2),
                            min_weight=getattr(config, "source_structure_temporal_window_min_weight", 0.15),
                            reliability_gate=getattr(config, "source_structure_temporal_window_reliability_gate", True),
                            reliability_low=getattr(config, "source_structure_temporal_window_reliability_low", 5e-4),
                            reliability_high=getattr(config, "source_structure_temporal_window_reliability_high", 4e-2),
                        )
                        structure_reliability_logs.update(temporal_mask_logs)
                compact_loss, compact_logs = compute_source_structure_loss(
                    spatial_feats,
                    positions,
                    targets,
                    weight_tracker=phase_weight_tracker,
                    version=getattr(config, "source_structure_loss_version", "compactness"),
                    intra_trade_off=intra_trade_off,
                    amplitude_trade_off=getattr(config, "source_structure_amplitude_trade_off", 0.25),
                    interphase_trade_off=getattr(config, "source_structure_interphase_trade_off", 0.25),
                    shape_trade_off=getattr(config, "source_structure_shape_trade_off", 0.15),
                    trend_trade_off=trend_trade_off,
                    season_trade_off=getattr(config, "source_structure_season_trade_off", 0.02),
                    segment_inter_trade_off=segment_inter_trade_off,
                    boundary_window_trade_off=boundary_window_trade_off,
                    boundary_window_size=getattr(config, "source_structure_boundary_window_size", 2),
                    warp_invariant_trade_off=getattr(config, "source_structure_warp_invariant_trade_off", 0.35),
                    prototype_dynamics_trade_off=getattr(config, "source_structure_prototype_dynamics_trade_off", 0.05),
                    trajectory_pooling=getattr(config, "source_structure_trajectory_pooling", "meanmax"),
                    prototype_dynamics_mode=getattr(config, "source_structure_prototype_dynamics_mode", "cosine"),
                    temporal_window_mode=getattr(config, "source_structure_temporal_window_mode", "none"),
                    temporal_window_center=getattr(config, "source_structure_temporal_window_center", 0.5),
                    temporal_window_width=getattr(config, "source_structure_temporal_window_width", 0.35),
                    temporal_window_min_weight=getattr(config, "source_structure_temporal_window_min_weight", 0.15),
                    temporal_window_weights=temporal_window_weights,
                    temporal_support_weights=temporal_support_weights,
                    anchor_spatial_feats=spatial_feats_anchor,
                    anchor_positions=positions,
                )
                compact_logs.update(structure_reliability_logs)
                compact_logs["source_structure_effective_intra_trade_off"] = float(intra_trade_off)
                compact_logs["source_structure_effective_trend_trade_off"] = float(trend_trade_off)
                compact_logs["source_structure_effective_segment_inter_trade_off"] = float(segment_inter_trade_off)
                compact_logs["source_structure_effective_boundary_window_trade_off"] = float(boundary_window_trade_off)
                compact_logs["source_structure_effective_prototype_dynamics_trade_off"] = float(
                    getattr(config, "source_structure_prototype_dynamics_trade_off", 0.05)
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
                        prototype_dynamics_trade_off=getattr(config, "source_structure_prototype_dynamics_trade_off", 0.05),
                        trajectory_pooling=getattr(config, "source_structure_trajectory_pooling", "meanmax"),
                        prototype_dynamics_mode=getattr(config, "source_structure_prototype_dynamics_mode", "cosine"),
                        temporal_window_mode=getattr(config, "source_structure_temporal_window_mode", "none"),
                        temporal_window_center=getattr(config, "source_structure_temporal_window_center", 0.5),
                        temporal_window_width=getattr(config, "source_structure_temporal_window_width", 0.35),
                        temporal_window_min_weight=getattr(config, "source_structure_temporal_window_min_weight", 0.15),
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
            if "source_structure_adaptive_factor" in compact_logs:
                adaptive_factor_meter.update(
                    compact_logs["source_structure_adaptive_factor"],
                    n=config.batch_size,
                )
                adaptive_raw_factor_meter.update(
                    compact_logs["source_structure_adaptive_raw_factor"],
                    n=config.batch_size,
                )
                adaptive_target_margin_meter.update(
                    compact_logs["source_structure_adaptive_target_margin_ratio"],
                    n=config.batch_size,
                )
                adaptive_source_reliability_meter.update(
                    compact_logs["source_structure_adaptive_source_reliability"],
                    n=config.batch_size,
                )
                adaptive_mismatch_cv_meter.update(
                    compact_logs["source_structure_adaptive_temporal_mismatch_cv"],
                    n=config.batch_size,
                )
            if "source_structure_mask_score_mean" in compact_logs:
                mask_score_meter.update(
                    compact_logs["source_structure_mask_score_mean"],
                    n=config.batch_size,
                )
                for key, value in compact_logs.items():
                    if key.startswith("source_structure_mask_weight_p"):
                        if key not in mask_weight_meters:
                            mask_weight_meters[key] = AverageMeter()
                        mask_weight_meters[key].update(value, n=config.batch_size)
            if "source_structure_support_smooth_score_mean" in compact_logs:
                support_score_meter.update(
                    compact_logs["source_structure_support_smooth_score_mean"],
                    n=config.batch_size,
                )
                support_gate_meter.update(
                    compact_logs.get("source_structure_support_gate_rho", 0.0),
                    n=config.batch_size,
                )
                for key, value in compact_logs.items():
                    if key.startswith("source_structure_support_weight_t"):
                        if key not in support_weight_meters:
                            support_weight_meters[key] = AverageMeter()
                        support_weight_meters[key].update(value, n=config.batch_size)
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
                    if key != "compactness_loss" and isinstance(value, (int, float)):
                        writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in reshaper_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in dual_relation_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)

        lr = optimizer.param_groups[0]["lr"]
        summary_parts = [
            f"loss={loss_meter.avg:.4f}",
            f"cls_raw={cls_loss_meter.avg:.4f}",
            f"structure={compact_loss_meter.avg:.4f}",
        ]
        if source_feature_reshaper is not None:
            summary_parts.append(f"reshaper_reg={reshaper_loss_meter.avg:.4f}")
            summary_parts.append(f"cls_reshaped={dual_cls_loss_meter.avg:.4f}")
            if getattr(config, "source_feature_dual_path", False):
                summary_parts.append(f"dual_relation={dual_relation_loss_meter.avg:.4f}")
        if adaptive_factor_meter.count > 0:
            summary_parts.append(f"adaptive_factor={adaptive_factor_meter.avg:.4f}")
            summary_parts.append(f"adaptive_raw={adaptive_raw_factor_meter.avg:.4f}")
            summary_parts.append(f"target_margin={adaptive_target_margin_meter.avg:.4f}")
            summary_parts.append(f"source_rel={adaptive_source_reliability_meter.avg:.4f}")
            summary_parts.append(f"mismatch_cv={adaptive_mismatch_cv_meter.avg:.4f}")
        if mask_score_meter.count > 0:
            summary_parts.append(f"mask_score={mask_score_meter.avg:.4f}")
            for key in sorted(mask_weight_meters):
                phase_id = key.rsplit("_p", 1)[-1]
                summary_parts.append(f"mask_p{phase_id}={mask_weight_meters[key].avg:.3f}")
        if support_score_meter.count > 0:
            summary_parts.append(f"support_score={support_score_meter.avg:.4f}")
            summary_parts.append(f"support_gate={support_gate_meter.avg:.4f}")
            for key in sorted(support_weight_meters):
                support_id = key.rsplit("_t", 1)[-1]
                summary_parts.append(f"support_t{support_id}={support_weight_meters[key].avg:.3f}")
        summary_parts.append(f"lr={lr:.6g}")
        summary_parts.append(f"batches={len(data_loader)}")
        print("Epoch train summary: " + ", ".join(summary_parts))

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
