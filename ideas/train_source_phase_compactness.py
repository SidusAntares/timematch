import torch
from tqdm import tqdm

from data_adapters.factory import create_train_dataset, create_training_loader, make_train_transform
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
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, to_cuda


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
        transform=make_train_transform(config),
    )
    data_loader = create_training_loader(dataset, config)
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

    best_f1 = 0
    for epoch in range(config.epochs):
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        compact_loss_meter = AverageMeter()
        reshaper_loss_meter = AverageMeter()
        dual_cls_loss_meter = AverageMeter()
        dual_relation_loss_meter = AverageMeter()

        print(f"Epoch {epoch + 1}/{config.epochs} source training start: steps={len(data_loader)}")
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
                    v271_trend_kernel_size=getattr(config, "source_structure_v271_trend_kernel_size", 5),
                    v271_trend_smoothing_mode=getattr(
                        config,
                        "source_structure_v271_trend_smoothing_mode",
                        "time",
                    ),
                    v271_trend_bandwidth=getattr(config, "source_structure_v271_trend_bandwidth", 0.0),
                    v271_trend_kernel=getattr(config, "source_structure_v271_trend_kernel", "gaussian"),
                    v271_trend_dynamics_trade_off=getattr(
                        config,
                        "source_structure_v271_trend_dynamics_trade_off",
                        0.05,
                    ),
                    v271_residual_variance_trade_off=getattr(
                        config,
                        "source_structure_v271_residual_variance_trade_off",
                        0.10,
                    ),
                    v271_residual_energy_trade_off=getattr(
                        config,
                        "source_structure_v271_residual_energy_trade_off",
                        0.05,
                    ),
                    v271_residual_energy_margin=getattr(
                        config,
                        "source_structure_v271_residual_energy_margin",
                        1.0,
                    ),
                    v271_segment_basis_trade_off=getattr(
                        config,
                        "source_structure_v271_segment_basis_trade_off",
                        1.0,
                    ),
                    v271_event_support_trade_off=getattr(
                        config,
                        "source_structure_v271_event_support_trade_off",
                        1.0,
                    ),
                    v271_event_support_count=getattr(
                        config,
                        "source_structure_v271_event_support_count",
                        2,
                    ),
                    v271_event_support_sigma_ratio=getattr(
                        config,
                        "source_structure_v271_event_support_sigma_ratio",
                        0.20,
                    ),
                    v271_event_support_mode=getattr(
                        config,
                        "source_structure_v271_event_support_mode",
                        "event",
                    ),
                    anchor_spatial_feats=spatial_feats_anchor,
                    anchor_positions=positions,
                )
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
                print(
                    f"Epoch {epoch + 1}/{config.epochs} "
                    f"step {step + 1}/{len(data_loader)} "
                    f"lr={lr:.1E} "
                    f"loss={loss_meter.avg:.4f} "
                    f"cls={cls_loss_meter.avg:.4f} "
                    f"structure={compact_loss_meter.avg:.4f} "
                    f"reshaper={reshaper_loss_meter.avg:.4f} "
                    f"dualcls={dual_cls_loss_meter.avg:.4f} "
                    f"dualrel={dual_relation_loss_meter.avg:.4f}",
                    flush=True,
                )
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

        print(
            f"Epoch {epoch + 1}/{config.epochs} source training done: "
            f"loss={loss_meter.avg:.4f} cls={cls_loss_meter.avg:.4f} "
            f"structure={compact_loss_meter.avg:.4f}",
            flush=True,
        )

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
