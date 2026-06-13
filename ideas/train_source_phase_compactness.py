import torch
import os
from torchvision import transforms
from tqdm import tqdm

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


def _parse_grad_diag_steps(value):
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(step) for step in value}
    text = str(value).replace(";", ",").replace(" ", ",")
    steps = set()
    for item in text.split(","):
        item = item.strip()
        if item:
            steps.add(int(item))
    return steps


def _parse_source_checkpoint_epochs(value):
    if value is None:
        return None, set()
    text = str(value).strip().lower()
    if not text:
        return None, set()
    if text in {"all", "*"}:
        return "all", set()
    epochs = set()
    for item in text.replace(";", ",").replace(" ", ",").split(","):
        item = item.strip()
        if item:
            epochs.add(int(item))
    return "selected", epochs


def _save_source_epoch_checkpoint(
    model,
    source_feature_reshaper,
    config,
    epoch,
    best_f1,
):
    mode, epochs = _parse_source_checkpoint_epochs(
        getattr(config, "source_checkpoint_epochs", "")
    )
    epoch_1based = epoch + 1
    if mode is None:
        return
    if mode == "selected" and epoch_1based not in epochs:
        return

    checkpoint_dir = os.path.join(config.fold_dir, "source_epoch_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch_1based:03d}.pt")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_f1": best_f1,
        "source_epoch_checkpoint": epoch_1based,
    }
    if source_feature_reshaper is not None:
        checkpoint["source_feature_reshaper_state_dict"] = source_feature_reshaper.state_dict()
    torch.save(checkpoint, checkpoint_path)

    manifest_path = os.path.join(checkpoint_dir, "manifest.tsv")
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("epoch\tcheckpoint\tbest_f1\n")
    with open(manifest_path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{epoch_1based}\t{checkpoint_path}\t{best_f1:.6f}\n")
    print(
        "SOURCE_CHECKPOINT|"
        f"epoch={epoch_1based}|"
        f"path={checkpoint_path}|"
        f"best_f1={best_f1:.6f}"
    )


def _named_trainable_params(module):
    if module is None:
        return []
    return [param for param in module.parameters() if param.requires_grad]


def _build_grad_diag_param_groups(model, source_feature_reshaper):
    groups = [
        ("spatial_encoder", _named_trainable_params(getattr(model, "spatial_encoder", None))),
        ("temporal_encoder", _named_trainable_params(getattr(model, "temporal_encoder", None))),
        ("decoder", _named_trainable_params(getattr(model, "decoder", None))),
    ]
    if source_feature_reshaper is not None:
        groups.append(("reshaper", _named_trainable_params(source_feature_reshaper)))
    return [(name, params) for name, params in groups if params]


def _component_grads_by_group(loss, param_groups):
    grads_by_group = {}
    if loss is None or not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        for group_name, _ in param_groups:
            grads_by_group[group_name] = None
        return grads_by_group

    all_params = []
    group_slices = []
    offset = 0
    for group_name, params in param_groups:
        all_params.extend(params)
        group_slices.append((group_name, offset, offset + len(params)))
        offset += len(params)

    grads = torch.autograd.grad(
        loss,
        all_params,
        retain_graph=True,
        allow_unused=True,
    )
    for group_name, start, end in group_slices:
        flat_parts = [
            grad.detach().reshape(-1)
            for grad in grads[start:end]
            if grad is not None
        ]
        grads_by_group[group_name] = torch.cat(flat_parts) if flat_parts else None
    return grads_by_group


def _grad_norm(flat_grad):
    if flat_grad is None:
        return 0.0
    return float(flat_grad.norm().item())


def _grad_cosine(left_grad, right_grad, eps=1e-12):
    left_norm = _grad_norm(left_grad)
    right_norm = _grad_norm(right_grad)
    if left_grad is None or right_grad is None or left_norm <= eps or right_norm <= eps:
        return 0.0, 0
    cosine = torch.dot(left_grad, right_grad) / (left_grad.norm() * right_grad.norm()).clamp_min(eps)
    return float(cosine.item()), 1


def _compute_deterministic_temporal_delta(temporal_encoder, raw_spatial_feats, reshaped_spatial_feats, positions):
    module_states = [(module, module.training) for module in temporal_encoder.modules()]
    temporal_encoder.eval()
    try:
        with torch.no_grad():
            raw_temporal = temporal_encoder(raw_spatial_feats.detach(), positions)
            reshaped_temporal = temporal_encoder(reshaped_spatial_feats.detach(), positions)
            return (
                (reshaped_temporal - raw_temporal).norm()
                / raw_temporal.norm().clamp_min(1e-12)
            )
    finally:
        for module, training in module_states:
            module.train(training)


def _resolve_structure_feature_target(config, source_feature_reshaper):
    target = str(getattr(config, "source_structure_feature_target", "auto")).lower()
    if target == "auto":
        return "reshaped" if source_feature_reshaper is not None else "raw"
    if source_feature_reshaper is None and target in {"reshaped", "both"}:
        return "raw"
    return target


def _compute_source_structure_loss_on_features(
    feats,
    positions,
    targets,
    config,
    phase_weight_tracker,
    anchor_feats,
    detach_features=False,
):
    structure_feats = feats.detach() if detach_features else feats
    structure_anchor = anchor_feats.detach() if detach_features else anchor_feats
    return compute_source_structure_loss(
        structure_feats,
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
        compact_distance=getattr(config, "source_structure_compact_distance", "mse"),
        norm_preserve_trade_off=getattr(config, "source_structure_norm_preserve_trade_off", 0.0),
        norm_preserve_target=getattr(config, "source_structure_norm_preserve_target", "min_mean"),
        norm_preserve_value=getattr(config, "source_structure_norm_preserve_value", 1.0),
        anchor_spatial_feats=structure_anchor,
        anchor_positions=positions,
    )


def _print_source_grad_diagnostics(losses, param_groups, epoch, step, global_step_1based):
    grad_maps = {
        loss_name: _component_grads_by_group(loss, param_groups)
        for loss_name, loss in losses.items()
    }
    for loss_name, group_map in grad_maps.items():
        for group_name, flat_grad in group_map.items():
            print(
                "SOURCE_GRAD_DIAG|"
                f"global_step={global_step_1based}|"
                f"epoch={epoch + 1}|"
                f"batch_step={step + 1}|"
                f"loss={loss_name}|"
                f"group={group_name}|"
                f"norm={_grad_norm(flat_grad):.8e}|"
                f"active={1 if flat_grad is not None and _grad_norm(flat_grad) > 0.0 else 0}"
            )

    pairs = [
        ("compact", "cls_raw"),
        ("compact", "cls_reshaped"),
        ("compact", "reshaper_reg"),
        ("reshaper_reg", "cls_raw"),
        ("reshaper_reg", "cls_reshaped"),
        ("dual_relation", "cls_raw"),
        ("dual_relation", "cls_reshaped"),
    ]
    for left_name, right_name in pairs:
        if left_name not in grad_maps or right_name not in grad_maps:
            continue
        for group_name, _ in param_groups:
            left_grad = grad_maps[left_name].get(group_name)
            right_grad = grad_maps[right_name].get(group_name)
            cosine, valid = _grad_cosine(left_grad, right_grad)
            left_norm = _grad_norm(left_grad)
            right_norm = _grad_norm(right_grad)
            ratio = left_norm / right_norm if right_norm > 1e-12 else 0.0
            print(
                "SOURCE_GRAD_COS|"
                f"global_step={global_step_1based}|"
                f"epoch={epoch + 1}|"
                f"batch_step={step + 1}|"
                f"left={left_name}|"
                f"right={right_name}|"
                f"group={group_name}|"
                f"left_norm={left_norm:.8e}|"
                f"right_norm={right_norm:.8e}|"
                f"ratio={ratio:.8e}|"
                f"cosine={cosine:.8e}|"
                f"valid={valid}"
            )


def train_supervised_source_phase_compactness(model, config, writer, splits, val_loader, device, best_model_path):
    """
    Source-only training with source-domain phase compactness regularization.

    Design choice:
    - when a source-only reshaper is enabled, structure regularization is attached
      to the reshaper branch;
    - when the reshaper is disabled, structure regularization can still be applied
      directly to raw source features. This is needed for reshaper-vs-compactness
      mechanism diagnostics.
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
        init_seed=getattr(config, "source_feature_reshaper_init_seed", -1),
    )
    if source_feature_reshaper is not None and not getattr(config, "source_feature_reshaper_trainable", True):
        for param in source_feature_reshaper.parameters():
            param.requires_grad_(False)
    structure_trade_off = (
        abs(float(getattr(config, "source_structure_intra_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_amplitude_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_interphase_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_shape_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_trend_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_season_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_segment_inter_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_boundary_window_trade_off", 0.0)))
        + abs(float(getattr(config, "source_structure_norm_preserve_trade_off", 0.0)))
    )
    use_structure_loss = structure_trade_off > 0.0
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
    grad_diag_enabled = bool(getattr(config, "source_structure_grad_diagnostic", False))
    grad_diag_steps = _parse_grad_diag_steps(getattr(config, "source_structure_grad_diag_steps", ""))
    grad_diag_param_groups = (
        _build_grad_diag_param_groups(model, source_feature_reshaper)
        if grad_diag_enabled
        else []
    )

    best_f1 = 0
    for epoch in range(config.epochs):
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        compact_loss_meter = AverageMeter()
        compact_raw_loss_meter = AverageMeter()
        compact_reshaped_loss_meter = AverageMeter()
        reshaper_loss_meter = AverageMeter()
        dual_cls_loss_meter = AverageMeter()
        dual_relation_loss_meter = AverageMeter()
        spatial_delta_ratio_meter = AverageMeter()
        temporal_delta_ratio_meter = AverageMeter()

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch + 1}/{config.epochs}')
        global_step = epoch * len(data_loader)
        for step, sample in progress_bar:
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
            else:
                spatial_feats_anchor = spatial_feats_raw

            if use_structure_loss:
                structure_target = _resolve_structure_feature_target(config, source_feature_reshaper)
                detach_structure_features = bool(getattr(config, "source_structure_detach_features", False))
                compact_raw_loss = spatial_feats_raw.sum() * 0.0
                compact_reshaped_loss = spatial_feats_raw.sum() * 0.0
                raw_logs = {}
                reshaped_logs = {}
                if structure_target in {"raw", "both"}:
                    compact_raw_loss, raw_logs = _compute_source_structure_loss_on_features(
                        spatial_feats_raw,
                        positions,
                        targets,
                        config,
                        phase_weight_tracker,
                        spatial_feats_raw,
                        detach_features=detach_structure_features,
                    )
                if structure_target in {"reshaped", "both"}:
                    compact_reshaped_loss, reshaped_logs = _compute_source_structure_loss_on_features(
                        spatial_feats,
                        positions,
                        targets,
                        config,
                        phase_weight_tracker,
                        spatial_feats_anchor,
                        detach_features=detach_structure_features,
                    )
                compact_loss = compact_raw_loss + compact_reshaped_loss
                compact_logs = {
                    "compactness_loss": float(raw_logs.get("compactness_loss", 0.0))
                    + float(reshaped_logs.get("compactness_loss", 0.0)),
                    "compactness_raw_loss": float(raw_logs.get("compactness_loss", 0.0)),
                    "compactness_reshaped_loss": float(reshaped_logs.get("compactness_loss", 0.0)),
                    "source_structure_detached": float(detach_structure_features),
                }
                for key, value in raw_logs.items():
                    if key != "compactness_loss":
                        compact_logs[f"raw_{key}"] = value
                for key, value in reshaped_logs.items():
                    if key != "compactness_loss":
                        compact_logs[f"reshaped_{key}"] = value

            if source_feature_reshaper is not None:
                temporal_feats = model.temporal_encoder(spatial_feats.detach(), positions)
                outputs = model.decoder(temporal_feats)
                spatial_delta_ratio = (
                    (spatial_feats.detach() - spatial_feats_raw.detach()).norm()
                    / spatial_feats_raw.detach().norm().clamp_min(1e-12)
                )
                temporal_delta_ratio = _compute_deterministic_temporal_delta(
                    model.temporal_encoder,
                    spatial_feats_raw,
                    spatial_feats,
                    positions,
                )
            else:
                temporal_feats = temporal_feats_raw
                outputs = outputs_raw
                spatial_delta_ratio = spatial_feats_raw.sum() * 0.0
                temporal_delta_ratio = temporal_feats_raw.sum() * 0.0

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

            global_step_1based = global_step + step + 1
            if grad_diag_enabled and global_step_1based in grad_diag_steps:
                _print_source_grad_diagnostics(
                    {
                        "cls_raw": cls_loss_raw,
                        "cls_reshaped": cls_loss,
                        "compact": compact_loss,
                        "reshaper_reg": reshaper_loss,
                        "dual_relation": dual_relation_loss,
                    },
                    grad_diag_param_groups,
                    epoch,
                    step,
                    global_step_1based,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)
            cls_loss_meter.update(cls_loss_raw.item(), n=config.batch_size)
            compact_loss_meter.update(compact_logs["compactness_loss"], n=config.batch_size)
            compact_raw_loss_meter.update(compact_logs.get("compactness_raw_loss", 0.0), n=config.batch_size)
            compact_reshaped_loss_meter.update(compact_logs.get("compactness_reshaped_loss", 0.0), n=config.batch_size)
            if source_feature_reshaper is not None:
                reshaper_loss_meter.update(reshaper_logs["source_reshaper_reg_loss"], n=config.batch_size)
                dual_cls_loss_meter.update(cls_loss.item(), n=config.batch_size)
                spatial_delta_ratio_meter.update(float(spatial_delta_ratio.item()), n=config.batch_size)
                temporal_delta_ratio_meter.update(float(temporal_delta_ratio.item()), n=config.batch_size)
                if getattr(config, "source_feature_dual_path", False):
                    dual_relation_loss_meter.update(dual_relation_logs["source_dual_relation_loss"], n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    lr=f'{lr:.1E}',
                    loss=f"{loss_meter.avg:.3f}",
                    cls=f"{cls_loss_meter.avg:.3f}",
                    compact=f"{compact_loss_meter.avg:.3f}",
                    rawc=f"{compact_raw_loss_meter.avg:.3f}",
                    reshapedc=f"{compact_reshaped_loss_meter.avg:.3f}",
                    reshaper=f"{reshaper_loss_meter.avg:.3f}",
                    dualcls=f"{dual_cls_loss_meter.avg:.3f}",
                    dualrel=f"{dual_relation_loss_meter.avg:.3f}",
                )
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)
                writer.add_scalar("train/source_cls_loss_raw", cls_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_phase_compactness_loss", compact_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_structure_loss", compact_loss_meter.val, global_step + step)
                if source_feature_reshaper is not None:
                    writer.add_scalar("train/source_feature_reshaper_reg_loss", reshaper_loss_meter.val, global_step + step)
                    writer.add_scalar("train/source_cls_loss_reshaped", dual_cls_loss_meter.val, global_step + step)
                    writer.add_scalar("train/source_reshaper_spatial_delta_ratio", spatial_delta_ratio_meter.val, global_step + step)
                    writer.add_scalar("train/source_reshaper_temporal_delta_ratio", temporal_delta_ratio_meter.val, global_step + step)
                    if getattr(config, "source_feature_dual_path", False):
                        writer.add_scalar("train/source_dual_relation_loss", dual_relation_loss_meter.val, global_step + step)
                for key, value in compact_logs.items():
                    if key != "compactness_loss":
                        writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in reshaper_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in dual_relation_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)

        progress_bar.close()
        print(
            "SOURCE_EPOCH_SUMMARY|"
            f"epoch={epoch + 1}|"
            f"loss={loss_meter.avg:.6f}|"
            f"cls={cls_loss_meter.avg:.6f}|"
            f"compact={compact_loss_meter.avg:.6f}|"
            f"compact_raw={compact_raw_loss_meter.avg:.6f}|"
            f"compact_reshaped={compact_reshaped_loss_meter.avg:.6f}|"
            f"target={_resolve_structure_feature_target(config, source_feature_reshaper)}|"
            f"detached={bool(getattr(config, 'source_structure_detach_features', False))}|"
            f"compact_distance={getattr(config, 'source_structure_compact_distance', 'mse')}|"
            f"norm_preserve={float(getattr(config, 'source_structure_norm_preserve_trade_off', 0.0)):.6f}|"
            f"norm_target={getattr(config, 'source_structure_norm_preserve_target', 'min_mean')}|"
            f"norm_value={float(getattr(config, 'source_structure_norm_preserve_value', 1.0)):.6f}|"
            f"reshaper={reshaper_loss_meter.avg:.6f}|"
            f"dualcls={dual_cls_loss_meter.avg:.6f}|"
            f"dualrel={dual_relation_loss_meter.avg:.6f}|"
            f"spatial_delta={spatial_delta_ratio_meter.avg:.8f}|"
            f"temporal_delta={temporal_delta_ratio_meter.avg:.8f}"
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
        _save_source_epoch_checkpoint(
            model,
            source_feature_reshaper,
            config,
            epoch,
            best_f1,
        )
