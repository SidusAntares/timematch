import torch
import os
import time
from torchvision import transforms

from dataset import PixelSetData, create_train_loader
from evaluation import validation
from methods.legacy.old_source_phase_compactness import (
    SourceSegmentWeightTracker,
    build_source_segment_partition_spec,
    compute_source_structure_loss,
    describe_source_segment_partition_spec,
)
from methods.source_structure.losses import (
    compute_source_raw_global_compactness_loss,
    is_raw_global_compactness_version,
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


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _format_elapsed_seconds(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def _build_grad_diag_param_groups(model):
    groups = [
        ("spatial_encoder", _named_trainable_params(getattr(model, "spatial_encoder", None))),
        ("temporal_encoder", _named_trainable_params(getattr(model, "temporal_encoder", None))),
        ("decoder", _named_trainable_params(getattr(model, "decoder", None))),
    ]
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


def _resolve_structure_feature_target(config):
    target = str(getattr(config, "source_structure_feature_target", "auto")).lower()
    if target == "auto":
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
    if is_raw_global_compactness_version(
        getattr(config, "source_structure_loss_version", "compactness")
    ):
        return compute_source_raw_global_compactness_loss(
            structure_feats,
            targets,
            version=getattr(config, "source_structure_loss_version", "compactness"),
            intra_trade_off=getattr(config, "source_structure_intra_trade_off", 1.0),
            compact_distance=getattr(config, "source_structure_compact_distance", "mse"),
            time_smooth_kernel_size=getattr(
                config, "source_structure_time_smooth_kernel_size", 3
            ),
            time_permutation_seed=getattr(
                config, "source_structure_time_permutation_seed", 0
            ),
            elastic_radius=getattr(config, "source_structure_elastic_radius", 0),
            elastic_eta=getattr(config, "source_structure_elastic_eta", 0.1),
            elastic_softmin_tau=getattr(
                config, "source_structure_elastic_softmin_tau", 0.1
            ),
            elastic_detach_center=getattr(
                config, "source_structure_elastic_detach_center", False
            ),
            norm_preserve_trade_off=getattr(
                config, "source_structure_norm_preserve_trade_off", 0.0
            ),
            norm_preserve_target=getattr(
                config, "source_structure_norm_preserve_target", "min_mean"
            ),
            norm_preserve_value=getattr(
                config, "source_structure_norm_preserve_value", 1.0
            ),
        )
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


def _resolve_source_structure_lambda_config(config):
    base = float(getattr(config, "source_structure_lambda_base", -1.0))
    if base < 0.0:
        base = float(getattr(config, "source_structure_intra_trade_off", 1.0))
    final = float(getattr(config, "source_structure_lambda_final", -1.0))
    if final < 0.0:
        final = base
    max_epoch = int(getattr(config, "source_structure_lambda_max_epoch", 0))
    if max_epoch <= 0:
        max_epoch = int(getattr(config, "epochs", 1))
    return {
        "schedule": str(getattr(config, "source_structure_lambda_schedule", "constant")).lower(),
        "base": base,
        "final": final,
        "decay_start": int(getattr(config, "source_structure_lambda_decay_start_epoch", 0)),
        "warmup_epochs": int(getattr(config, "source_structure_lambda_warmup_epochs", 0)),
        "max_epoch": max(1, max_epoch),
    }


def _source_structure_lambda_for_epoch(epoch, config):
    spec = _resolve_source_structure_lambda_config(config)
    schedule = spec["schedule"]
    base = spec["base"]
    final = spec["final"]
    max_epoch = spec["max_epoch"]
    epoch_1based = int(epoch) + 1

    if schedule == "constant":
        return base
    if schedule == "linear_decay":
        decay_start = max(0, int(spec["decay_start"]))
        if decay_start <= 0:
            decay_start = 1
        if epoch_1based <= decay_start:
            return base
        denom = max(1, max_epoch - decay_start)
        progress = min(1.0, max(0.0, (epoch_1based - decay_start) / float(denom)))
        return base + progress * (final - base)
    if schedule == "warmup_then_constant":
        return 0.0 if epoch_1based <= int(spec["warmup_epochs"]) else base
    if schedule == "cosine_decay":
        progress = min(1.0, max(0.0, epoch_1based / float(max_epoch)))
        return final + 0.5 * (base - final) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    raise ValueError(f"Unsupported source_structure_lambda_schedule: {schedule}")


def _write_source_lambda_curve(config):
    spec = _resolve_source_structure_lambda_config(config)
    curve_path = os.path.join(config.fold_dir, "source_structure_lambda_curve.tsv")
    os.makedirs(config.fold_dir, exist_ok=True)
    values = []
    with open(curve_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "epoch\tlambda_value\tschedule_type\tlambda_base\tlambda_final\t"
            "decay_start_epoch\twarmup_epochs\tmax_epoch\n"
        )
        for epoch in range(int(getattr(config, "epochs", 1))):
            value = float(_source_structure_lambda_for_epoch(epoch, config))
            values.append(value)
            handle.write(
                f"{epoch + 1}\t{value:.8f}\t{spec['schedule']}\t{spec['base']:.8f}\t"
                f"{spec['final']:.8f}\t{spec['decay_start']}\t"
                f"{spec['warmup_epochs']}\t{spec['max_epoch']}\n"
            )
    if values:
        print(
            "SOURCE_LAMBDA_CURVE|"
            f"schedule={spec['schedule']}|"
            f"base={spec['base']:.8f}|"
            f"final={spec['final']:.8f}|"
            f"decay_start_epoch={spec['decay_start']}|"
            f"warmup_epochs={spec['warmup_epochs']}|"
            f"max_epoch={spec['max_epoch']}|"
            f"min={min(values):.8f}|"
            f"max={max(values):.8f}|"
            f"mean={sum(values) / len(values):.8f}|"
            f"final_value={values[-1]:.8f}|"
            f"path={curve_path}"
        )
    return curve_path


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
    Source-only training with source-domain structure regularization on raw encoder features.
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
    data_loader = create_train_loader(
        dataset,
        config.batch_size,
        config.num_workers,
        timeout=getattr(config, "data_loader_timeout", 0),
    )
    print(f'training dataset: {dataset_name}, n={len(dataset)}, batches={len(data_loader)}')
    print(
        "SOURCE_TRAIN_START|"
        f"method=sourcephasecompact|"
        f"dataset={dataset_name}|"
        f"samples={len(dataset)}|"
        f"batches={len(data_loader)}|"
        f"batch_size={config.batch_size}|"
        f"num_workers={config.num_workers}|"
        f"data_loader_timeout={getattr(config, 'data_loader_timeout', 0)}|"
        f"epochs={config.epochs}",
        flush=True,
    )
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
    _write_source_lambda_curve(config)

    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    steps_per_epoch = len(data_loader)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)
    phase_weight_tracker = SourceSegmentWeightTracker(
        phase_count=phase_partition_spec["phase_count"],
        phase_partition_spec=phase_partition_spec,
        min_sample_points_per_phase=getattr(config, "source_phase_min_sample_points", 2),
    )
    grad_diag_enabled = bool(getattr(config, "source_structure_grad_diagnostic", False))
    grad_diag_steps = _parse_grad_diag_steps(getattr(config, "source_structure_grad_diag_steps", ""))
    grad_diag_param_groups = (
        _build_grad_diag_param_groups(model)
        if grad_diag_enabled
        else []
    )

    best_f1 = 0
    train_start_time = time.time()
    for epoch in range(config.epochs):
        scheduled_lambda = float(_source_structure_lambda_for_epoch(epoch, config))
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        compact_loss_meter = AverageMeter()
        compact_raw_loss_meter = AverageMeter()
        elastic_metric_meters = {
            "mean_abs_offset": AverageMeter(),
            "center_weight": AverageMeter(),
            "boundary_weight": AverageMeter(),
            "distance_scale": AverageMeter(),
            "struct_loss": AverageMeter(),
        }

        epoch_start_time = time.time()
        print(
            f"---------epoch {epoch + 1}/{config.epochs} | "
            f"timestamp={_timestamp()} | "
            f"elapsed={_format_elapsed_seconds(epoch_start_time - train_start_time)} ---------",
            flush=True,
        )
        speed_probe_steps = {
            step for step in (10, 50, 100, 200, 500)
            if step <= len(data_loader)
        }
        speed_probe_steps.add(len(data_loader))
        global_step = epoch * len(data_loader)
        for step, sample in enumerate(data_loader):
            targets = sample['label'].cuda(device=device, non_blocking=True)
            pixels, mask, positions, extra = to_cuda(sample, device)

            spatial_feats_raw = model.spatial_encoder(pixels, mask, extra)
            temporal_feats_raw = model.temporal_encoder(spatial_feats_raw, positions)
            outputs_raw = model.decoder(temporal_feats_raw)

            compact_loss = spatial_feats_raw.sum() * 0.0
            compact_logs = {"compactness_loss": 0.0}

            if use_structure_loss:
                structure_target = _resolve_structure_feature_target(config)
                detach_structure_features = bool(getattr(config, "source_structure_detach_features", False))
                compact_raw_loss = spatial_feats_raw.sum() * 0.0
                raw_logs = {}
                if structure_target == "raw":
                    original_intra_trade_off = getattr(config, "source_structure_intra_trade_off", 1.0)
                    config.source_structure_intra_trade_off = scheduled_lambda
                    compact_raw_loss, raw_logs = _compute_source_structure_loss_on_features(
                        spatial_feats_raw,
                        positions,
                        targets,
                        config,
                        phase_weight_tracker,
                        spatial_feats_raw,
                        detach_features=detach_structure_features,
                    )
                    config.source_structure_intra_trade_off = original_intra_trade_off
                compact_loss = compact_raw_loss
                compact_logs = {
                    "compactness_loss": float(raw_logs.get("compactness_loss", 0.0)),
                    "compactness_raw_loss": float(raw_logs.get("compactness_loss", 0.0)),
                    "source_structure_detached": float(detach_structure_features),
                }
                for key, value in raw_logs.items():
                    if key != "compactness_loss":
                        compact_logs[f"raw_{key}"] = value

            cls_loss_raw = criterion(outputs_raw, targets)
            loss = cls_loss_raw + compact_loss

            global_step_1based = global_step + step + 1
            if grad_diag_enabled and global_step_1based in grad_diag_steps:
                _print_source_grad_diagnostics(
                    {
                        "cls_raw": cls_loss_raw,
                        "compact": compact_loss,
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
            elastic_metric_meters["mean_abs_offset"].update(
                compact_logs.get("raw_elastic_mean_abs_offset", 0.0),
                n=config.batch_size,
            )
            elastic_metric_meters["center_weight"].update(
                compact_logs.get("raw_elastic_center_weight", 0.0),
                n=config.batch_size,
            )
            elastic_metric_meters["boundary_weight"].update(
                compact_logs.get("raw_elastic_boundary_weight", 0.0),
                n=config.batch_size,
            )
            elastic_metric_meters["distance_scale"].update(
                compact_logs.get("raw_elastic_distance_scale", 0.0),
                n=config.batch_size,
            )
            elastic_metric_meters["struct_loss"].update(
                compact_logs.get("raw_elastic_struct_loss", 0.0),
                n=config.batch_size,
            )

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)
                writer.add_scalar("train/source_cls_loss_raw", cls_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_phase_compactness_loss", compact_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_structure_loss", compact_loss_meter.val, global_step + step)
                for key, value in compact_logs.items():
                    if key != "compactness_loss":
                        writer.add_scalar(f"train/{key}", value, global_step + step)
            if epoch == 0 and (step + 1) in speed_probe_steps:
                elapsed = time.time() - epoch_start_time
                print(
                    "SOURCE_SPEED_PROBE|"
                    f"timestamp={_timestamp()}|"
                    f"method=sourcephasecompact|"
                    f"epoch={epoch + 1}|"
                    f"batches={step + 1}|"
                    f"samples={loss_meter.count}|"
                    f"elapsed={_format_elapsed_seconds(elapsed)}|"
                    f"seconds={elapsed:.3f}|"
                    f"batches_per_sec={(step + 1) / max(elapsed, 1e-9):.3f}|"
                    f"seconds_per_batch={elapsed / max(step + 1, 1):.3f}|"
                    f"loss={loss_meter.avg:.6f}|"
                    f"cls={cls_loss_meter.avg:.6f}|"
                    f"compact={compact_loss_meter.avg:.6f}",
                    flush=True,
                )

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        print(
            "SOURCE_EPOCH_SUMMARY|"
            f"timestamp={_timestamp()}|"
            f"epoch={epoch + 1}|"
            f"source_structure_lambda={scheduled_lambda:.8f}|"
            f"schedule={getattr(config, 'source_structure_lambda_schedule', 'constant')}|"
            f"loss={loss_meter.avg:.6f}|"
            f"cls={cls_loss_meter.avg:.6f}|"
            f"compact={compact_loss_meter.avg:.6f}|"
            f"compact_raw={compact_raw_loss_meter.avg:.6f}|"
            f"target={_resolve_structure_feature_target(config)}|"
            f"detached={bool(getattr(config, 'source_structure_detach_features', False))}|"
            f"compact_distance={getattr(config, 'source_structure_compact_distance', 'mse')}|"
            f"norm_preserve={float(getattr(config, 'source_structure_norm_preserve_trade_off', 0.0)):.6f}|"
            f"norm_target={getattr(config, 'source_structure_norm_preserve_target', 'min_mean')}|"
            f"norm_value={float(getattr(config, 'source_structure_norm_preserve_value', 1.0)):.6f}|"
            f"elastic_radius={int(getattr(config, 'source_structure_elastic_radius', 0))}|"
            f"elastic_eta={float(getattr(config, 'source_structure_elastic_eta', 0.1)):.6f}|"
            f"elastic_softmin_tau={float(getattr(config, 'source_structure_elastic_softmin_tau', 0.1)):.6f}|"
            f"elastic_detach_center={bool(getattr(config, 'source_structure_elastic_detach_center', False))}|"
            f"elastic_mean_abs_offset={elastic_metric_meters['mean_abs_offset'].avg:.6f}|"
            f"elastic_center_weight={elastic_metric_meters['center_weight'].avg:.6f}|"
            f"elastic_boundary_weight={elastic_metric_meters['boundary_weight'].avg:.6f}|"
            f"elastic_distance_scale={elastic_metric_meters['distance_scale'].avg:.6f}|"
            f"elastic_struct_loss={elastic_metric_meters['struct_loss'].avg:.6f}|"
            f"elapsed={_format_elapsed_seconds(total_elapsed)}|"
            f"epoch_elapsed={_format_elapsed_seconds(epoch_elapsed)}|"
            f"seconds={epoch_elapsed:.3f}|"
            f"batches_per_sec={len(data_loader) / max(epoch_elapsed, 1e-9):.3f}",
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
        )
        _save_source_epoch_checkpoint(
            model,
            config,
            epoch,
            best_f1,
        )

