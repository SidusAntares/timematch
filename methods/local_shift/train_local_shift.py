"""Minimal trainable TimeMatch local-shift entry for v3.2.1 Stage 2."""

from __future__ import annotations

import json
import os
import time
import hashlib
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from evaluation import validation
from methods.local_shift.local_position import (
    check_temporal_positions_in_range,
    compute_stage_residual_shift,
    expand_stage_shift_to_time,
)
from methods.local_shift.logging import append_local_shift_tsv
from methods.local_shift.soft_alignment import SoftStageAligner
from methods.local_shift.target_partition import BudgetedFeatureChangePartitioner
from methods.timematch_base.train_loop import (
    _check_temporal_index_range,
    _format_diag_value,
    _format_elapsed_seconds,
    _timestamp,
    estimate_temporal_shift_details,
    estimate_class_distribution,
    get_data_loaders,
    get_pseudo_labels,
    update_ema_variables,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, cycle, to_cuda


REQUIRED_REFERENCE_KEYS = {
    "class_stage_feats",
    "class_stage_centers",
    "class_stage_durations",
    "class_stage_mask",
    "class_counts",
}


def load_source_stage_reference(path: str, device) -> Dict[str, torch.Tensor]:
    if not path:
        raise ValueError("--source_stage_reference_path is required for timematch_local_shift")
    if not os.path.exists(path):
        raise ValueError(f"source stage reference does not exist: {path}")
    reference = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(reference, dict):
        raise ValueError("source stage reference must be a dict")
    for key, value in list(reference.items()):
        if torch.is_tensor(value):
            reference[key] = value.to(device)
    return reference


def validate_source_stage_reference(
    reference: Dict[str, torch.Tensor],
    num_classes: int,
    feature_dim: int,
    kmax: int,
) -> None:
    missing = sorted(REQUIRED_REFERENCE_KEYS - set(reference))
    if missing:
        raise ValueError(f"source stage reference missing keys: {missing}")

    feats = reference["class_stage_feats"]
    centers = reference["class_stage_centers"]
    durations = reference["class_stage_durations"]
    mask = reference["class_stage_mask"]
    class_counts = reference["class_counts"]

    if feats.ndim != 3:
        raise ValueError("class_stage_feats must be [C,K,D]")
    if feats.shape[0] != num_classes:
        raise ValueError(f"reference num_classes mismatch: got {feats.shape[0]}, expected {num_classes}")
    if feats.shape[1] < kmax:
        raise ValueError(f"reference Kmax {feats.shape[1]} is smaller than requested {kmax}")
    if feats.shape[2] != feature_dim:
        raise ValueError(f"reference feature dim {feats.shape[2]} does not match target dim {feature_dim}")
    if centers.shape != feats.shape[:2]:
        raise ValueError("class_stage_centers must be [C,K]")
    if durations.shape != feats.shape[:2]:
        raise ValueError("class_stage_durations must be [C,K]")
    if mask.shape != feats.shape[:2]:
        raise ValueError("class_stage_mask must be [C,K]")
    if class_counts.shape[0] != num_classes:
        raise ValueError("class_counts must match num_classes")
    if not bool(mask[:, :kmax].any().item()):
        raise ValueError("source stage reference contains no valid stage")


def _temporal_position_bounds(model) -> Tuple[float, float]:
    encoder = model.temporal_encoder
    max_temporal_shift = float(getattr(encoder, "max_temporal_shift", 0))
    table_size = int(encoder.positional_enc.num_embeddings)
    return -max_temporal_shift, float(table_size - max_temporal_shift - 1)


def compute_local_shift_positions(
    temporal_features: torch.Tensor,
    positions: torch.Tensor,
    pseudo_labels: torch.Tensor,
    pseudo_mask: torch.Tensor,
    source_reference: Dict[str, torch.Tensor],
    global_shift,
    kmax: int = 8,
    min_stage_len: int = 3,
    top_m: int = 3,
    change_threshold: Optional[float] = None,
    change_quantile: Optional[float] = 0.75,
    nms_radius: int = 2,
    time_weight: float = 1.0,
    duration_weight: float = 0.2,
    feature_weight: float = 0.5,
    local_shift_clip: Optional[float] = 20.0,
    mode: str = "residual",
    detach_correspondence: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute adjusted target positions from target features and source stage references."""

    if mode not in _LOCAL_SHIFT_POSITION_MODES:
        raise ValueError(f"unsupported local shift mode: {mode}")
    features_for_correspondence = temporal_features.detach() if detach_correspondence else temporal_features
    positions_for_correspondence = positions.detach() if detach_correspondence else positions

    partitioner = BudgetedFeatureChangePartitioner(
        kmax=kmax,
        min_stage_len=min_stage_len,
        change_threshold=change_threshold,
        change_quantile=change_quantile,
        nms_radius=nms_radius,
    )
    partition = partitioner(features_for_correspondence, positions_for_correspondence)
    aligner = SoftStageAligner(
        top_m=top_m,
        time_gap_weight=time_weight,
        duration_gap_weight=duration_weight,
        feature_weight=feature_weight,
    )
    alignment = aligner(
        partition["stage_feats"],
        partition["stage_centers"],
        partition["stage_durations"],
        partition["stage_mask"],
        pseudo_labels,
        pseudo_mask,
        source_reference,
        global_shift,
    )

    valid_stage_mask = partition["stage_mask"] & alignment["valid_mask"]
    if mode == "global_only":
        residual = torch.zeros_like(partition["stage_centers"])
        residual_logs = {
            "residual_mean": 0.0,
            "residual_std": 0.0,
            "residual_abs_mean": 0.0,
            "residual_clip_fraction": 0.0,
            "residual_gate_keep_ratio": 0.0,
            "residual_alpha": 0.0,
        }
    else:
        residual, residual_logs = _compute_safety_residual_shift(
            partition["stage_centers"],
            alignment["expected_source_centers"],
            global_shift,
            mode=mode,
            local_shift_clip=local_shift_clip,
            stage_mask=valid_stage_mask,
            alignment_weights=alignment.get("weights"),
            return_logs=True,
        )

    positions_local, position_logs = expand_stage_shift_to_time(
        positions.float(),
        partition["stage_to_time"],
        residual,
        valid_stage_mask,
        global_shift,
    )
    stage_counts = partition["stage_count"].float()
    logs = {
        "local_shift_mean": residual_logs["residual_mean"],
        "local_shift_std": residual_logs["residual_std"],
        "local_shift_abs_mean": residual_logs["residual_abs_mean"],
        "local_shift_clip_fraction": residual_logs["residual_clip_fraction"],
        "stage_count_mean": float(stage_counts.mean().item()),
        "stage_count_std": float(stage_counts.std(unbiased=False).item()) if stage_counts.numel() else 0.0,
        "stage_count_max": int(stage_counts.max().item()) if stage_counts.numel() else 0,
        "stage_count_p90": float(torch.quantile(stage_counts, 0.9).item()) if stage_counts.numel() else 0.0,
        "alignment_entropy": alignment["logs"].get("alignment_entropy", 0.0),
        "alignment_top1_mass": alignment["logs"].get("top1_mass", 0.0),
        "alignment_valid_ratio": alignment["logs"].get("valid_ratio", 0.0),
        "alignment_fallback_ratio": alignment["logs"].get("fallback_ratio", 0.0),
        "position_min": float(positions_local.min().item()),
        "position_max": float(positions_local.max().item()),
        "position_clamp_ratio": position_logs.get("position_clamp_ratio", 0.0),
    }
    return positions_local, logs


_LOCAL_SHIFT_POSITION_MODES = {
    "global_only",
    "residual",
    "residual_raw",
    "residual_zero_mean",
    "residual_scaled_zero_mean_alpha05",
    "residual_gated_scaled_zero_mean_alpha05_top065",
}


def _residual_mode_settings(mode: str) -> Dict[str, object]:
    if mode in {"residual", "residual_raw"}:
        return {"zero_mean": False, "alpha": 1.0, "gate_top1": None}
    if mode == "residual_zero_mean":
        return {"zero_mean": True, "alpha": 1.0, "gate_top1": None}
    if mode == "residual_scaled_zero_mean_alpha05":
        return {"zero_mean": True, "alpha": 0.5, "gate_top1": None}
    if mode == "residual_gated_scaled_zero_mean_alpha05_top065":
        return {"zero_mean": True, "alpha": 0.5, "gate_top1": 0.65}
    raise ValueError(f"unsupported residual mode: {mode}")


def _compute_safety_residual_shift(
    target_stage_centers: torch.Tensor,
    expected_source_centers: torch.Tensor,
    global_shift,
    mode: str,
    local_shift_clip: Optional[float] = None,
    stage_mask: Optional[torch.Tensor] = None,
    alignment_weights: Optional[torch.Tensor] = None,
    return_logs: bool = False,
):
    settings = _residual_mode_settings(mode)
    batch = target_stage_centers.shape[0]
    raw = compute_stage_residual_shift(
        target_stage_centers,
        expected_source_centers,
        global_shift,
        local_shift_clip=None,
        stage_mask=stage_mask,
        return_logs=False,
    )
    valid = torch.ones_like(raw, dtype=torch.bool) if stage_mask is None else stage_mask.bool()
    residual = raw

    if settings["zero_mean"]:
        valid_float = valid.float()
        denom = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (residual * valid_float).sum(dim=1, keepdim=True) / denom
        residual = torch.where(valid, residual - mean, torch.zeros_like(residual))

    residual = residual * float(settings["alpha"])

    gate_top1 = settings["gate_top1"]
    gate_keep_ratio = 1.0
    if gate_top1 is not None:
        if alignment_weights is None:
            raise ValueError("gated residual mode requires alignment weights")
        top1 = alignment_weights.max(dim=-1).values
        gate = top1 >= float(gate_top1)
        active = valid & gate
        possible = valid.float().sum().clamp_min(1.0)
        gate_keep_ratio = float(active.float().sum().item() / possible.item())
        valid = active
        residual = torch.where(active, residual, torch.zeros_like(residual))

    unclipped = residual
    if local_shift_clip is not None:
        residual = residual.clamp(min=-float(local_shift_clip), max=float(local_shift_clip))

    if not return_logs:
        return residual

    valid_residual = residual[valid]
    if valid_residual.numel() == 0:
        logs = {
            "residual_mean": 0.0,
            "residual_std": 0.0,
            "residual_abs_mean": 0.0,
            "residual_clip_fraction": 0.0,
        }
    else:
        clipped = (unclipped != residual) & valid
        logs = {
            "residual_mean": float(valid_residual.mean().item()),
            "residual_std": float(valid_residual.std(unbiased=False).item()),
            "residual_abs_mean": float(valid_residual.abs().mean().item()),
            "residual_clip_fraction": float(clipped.float().sum().item() / valid.float().sum().clamp_min(1.0).item()),
        }
    logs.update(
        {
            "residual_gate_keep_ratio": gate_keep_ratio,
            "residual_alpha": float(settings["alpha"]),
            "residual_zero_mean": 1.0 if settings["zero_mean"] else 0.0,
        }
    )
    return residual, logs


def _mean_logs(rows):
    merged = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)):
                merged[key].append(float(value))
    return {key: float(np.mean(values)) if values else 0.0 for key, values in merged.items()}


def _append_epoch_log(path: str, row: Dict[str, object]) -> None:
    if path:
        append_local_shift_tsv(path, row)


def _hash_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha1(tensor.numpy().tobytes()).hexdigest()[:12]


def _hash_state_dict(model: torch.nn.Module) -> str:
    digest = hashlib.sha1()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _append_debug_tsv(path: str, row: Dict[str, object]) -> None:
    if path:
        append_local_shift_tsv(path, row)


def _sync_if_cuda(device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _elapsed_ms(start: float, device) -> float:
    _sync_if_cuda(device)
    return (time.perf_counter() - start) * 1000.0


def _add_timing(timing: Dict[str, float], key: str, value_ms: float) -> None:
    timing[key] = timing.get(key, 0.0) + float(value_ms)


def _estimate_shift_without_target_labels(
    teacher,
    target_loader_no_aug,
    device,
    config,
    min_shift,
    max_shift,
    class_distribution=None,
):
    estimator = str(config.shift_estimator).upper()
    if estimator in {"ACC", "F1"}:
        raise ValueError("timematch_local_shift forbids ACC/F1 shift estimators because they use target labels")
    if class_distribution is None:
        class_distribution = np.ones(config.num_classes, dtype=np.float64) / float(config.num_classes)
    return estimate_temporal_shift_details_unlabeled(
        teacher,
        target_loader_no_aug,
        device,
        class_distribution=class_distribution,
        min_shift=min_shift,
        max_shift=max_shift,
        sample_size=config.sample_size,
        shift_estimator=config.shift_estimator,
        num_classes=config.num_classes,
        pseudo_threshold=config.pseudo_threshold,
        topk=getattr(config, "timematch_topk_shifts", 3),
    )


@torch.no_grad()
def collect_shift_softmaxes_unlabeled(model, target_loader, device, min_shift=-60, max_shift=60, sample_size=100):
    shifts = list(range(min_shift, max_shift + 1))
    model.eval()
    if sample_size is None:
        sample_size = len(target_loader)

    target_iter = iter(target_loader)
    shift_softmaxes = []
    start_time = time.time()
    print(
        "LOCAL_SHIFT_ESTIMATION_START|"
        f"timestamp={_timestamp()}|"
        f"min_shift={min_shift}|"
        f"max_shift={max_shift}|"
        f"sample_size={sample_size}",
        flush=True,
    )
    for _ in range(sample_size):
        try:
            sample = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            sample = next(target_iter)
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        spatial_feats = model.spatial_encoder.forward(pixels, valid_pixels, extra)
        shift_logits = torch.stack(
            [model.decoder(model.temporal_encoder(spatial_feats, positions + shift)) for shift in shifts],
            dim=1,
        )
        shift_softmaxes.append(F.softmax(shift_logits, dim=2))
    shift_softmaxes = torch.cat(shift_softmaxes).cpu().numpy()
    elapsed = time.time() - start_time
    print(
        "LOCAL_SHIFT_ESTIMATION_DONE|"
        f"timestamp={_timestamp()}|"
        f"sample_size={sample_size}|"
        f"elapsed={_format_elapsed_seconds(elapsed)}|"
        f"seconds={elapsed:.3f}",
        flush=True,
    )
    return shifts, shift_softmaxes


def estimate_temporal_shift_details_unlabeled(
    model,
    target_loader,
    device,
    class_distribution,
    min_shift=-60,
    max_shift=60,
    sample_size=100,
    shift_estimator="AM",
    num_classes=None,
    pseudo_threshold=0.9,
    topk=3,
):
    shifts, shift_softmaxes = collect_shift_softmaxes_unlabeled(
        model,
        target_loader,
        device,
        min_shift=min_shift,
        max_shift=max_shift,
        sample_size=sample_size,
    )
    if num_classes is None:
        num_classes = int(shift_softmaxes.shape[-1])
    p_yx = shift_softmaxes
    p_y = shift_softmaxes.mean(axis=0)
    shift_predictions = np.argmax(shift_softmaxes, axis=2)
    inception_score = np.mean(
        np.sum(p_yx * (np.log(p_yx + 1e-12) - np.log(p_y[np.newaxis] + 1e-12)), axis=2),
        axis=0,
    )
    entropy_score = -np.mean(np.sum(p_yx * np.log(p_yx + 1e-12), axis=2), axis=0)

    one_hot_p_y = np.zeros_like(p_y)
    for idx in range(len(shifts)):
        one_hot = np.zeros((shift_softmaxes.shape[0], shift_softmaxes.shape[-1]))
        one_hot[np.arange(one_hot.shape[0]), shift_predictions[:, idx]] = 1
        one_hot_p_y[idx] = one_hot.mean(axis=0)
    class_distribution = np.asarray(class_distribution, dtype=np.float64)
    am_score = (
        np.sum(class_distribution * (np.log(class_distribution + 1e-12) - np.log(one_hot_p_y + 1e-12)), axis=1)
        + np.mean(np.sum(-p_yx * np.log(p_yx + 1e-12), axis=2), axis=0)
    )

    estimator = str(shift_estimator).upper()
    if estimator == "IS":
        best_idx = int(np.argsort(inception_score)[::-1][0])
        ranked = np.argsort(inception_score)[::-1]
    elif estimator == "ENT":
        best_idx = int(np.argsort(entropy_score)[0])
        ranked = np.argsort(entropy_score)
    elif estimator == "AM":
        best_idx = int(np.argsort(am_score)[0])
        ranked = np.argsort(am_score)
    else:
        raise ValueError("unlabeled local shift only supports AM, IS, and ENT estimators")

    topk = max(1, int(topk))
    topk_indices = [int(idx) for idx in ranked[:topk]]
    return {
        "shifts": shifts,
        "shift_softmaxes": shift_softmaxes,
        "is_scores": inception_score,
        "am_scores": am_score,
        "entropy_scores": entropy_score,
        "best_shift": shifts[best_idx],
        "best_shift_idx": int(best_idx),
        "topk_shifts": [shifts[idx] for idx in topk_indices],
        "topk_indices": topk_indices,
    }


def train_timematch_local_shift(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(splits, config, config.balance_source)

    pretrained_path = f"{config.weights}/fold_{fold_num}"
    pretrained_weights = torch.load(f"{pretrained_path}/model.pt", weights_only=False)["state_dict"]
    student.load_state_dict(pretrained_weights)
    teacher = deepcopy(student)
    student.to(device)
    teacher.to(device)

    if not hasattr(student, "forward_from_temporal_features"):
        raise ValueError("model must implement forward_from_temporal_features for timematch_local_shift")

    local_shift_mode = getattr(config, "local_shift_mode", "residual")
    needs_reference = local_shift_mode.startswith("residual") or (
        local_shift_mode == "global_only" and config.local_shift_compute_alignment_in_global_only
    )
    if needs_reference:
        reference = load_source_stage_reference(config.source_stage_reference_path, device)
    else:
        reference = None
    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_loss_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs * config.steps_per_epoch,
        eta_min=0,
    )

    min_shift, max_shift = -config.max_temporal_shift, config.max_temporal_shift
    target_to_source_shift = 0
    target_to_source_topk_shifts = [0]
    source_to_target_shift = 0
    last_shift_details = None
    if config.estimate_shift:
        shift_estimator = "IS" if config.shift_estimator == "AM" else config.shift_estimator
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
        if target_to_source_shift >= 0:
            min_shift = 0
        else:
            max_shift = 0

    pseudo_softmaxes = get_pseudo_labels(teacher, target_loader_no_aug, device, target_to_source_shift, n=None)
    all_pseudo_labels = torch.max(pseudo_softmaxes, dim=1)[1]

    source_iter = iter(cycle(source_loader))
    target_iter = iter(cycle(target_loader))
    pos_min, pos_max = _temporal_position_bounds(student)
    global_step = 0
    best_f1 = 0
    train_start_time = time.time()
    student_start_hash = _hash_state_dict(student)
    teacher_start_hash = _hash_state_dict(teacher)

    print(
        "LOCAL_SHIFT_TRAIN_START|"
        f"timestamp={_timestamp()}|"
        f"mode={local_shift_mode}|"
        f"source_reference={config.source_stage_reference_path}|"
        f"epochs={config.epochs}|"
        f"steps_per_epoch={config.steps_per_epoch}",
        flush=True,
    )

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        print(
            f"---------epoch {epoch + 1}/{config.epochs} | "
            f"timestamp={_timestamp()} | "
            f"elapsed={_format_elapsed_seconds(epoch_start_time - train_start_time)} ---------",
            flush=True,
        )

        if config.estimate_shift:
            estimated_class_distr = estimate_class_distribution(all_pseudo_labels.cpu().numpy(), config.num_classes)
            last_shift_details = estimate_temporal_shift_details(
            teacher,
            target_loader_no_aug,
            device,
            class_distribution=estimated_class_distr,
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

        student.train()
        teacher.eval()
        loss_meter = AverageMeter()
        source_loss_meter = AverageMeter()
        target_loss_meter = AverageMeter()
        pseudo_conf_meter = AverageMeter()
        pseudo_ratio_meter = AverageMeter()
        local_log_rows = []
        timing_totals: Dict[str, float] = defaultdict(float)

        for step in range(config.steps_per_epoch):
            sample_source, (sample_target_weak, sample_target_strong) = next(source_iter), next(target_iter)

            pixels_t_weak, mask_t_weak, position_t_weak, extra_t_weak = to_cuda(sample_target_weak, device)
            segment_start = time.perf_counter()
            with torch.no_grad():
                _check_temporal_index_range(teacher, position_t_weak, target_to_source_shift, "target_teacher")
                teacher_logits = teacher.forward(
                    pixels_t_weak,
                    mask_t_weak,
                    position_t_weak + target_to_source_shift,
                    extra_t_weak,
                )
                teacher_probs = F.softmax(teacher_logits, dim=1)
                pseudo_conf, pseudo_targets = torch.max(teacher_probs, dim=1)
                pseudo_mask = pseudo_conf > config.pseudo_threshold
            _add_timing(timing_totals, "teacher_pseudo_time_ms", _elapsed_ms(segment_start, device))

            pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
            source_labels = sample_source["label"].cuda(device, non_blocking=True)
            segment_start = time.perf_counter()
            _check_temporal_index_range(student, position_s, source_to_target_shift, "source_student")
            logits_source = student.forward(pixels_s, mask_s, position_s + source_to_target_shift, extra_s)
            loss_source = criterion(logits_source, source_labels)
            _add_timing(timing_totals, "source_forward_time_ms", _elapsed_ms(segment_start, device))

            pixels_t, mask_t, position_t, extra_t = to_cuda(sample_target_strong, device)
            local_logs = {
                "local_shift_mean": 0.0,
                "local_shift_std": 0.0,
                "local_shift_abs_mean": 0.0,
                "local_shift_clip_fraction": 0.0,
                "stage_count_mean": 0.0,
                "stage_count_std": 0.0,
                "stage_count_max": 0.0,
                "stage_count_p90": 0.0,
                "alignment_entropy": 0.0,
                "alignment_top1_mass": 0.0,
                "alignment_valid_ratio": 0.0,
                "alignment_fallback_ratio": 0.0,
                "position_min": float(position_t.min().item()),
                "position_max": float(position_t.max().item()),
                "position_clamp_ratio": 0.0,
                "residual_gate_keep_ratio": 0.0,
                "residual_alpha": 0.0,
                "residual_zero_mean": 0.0,
            }
            logits_target = None
            enough_target = bool((pseudo_mask.sum() >= 2).item())

            if enough_target:
                if local_shift_mode == "base_equiv":
                    segment_start = time.perf_counter()
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target_student")
                    logits_target = student.forward(
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        position_t[pseudo_mask],
                        extra_t[pseudo_mask],
                    )
                    _add_timing(timing_totals, "target_normal_forward_time_ms", _elapsed_ms(segment_start, device))

                elif local_shift_mode == "global_forward":
                    segment_start = time.perf_counter()
                    temporal_features_masked = student.spatial_encoder(
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        extra_t[pseudo_mask],
                    )
                    _add_timing(timing_totals, "target_temporal_feature_time_ms", _elapsed_ms(segment_start, device))

                    segment_start = time.perf_counter()
                    positions_global, clamp_logs = check_temporal_positions_in_range(
                        position_t[pseudo_mask].float(),
                        min_position=pos_min,
                        max_position=pos_max,
                        round_to_long=True,
                    )
                    local_logs["position_clamp_ratio"] = clamp_logs["position_clamp_ratio"]
                    logits_target = student.forward_from_temporal_features(temporal_features_masked, positions_global)
                    _add_timing(timing_totals, "target_forward_from_features_time_ms", _elapsed_ms(segment_start, device))

                elif local_shift_mode == "global_only" and not config.local_shift_compute_alignment_in_global_only:
                    segment_start = time.perf_counter()
                    temporal_features_masked = student.spatial_encoder(
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        extra_t[pseudo_mask],
                    )
                    _add_timing(timing_totals, "target_temporal_feature_time_ms", _elapsed_ms(segment_start, device))

                    segment_start = time.perf_counter()
                    positions_global, clamp_logs = check_temporal_positions_in_range(
                        position_t[pseudo_mask].float(),
                        min_position=pos_min,
                        max_position=pos_max,
                        round_to_long=True,
                    )
                    local_logs["position_clamp_ratio"] = clamp_logs["position_clamp_ratio"]
                    logits_target = student.forward_from_temporal_features(temporal_features_masked, positions_global)
                    _add_timing(timing_totals, "target_forward_from_features_time_ms", _elapsed_ms(segment_start, device))

                else:
                    segment_start = time.perf_counter()
                    temporal_features_t = student.spatial_encoder(pixels_t, mask_t, extra_t)
                    _add_timing(timing_totals, "target_temporal_feature_time_ms", _elapsed_ms(segment_start, device))
                    validate_source_stage_reference(
                        reference,
                        num_classes=config.num_classes,
                        feature_dim=int(temporal_features_t.shape[-1]),
                        kmax=config.local_shift_kmax,
                    )
                    with torch.no_grad() if config.local_shift_detach_correspondence else torch.enable_grad():
                        features_for_correspondence = (
                            temporal_features_t.detach() if config.local_shift_detach_correspondence else temporal_features_t
                        )
                        positions_for_correspondence = position_t.detach() if config.local_shift_detach_correspondence else position_t

                        segment_start = time.perf_counter()
                        partitioner = BudgetedFeatureChangePartitioner(
                            kmax=config.local_shift_kmax,
                            min_stage_len=config.local_shift_min_stage_len,
                            change_threshold=config.local_shift_change_threshold,
                            change_quantile=config.local_shift_change_quantile,
                            nms_radius=config.local_shift_nms_radius,
                        )
                        partition = partitioner(features_for_correspondence, positions_for_correspondence)
                        _add_timing(timing_totals, "partition_time_ms", _elapsed_ms(segment_start, device))

                        segment_start = time.perf_counter()
                        aligner = SoftStageAligner(
                            top_m=config.local_shift_topm,
                            time_gap_weight=config.local_shift_time_weight,
                            duration_gap_weight=config.local_shift_duration_weight,
                            feature_weight=config.local_shift_feature_weight,
                        )
                        alignment = aligner(
                            partition["stage_feats"],
                            partition["stage_centers"],
                            partition["stage_durations"],
                            partition["stage_mask"],
                            pseudo_targets,
                            pseudo_mask,
                            reference,
                            target_to_source_shift,
                        )
                        _add_timing(timing_totals, "alignment_time_ms", _elapsed_ms(segment_start, device))

                        segment_start = time.perf_counter()
                        valid_stage_mask = partition["stage_mask"] & alignment["valid_mask"]
                        if local_shift_mode == "global_only":
                            residual = torch.zeros_like(partition["stage_centers"])
                            residual_logs = {
                                "residual_mean": 0.0,
                                "residual_std": 0.0,
                                "residual_abs_mean": 0.0,
                                "residual_clip_fraction": 0.0,
                                "residual_gate_keep_ratio": 0.0,
                                "residual_alpha": 0.0,
                                "residual_zero_mean": 0.0,
                            }
                        else:
                            residual, residual_logs = _compute_safety_residual_shift(
                                partition["stage_centers"],
                                alignment["expected_source_centers"],
                                target_to_source_shift,
                                mode=local_shift_mode,
                                local_shift_clip=config.local_shift_clip,
                                stage_mask=valid_stage_mask,
                                alignment_weights=alignment.get("weights"),
                                return_logs=True,
                            )
                        if local_shift_mode == "global_only":
                            positions_local_float = position_t.float()
                            position_logs = {"position_clamp_ratio": 0.0}
                        else:
                            positions_local_float, position_logs = expand_stage_shift_to_time(
                                position_t.float(),
                                partition["stage_to_time"],
                                residual,
                                valid_stage_mask,
                                target_to_source_shift,
                            )
                        positions_local, clamp_logs = check_temporal_positions_in_range(
                            positions_local_float,
                            min_position=pos_min,
                            max_position=pos_max,
                            round_to_long=True,
                        )
                        _add_timing(timing_totals, "local_position_time_ms", _elapsed_ms(segment_start, device))

                        stage_counts = partition["stage_count"].float()
                        local_logs.update(
                            {
                                "local_shift_mean": residual_logs["residual_mean"],
                                "local_shift_std": residual_logs["residual_std"],
                                "local_shift_abs_mean": residual_logs["residual_abs_mean"],
                                "local_shift_clip_fraction": residual_logs["residual_clip_fraction"],
                                "stage_count_mean": float(stage_counts.mean().item()),
                                "stage_count_std": float(stage_counts.std(unbiased=False).item()) if stage_counts.numel() else 0.0,
                                "stage_count_max": int(stage_counts.max().item()) if stage_counts.numel() else 0,
                                "stage_count_p90": float(torch.quantile(stage_counts, 0.9).item()) if stage_counts.numel() else 0.0,
                                "alignment_entropy": alignment["logs"].get("alignment_entropy", 0.0),
                                "alignment_top1_mass": alignment["logs"].get("top1_mass", 0.0),
                                "alignment_valid_ratio": alignment["logs"].get("valid_ratio", 0.0),
                                "alignment_fallback_ratio": alignment["logs"].get("fallback_ratio", 0.0),
                                "position_min": float(positions_local_float.min().item()),
                                "position_max": float(positions_local_float.max().item()),
                                "position_clamp_ratio": clamp_logs.get(
                                    "position_clamp_ratio",
                                    position_logs.get("position_clamp_ratio", 0.0),
                                ),
                                "residual_gate_keep_ratio": residual_logs.get("residual_gate_keep_ratio", 0.0),
                                "residual_alpha": residual_logs.get("residual_alpha", 0.0),
                                "residual_zero_mean": residual_logs.get("residual_zero_mean", 0.0),
                            }
                        )

                    segment_start = time.perf_counter()
                    logits_target_all = student.forward_from_temporal_features(temporal_features_t, positions_local)
                    logits_target = logits_target_all[pseudo_mask]
                    _add_timing(timing_totals, "target_forward_from_features_time_ms", _elapsed_ms(segment_start, device))

            if logits_target is not None:
                loss_target = criterion(logits_target, pseudo_targets[pseudo_mask])
            else:
                loss_target = logits_source.sum() * 0.0
            loss = loss_source + config.trade_off * loss_target

            if global_step < int(getattr(config, "local_shift_equiv_debug_steps", 0)):
                pseudo_counts = torch.bincount(pseudo_targets.detach().cpu(), minlength=config.num_classes).tolist()
                debug_row = {
                    "task": getattr(config, "timematch_diagnostic_task", "")
                    or f"{config.source.split('/')[1]}_to_{config.target.split('/')[1]}",
                    "mode": local_shift_mode,
                    "seed": config.seed,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "source_labels_hash": _hash_tensor(source_labels),
                    "target_position_hash": _hash_tensor(position_t_weak),
                    "source_logits_mean": float(logits_source.detach().mean().item()),
                    "source_logits_std": float(logits_source.detach().std(unbiased=False).item()),
                    "teacher_target_logits_mean": float(teacher_logits.detach().mean().item()),
                    "teacher_target_logits_std": float(teacher_logits.detach().std(unbiased=False).item()),
                    "pseudo_label_distribution_json": json.dumps(pseudo_counts, ensure_ascii=True),
                    "pseudo_confidence_mean": float(pseudo_conf.detach().mean().item()),
                    "pseudo_mask_ratio": float(pseudo_mask.float().mean().item()),
                    "global_shift": target_to_source_shift,
                    "target_positions_min_after_teacher_shift": float((position_t_weak + target_to_source_shift).min().item()),
                    "target_positions_max_after_teacher_shift": float((position_t_weak + target_to_source_shift).max().item()),
                    "source_loss": float(loss_source.detach().item()),
                    "target_loss": float(loss_target.detach().item()),
                    "total_loss": float(loss.detach().item()),
                    "student_start_hash": student_start_hash,
                    "teacher_start_hash": teacher_start_hash,
                    "weights": str(config.weights),
                }
                _append_debug_tsv(getattr(config, "local_shift_equiv_debug_path", ""), debug_row)

            segment_start = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)
            _add_timing(timing_totals, "backward_time_ms", _elapsed_ms(segment_start, device))

            pseudo_ratio = float(pseudo_mask.float().mean().detach().cpu().item())
            pseudo_confidence = float(pseudo_conf.mean().detach().cpu().item())
            local_logs.update(
                {
                    "target_pseudo_confidence": pseudo_confidence,
                    "target_pseudo_ratio": pseudo_ratio,
                }
            )
            local_log_rows.append(local_logs)
            loss_meter.update(float(loss.item()))
            source_loss_meter.update(float(loss_source.item()))
            target_loss_meter.update(float(loss_target.item()))
            pseudo_conf_meter.update(pseudo_confidence)
            pseudo_ratio_meter.update(pseudo_ratio)

            if step % config.log_step == 0:
                writer.add_scalar("train/loss", loss_meter.val, global_step)
                writer.add_scalar("train/source_loss", source_loss_meter.val, global_step)
                writer.add_scalar("train/target_loss", target_loss_meter.val, global_step)
            global_step += 1

        pseudo_softmaxes = get_pseudo_labels(teacher, target_loader_no_aug, device, target_to_source_shift, n=None)
        all_pseudo_labels = torch.max(pseudo_softmaxes, dim=1)[1]
        mean_local_logs = _mean_logs(local_log_rows)
        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        timing_denominator = max(1, config.steps_per_epoch)
        timing_means = {
            "epoch_time_s": epoch_elapsed,
            "source_forward_time_ms": timing_totals.get("source_forward_time_ms", 0.0) / timing_denominator,
            "teacher_pseudo_time_ms": timing_totals.get("teacher_pseudo_time_ms", 0.0) / timing_denominator,
            "target_temporal_feature_time_ms": timing_totals.get("target_temporal_feature_time_ms", 0.0) / timing_denominator,
            "target_normal_forward_time_ms": timing_totals.get("target_normal_forward_time_ms", 0.0) / timing_denominator,
            "partition_time_ms": timing_totals.get("partition_time_ms", 0.0) / timing_denominator,
            "alignment_time_ms": timing_totals.get("alignment_time_ms", 0.0) / timing_denominator,
            "local_position_time_ms": timing_totals.get("local_position_time_ms", 0.0) / timing_denominator,
            "target_forward_from_features_time_ms": timing_totals.get("target_forward_from_features_time_ms", 0.0) / timing_denominator,
            "backward_time_ms": timing_totals.get("backward_time_ms", 0.0) / timing_denominator,
        }
        log_row = {
            "task": getattr(config, "timematch_diagnostic_task", "")
            or f"{config.source.split('/')[1]}_to_{config.target.split('/')[1]}",
            "seed": config.seed,
            "epoch": epoch + 1,
            "step": global_step,
            "source_config": os.path.basename(str(config.weights).rstrip("/")),
            "da_config": "timematch_local_shift",
            "global_shift": target_to_source_shift,
            "local_shift_mode": config.local_shift_mode,
            "local_shift_mean": mean_local_logs.get("local_shift_mean", 0.0),
            "local_shift_std": mean_local_logs.get("local_shift_std", 0.0),
            "local_shift_abs_mean": mean_local_logs.get("local_shift_abs_mean", 0.0),
            "local_shift_clip_fraction": mean_local_logs.get("local_shift_clip_fraction", 0.0),
            "stage_count_mean": mean_local_logs.get("stage_count_mean", 0.0),
            "stage_count_std": mean_local_logs.get("stage_count_std", 0.0),
            "stage_count_max": mean_local_logs.get("stage_count_max", 0.0),
            "stage_count_p90": mean_local_logs.get("stage_count_p90", 0.0),
            "alignment_entropy": mean_local_logs.get("alignment_entropy", 0.0),
            "alignment_top1_mass": mean_local_logs.get("alignment_top1_mass", 0.0),
            "alignment_valid_ratio": mean_local_logs.get("alignment_valid_ratio", 0.0),
            "alignment_fallback_ratio": mean_local_logs.get("alignment_fallback_ratio", 0.0),
            "position_min": mean_local_logs.get("position_min", 0.0),
            "position_max": mean_local_logs.get("position_max", 0.0),
            "position_clamp_ratio": mean_local_logs.get("position_clamp_ratio", 0.0),
            "residual_gate_keep_ratio": mean_local_logs.get("residual_gate_keep_ratio", 0.0),
            "residual_alpha": mean_local_logs.get("residual_alpha", 0.0),
            "residual_zero_mean": mean_local_logs.get("residual_zero_mean", 0.0),
            "target_pseudo_confidence": pseudo_conf_meter.avg,
            "target_pseudo_ratio": pseudo_ratio_meter.avg,
            "source_loss": source_loss_meter.avg,
            "target_loss": target_loss_meter.avg,
            "total_loss": loss_meter.avg,
            **timing_means,
        }
        logging_start = time.perf_counter()
        logging_time_ms = _elapsed_ms(logging_start, device)
        log_row["logging_time_ms"] = logging_time_ms
        _append_epoch_log(config.local_shift_log_path, log_row)
        print(
            "LOCAL_SHIFT_EPOCH_SUMMARY|"
            f"timestamp={_timestamp()}|"
            f"epoch={epoch + 1}|"
            f"elapsed={_format_elapsed_seconds(total_elapsed)}|"
            f"epoch_elapsed={_format_elapsed_seconds(epoch_elapsed)}|"
            + "|".join(f"{key}={_format_diag_value(value)}" for key, value in log_row.items()),
            flush=True,
        )

        if config.run_validation:
            model_for_validation = student if config.output_student else teacher
            model_for_validation.eval()
            best_f1 = validation(best_f1, None, config, criterion, device, epoch, model_for_validation, val_loader, writer)

    model_to_save = student if config.output_student else teacher
    torch.save({"state_dict": model_to_save.state_dict()}, best_model_path)
