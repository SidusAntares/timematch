"""Source class-stage reference construction for v3.2.1."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from methods.local_shift.target_partition import BudgetedFeatureChangePartitioner, pool_stage_features
from utils.train_utils import to_cuda


def _as_positions_for_batch(positions: torch.Tensor, batch_size: int) -> torch.Tensor:
    if positions.ndim == 1:
        return positions.unsqueeze(0).expand(batch_size, -1)
    if positions.ndim == 2:
        return positions
    raise ValueError("positions must be [T] or [B,T]")


def build_reference_from_temporal_features(
    temporal_features: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    num_classes: int,
    partitioner: Optional[BudgetedFeatureChangePartitioner] = None,
) -> Dict[str, torch.Tensor]:
    """Build class-stage references from source temporal features.

    Each sample is partitioned independently.  Stage prototypes are then
    aggregated by source class and stage index.  Invalid padded stages are
    represented by ``class_stage_mask=False``.
    """

    if temporal_features.ndim != 3:
        raise ValueError("temporal_features must be shaped [N,T,D]")
    if labels.ndim != 1 or labels.shape[0] != temporal_features.shape[0]:
        raise ValueError("labels must be shaped [N] and match temporal_features")

    partitioner = partitioner or BudgetedFeatureChangePartitioner()
    num_samples, steps, dim = temporal_features.shape
    positions = _as_positions_for_batch(positions, num_samples).to(temporal_features.device)
    if positions.shape[:2] != (num_samples, steps):
        raise ValueError("positions must match temporal feature time length")

    dtype = temporal_features.dtype
    device = temporal_features.device
    kmax = partitioner.kmax

    feat_sum = torch.zeros(num_classes, kmax, dim, device=device, dtype=dtype)
    center_sum = torch.zeros(num_classes, kmax, device=device, dtype=dtype)
    duration_sum = torch.zeros(num_classes, kmax, device=device, dtype=dtype)
    stage_counts = torch.zeros(num_classes, kmax, device=device, dtype=dtype)
    class_counts = torch.zeros(num_classes, device=device, dtype=dtype)

    for idx in range(num_samples):
        cls = int(labels[idx].item())
        if cls < 0 or cls >= num_classes:
            continue
        class_counts[cls] += 1
        intervals = partitioner.partition_curve(temporal_features[idx], positions[idx])
        pooled = pool_stage_features(temporal_features[idx], intervals)
        for stage_idx, interval in enumerate(intervals[:kmax]):
            feat_sum[cls, stage_idx] += pooled[stage_idx]
            center_sum[cls, stage_idx] += float(interval.center_time)
            duration_sum[cls, stage_idx] += float(interval.duration)
            stage_counts[cls, stage_idx] += 1

    valid = stage_counts > 0
    denom = stage_counts.clamp_min(1.0)
    class_stage_feats = feat_sum / denom.unsqueeze(-1)
    class_stage_centers = center_sum / denom
    class_stage_durations = duration_sum / denom

    config = {
        "kmax": int(kmax),
        "min_stage_len": int(partitioner.min_stage_len),
        "change_threshold": partitioner.change_threshold,
        "change_quantile": partitioner.change_quantile,
        "nms_radius": int(partitioner.nms_radius),
    }
    return {
        "class_stage_feats": class_stage_feats,
        "class_stage_centers": class_stage_centers,
        "class_stage_durations": class_stage_durations,
        "class_stage_mask": valid,
        "class_counts": class_counts,
        "stage_sample_counts": stage_counts,
        "config": config,
    }


@torch.no_grad()
def build_source_stage_reference(
    model,
    source_loader: Iterable,
    device,
    num_classes: int,
    kmax: int = 8,
    min_stage_len: int = 3,
    change_threshold: Optional[float] = None,
    change_quantile: Optional[float] = 0.75,
    nms_radius: int = 2,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Extract source temporal features and build class-stage references."""

    partitioner = BudgetedFeatureChangePartitioner(
        kmax=kmax,
        min_stage_len=min_stage_len,
        change_threshold=change_threshold,
        change_quantile=change_quantile,
        nms_radius=nms_radius,
    )
    was_training = model.training
    model.eval()

    features, labels, positions = [], [], []
    for batch_idx, sample in enumerate(source_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        pixels, valid_pixels, batch_positions, extra = to_cuda(sample, device)
        output = model(
            pixels,
            valid_pixels,
            batch_positions,
            extra,
            return_temporal_features=True,
        )
        _, temporal_features = output
        features.append(temporal_features.detach().cpu())
        labels.append(sample["label"].detach().cpu().long())
        positions.append(batch_positions.detach().cpu())

    if was_training:
        model.train()

    if not features:
        raise ValueError("source_loader produced no batches for source reference construction")

    all_features = torch.cat(features, dim=0)
    all_labels = torch.cat(labels, dim=0)
    all_positions = torch.cat(positions, dim=0)
    reference = build_reference_from_temporal_features(
        all_features,
        all_labels,
        all_positions,
        num_classes=num_classes,
        partitioner=partitioner,
    )
    reference["summary"] = summarize_source_stage_reference(reference)
    return reference


def summarize_source_stage_reference(reference: Dict[str, torch.Tensor]) -> Dict[str, float]:
    mask = reference["class_stage_mask"]
    class_counts = reference["class_counts"]
    stage_counts = mask.sum(dim=1).float()
    return {
        "class_count_nonzero": int((class_counts > 0).sum().item()),
        "class_counts": [int(x) for x in class_counts.cpu().tolist()],
        "avg_stage_count": float(stage_counts.mean().item()),
        "max_stage_count": int(stage_counts.max().item()) if stage_counts.numel() else 0,
        "stage_mask_ratio": float(mask.float().mean().item()),
        "kmax": int(mask.shape[1]),
        "min_stage_len": int(reference["config"]["min_stage_len"]),
        "change_quantile": reference["config"]["change_quantile"],
        "change_threshold": reference["config"]["change_threshold"],
        "nms_radius": int(reference["config"]["nms_radius"]),
    }
