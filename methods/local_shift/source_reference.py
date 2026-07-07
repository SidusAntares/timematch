"""Source class-stage reference construction for v3.2.1.

The first v3.2.1 version is only a scaffold: source checkpoints can later be
converted into class-stage references here, then reused by local shift DA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from methods.local_shift.target_partition import (
    BudgetedFeatureChangePartitioner,
    StageInterval,
    pool_stage_features,
)


@dataclass
class SourceStageReference:
    """Padded source class-stage prototypes.

    Attributes:
        prototypes: Tensor shaped ``[C, Kmax, D]``.
        times: Tensor shaped ``[C, Kmax]`` with reference stage center times.
        mask: Boolean tensor shaped ``[C, Kmax]``.
        counts: Tensor shaped ``[C, Kmax]`` with sample counts per stage.
    """

    prototypes: torch.Tensor
    times: torch.Tensor
    mask: torch.Tensor
    counts: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "prototypes": self.prototypes,
            "times": self.times,
            "mask": self.mask,
            "counts": self.counts,
        }


def build_reference_from_temporal_features(
    temporal_features: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    num_classes: int,
    partitioner: Optional[BudgetedFeatureChangePartitioner] = None,
) -> SourceStageReference:
    """Build source class-stage references from already extracted features.

    Args:
        temporal_features: Source temporal features shaped ``[N, T, D]``.
        labels: Source labels shaped ``[N]``.
        times: Time coordinates shaped ``[T]`` or ``[N, T]``.
        num_classes: Number of source classes.
        partitioner: Feature-change partitioner.  A default one is used if
            omitted.

    Returns:
        A padded ``SourceStageReference``.
    """

    if temporal_features.ndim != 3:
        raise ValueError("temporal_features must be shaped [N, T, D]")
    if labels.ndim != 1 or labels.shape[0] != temporal_features.shape[0]:
        raise ValueError("labels must be shaped [N] and match features")

    partitioner = partitioner or BudgetedFeatureChangePartitioner()
    device = temporal_features.device
    dtype = temporal_features.dtype
    _, _, dim = temporal_features.shape

    class_stage_features: List[List[torch.Tensor]] = []
    class_stage_times: List[List[torch.Tensor]] = []
    class_stage_counts: List[List[torch.Tensor]] = []
    max_stages = 1

    for cls in range(num_classes):
        cls_mask = labels == cls
        cls_features = temporal_features[cls_mask]
        if cls_features.numel() == 0:
            class_stage_features.append([])
            class_stage_times.append([])
            class_stage_counts.append([])
            continue

        cls_times = times
        if times.ndim == 2:
            cls_times = times[cls_mask].float().mean(dim=0)
        intervals = partitioner(cls_features, cls_times)
        pooled = pool_stage_features(cls_features, intervals)
        stage_times = torch.tensor(
            [stage.center_time for stage in intervals],
            device=device,
            dtype=dtype,
        )
        counts = torch.full(
            (len(intervals),),
            float(cls_features.shape[0]),
            device=device,
            dtype=dtype,
        )
        class_stage_features.append([x for x in pooled])
        class_stage_times.append([x for x in stage_times])
        class_stage_counts.append([x for x in counts])
        max_stages = max(max_stages, len(intervals))

    prototypes = torch.zeros(num_classes, max_stages, dim, device=device, dtype=dtype)
    ref_times = torch.zeros(num_classes, max_stages, device=device, dtype=dtype)
    ref_mask = torch.zeros(num_classes, max_stages, device=device, dtype=torch.bool)
    counts = torch.zeros(num_classes, max_stages, device=device, dtype=dtype)

    for cls in range(num_classes):
        for idx, feature in enumerate(class_stage_features[cls]):
            prototypes[cls, idx] = feature
            ref_times[cls, idx] = class_stage_times[cls][idx]
            counts[cls, idx] = class_stage_counts[cls][idx]
            ref_mask[cls, idx] = True

    return SourceStageReference(
        prototypes=prototypes,
        times=ref_times,
        mask=ref_mask,
        counts=counts,
    )


def build_source_stage_reference(
    model,
    source_loader: Iterable,
    device,
    num_classes: int,
    kmax: int = 6,
    min_stage_len: int = 2,
    change_threshold: Optional[float] = None,
    change_quantile: Optional[float] = 0.8,
    nms_radius: int = 1,
    max_batches: Optional[int] = None,
) -> SourceStageReference:
    """Extract temporal features and build source stage references.

    The full project-specific extraction path is intentionally left as a hook:
    models must expose temporal features before this function becomes a
    production entry point.
    """

    raise NotImplementedError(
        "v3.2.1 scaffold only: expose temporal features from the encoder, "
        "then call build_reference_from_temporal_features(...)."
    )
