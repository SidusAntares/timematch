"""Source-only raw-global compactness on pre-LTAE temporal features."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class RawGlobalCompactnessResult:
    """Raw-global loss and batch diagnostics."""

    loss: torch.Tensor
    valid_class_count: int
    samples_per_class: Dict[int, int]
    skipped_class_ids: Tuple[int, ...]


def compute_raw_global_compactness(
    temporal_features: torch.Tensor,
    labels: torch.Tensor,
) -> RawGlobalCompactnessResult:
    """Compute equal-class compactness after mean pooling over time.

    ``temporal_features`` must be the raw PSE output ``H`` with shape
    ``[batch, time, feature]``. The loss intentionally does not accept or use
    timestamps. Classes represented by fewer than two batch samples are
    skipped. If every class is skipped, the returned zero remains connected
    to ``temporal_features`` for autograd.
    """

    if temporal_features.ndim != 3:
        raise ValueError(
            "temporal_features must have shape [batch, time, feature], "
            f"got {tuple(temporal_features.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [batch], got {tuple(labels.shape)}")
    if temporal_features.shape[0] != labels.shape[0]:
        raise ValueError(
            "temporal_features and labels must have the same batch size, "
            f"got {temporal_features.shape[0]} and {labels.shape[0]}"
        )

    pooled_features = temporal_features.mean(dim=1)
    zero = temporal_features.sum() * 0.0
    class_losses = []
    samples_per_class = {}
    skipped_class_ids = []

    for class_id_tensor in labels.unique(sorted=True):
        class_id = int(class_id_tensor.item())
        class_features = pooled_features[labels == class_id_tensor]
        sample_count = int(class_features.shape[0])
        samples_per_class[class_id] = sample_count
        if sample_count < 2:
            skipped_class_ids.append(class_id)
            continue

        class_center = class_features.mean(dim=0, keepdim=True)
        class_loss = (class_features - class_center).pow(2).sum(dim=-1).mean()
        class_losses.append(class_loss)

    loss = torch.stack(class_losses).mean() if class_losses else zero
    return RawGlobalCompactnessResult(
        loss=loss,
        valid_class_count=len(class_losses),
        samples_per_class=samples_per_class,
        skipped_class_ids=tuple(skipped_class_ids),
    )
