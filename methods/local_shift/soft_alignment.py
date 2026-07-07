"""Soft stage alignment for v3.2.1 local shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class StageAlignment:
    weights: torch.Tensor
    expected_source_times: torch.Tensor
    distances: torch.Tensor


class SoftStageAligner:
    """Align target stages to source class-stage references.

    The alignment matrix is used for local shift estimation only.  This scaffold
    does not add a contrastive or domain loss.
    """

    def __init__(self, temperature: float = 0.1, top_m: Optional[int] = None) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.top_m = top_m

    def __call__(
        self,
        target_stage_features: torch.Tensor,
        source_prototypes: torch.Tensor,
        source_times: torch.Tensor,
        pseudo_labels: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
    ) -> StageAlignment:
        if target_stage_features.ndim != 3:
            raise ValueError("target_stage_features must be [B,Kt,D]")
        if source_prototypes.ndim != 3:
            raise ValueError("source_prototypes must be [C,Ks,D]")

        batch, target_stages, dim = target_stage_features.shape
        if pseudo_labels is None:
            prototypes = source_prototypes.mean(dim=0).unsqueeze(0).expand(batch, -1, -1)
            times = source_times.mean(dim=0).unsqueeze(0).expand(batch, -1)
            mask = None if source_mask is None else source_mask.any(dim=0).unsqueeze(0).expand(batch, -1)
        else:
            prototypes = source_prototypes[pseudo_labels.long()]
            times = source_times[pseudo_labels.long()]
            mask = None if source_mask is None else source_mask[pseudo_labels.long()]

        if prototypes.shape[-1] != dim:
            raise ValueError("target/source feature dimensions do not match")

        diff = target_stage_features.unsqueeze(2) - prototypes.unsqueeze(1)
        distances = diff.pow(2).mean(dim=-1)
        logits = -distances / self.temperature

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        if self.top_m is not None and self.top_m > 0 and self.top_m < logits.shape[-1]:
            top_values, top_indices = torch.topk(logits, k=self.top_m, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(dim=-1, index=top_indices, src=top_values)

        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        expected_times = (weights * times.unsqueeze(1)).sum(dim=-1)
        return StageAlignment(weights=weights, expected_source_times=expected_times, distances=distances)
