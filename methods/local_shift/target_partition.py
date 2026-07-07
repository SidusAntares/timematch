"""Budgeted feature-change partitioning for v3.2.1 local shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass(frozen=True)
class StageInterval:
    """A contiguous temporal support interval."""

    start: int
    end: int
    center_time: float
    score: float = 0.0

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class BudgetedFeatureChangePartitioner:
    """Create variable stages from feature-change peaks.

    This is not uniform segmentation.  It uses feature change along time as a
    candidate boundary signal and keeps at most ``kmax`` stages.
    """

    def __init__(
        self,
        kmax: int = 6,
        min_stage_len: int = 2,
        change_threshold: Optional[float] = None,
        change_quantile: Optional[float] = 0.8,
        nms_radius: int = 1,
    ) -> None:
        if kmax < 1:
            raise ValueError("kmax must be >= 1")
        if min_stage_len < 1:
            raise ValueError("min_stage_len must be >= 1")
        self.kmax = int(kmax)
        self.min_stage_len = int(min_stage_len)
        self.change_threshold = change_threshold
        self.change_quantile = change_quantile
        self.nms_radius = int(max(nms_radius, 0))

    def __call__(self, temporal_features: torch.Tensor, times: torch.Tensor) -> List[StageInterval]:
        if temporal_features.ndim == 3:
            curve = temporal_features.float().mean(dim=0)
        elif temporal_features.ndim == 2:
            curve = temporal_features.float()
        else:
            raise ValueError("temporal_features must be [B,T,D] or [T,D]")
        if times.ndim == 2:
            times_1d = times.float().mean(dim=0)
        elif times.ndim == 1:
            times_1d = times.float()
        else:
            raise ValueError("times must be [T] or [B,T]")

        steps = curve.shape[0]
        if steps == 0:
            raise ValueError("empty temporal feature curve")
        if steps == 1 or self.kmax == 1:
            return [self._interval(0, steps - 1, times_1d, 0.0)]

        change = (curve[1:] - curve[:-1]).pow(2).mean(dim=-1).sqrt()
        threshold = self._threshold(change)
        candidates = torch.nonzero(change >= threshold, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return [self._interval(0, steps - 1, times_1d, 0.0)]

        order = torch.argsort(change[candidates], descending=True)
        selected: List[int] = []
        max_cuts = min(self.kmax - 1, steps - 1)
        for candidate in candidates[order].tolist():
            cut = int(candidate) + 1
            if any(abs(cut - prev) <= self.nms_radius for prev in selected):
                continue
            selected.append(cut)
            if len(selected) >= max_cuts:
                break

        return self._cuts_to_intervals(sorted(selected), times_1d, change)

    def _threshold(self, change: torch.Tensor) -> torch.Tensor:
        if self.change_threshold is not None:
            return torch.as_tensor(self.change_threshold, device=change.device, dtype=change.dtype)
        if self.change_quantile is not None:
            return torch.quantile(change, float(self.change_quantile))
        return change.mean()

    def _cuts_to_intervals(
        self,
        cuts: Sequence[int],
        times: torch.Tensor,
        change: torch.Tensor,
    ) -> List[StageInterval]:
        steps = int(times.shape[0])
        valid_cuts: List[int] = []
        prev = 0
        for cut in cuts:
            if cut - prev < self.min_stage_len:
                continue
            if steps - cut < self.min_stage_len:
                continue
            valid_cuts.append(cut)
            prev = cut

        bounds = [0] + valid_cuts + [steps]
        intervals: List[StageInterval] = []
        for left, right in zip(bounds[:-1], bounds[1:]):
            start = int(left)
            end = int(right) - 1
            score = 0.0
            if start > 0:
                score = float(change[start - 1].detach().cpu())
            intervals.append(self._interval(start, end, times, score))
        return intervals or [self._interval(0, steps - 1, times, 0.0)]

    @staticmethod
    def _interval(start: int, end: int, times: torch.Tensor, score: float) -> StageInterval:
        center = times[start : end + 1].float().mean()
        return StageInterval(start=start, end=end, center_time=float(center.detach().cpu()), score=score)


def pool_stage_features(temporal_features: torch.Tensor, intervals: Sequence[StageInterval]) -> torch.Tensor:
    """Average-pool temporal features over variable intervals.

    Args:
        temporal_features: ``[B,T,D]`` or ``[T,D]``.
        intervals: Stage intervals.

    Returns:
        ``[K,D]`` stage features.
    """

    if temporal_features.ndim == 3:
        curve = temporal_features.mean(dim=0)
    elif temporal_features.ndim == 2:
        curve = temporal_features
    else:
        raise ValueError("temporal_features must be [B,T,D] or [T,D]")
    return torch.stack([curve[stage.start : stage.end + 1].mean(dim=0) for stage in intervals], dim=0)


def partition_target_temporal_features(
    temporal_features: torch.Tensor,
    times: torch.Tensor,
    partitioner: Optional[BudgetedFeatureChangePartitioner] = None,
):
    """Partition target features using the same feature-change budgeted rule."""

    partitioner = partitioner or BudgetedFeatureChangePartitioner()
    intervals = partitioner(temporal_features, times)
    pooled = pool_stage_features(temporal_features, intervals)
    stage_times = torch.tensor(
        [stage.center_time for stage in intervals],
        device=temporal_features.device,
        dtype=temporal_features.dtype,
    )
    return intervals, pooled, stage_times
