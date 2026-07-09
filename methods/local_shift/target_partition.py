"""Budgeted feature-change partitioning for v3.2.1 local shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class StageInterval:
    """A contiguous temporal support interval."""

    start: int
    end: int
    center_time: float
    duration: float
    score: float = 0.0

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class BudgetedFeatureChangePartitioner:
    """Create variable stages from feature-change peaks.

    It is not uniform segmentation.  Boundaries are selected from temporal
    feature-change peaks with min-length, NMS, and Kmax constraints.
    """

    def __init__(
        self,
        kmax: int = 6,
        min_stage_len: int = 2,
        change_threshold: Optional[float] = None,
        change_quantile: Optional[float] = 0.75,
        nms_radius: int = 2,
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

    def __call__(self, temporal_features: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.partition_batch(temporal_features, positions)

    def partition_batch(self, temporal_features: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Partition a batch of temporal features.

        Args:
            temporal_features: Tensor shaped ``[B,T,D]``.
            positions: Tensor shaped ``[B,T]`` or ``[T]``.
        """

        if temporal_features.ndim != 3:
            raise ValueError("temporal_features must be shaped [B,T,D]")
        batch, steps, dim = temporal_features.shape
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        if positions.shape[:2] != (batch, steps):
            raise ValueError("positions must be shaped [B,T] or [T]")

        stage_feats = temporal_features.new_zeros(batch, self.kmax, dim)
        stage_mask = torch.zeros(batch, self.kmax, device=temporal_features.device, dtype=torch.bool)
        stage_centers = temporal_features.new_zeros(batch, self.kmax)
        stage_durations = temporal_features.new_zeros(batch, self.kmax)
        stage_to_time = torch.full((batch, steps), -1, device=temporal_features.device, dtype=torch.long)
        stage_count = torch.zeros(batch, device=temporal_features.device, dtype=torch.long)

        change = (temporal_features[:, 1:] - temporal_features[:, :-1]).float().pow(2).mean(dim=-1).sqrt()
        if steps == 1 or self.kmax == 1:
            cut_values = torch.empty(batch, 0, device=temporal_features.device, dtype=torch.long)
            cut_valid = torch.empty(batch, 0, device=temporal_features.device, dtype=torch.bool)
        else:
            if self.change_threshold is not None:
                threshold = torch.full((batch, 1), float(self.change_threshold), device=change.device, dtype=change.dtype)
            elif self.change_quantile is not None:
                threshold = torch.quantile(change, float(self.change_quantile), dim=1, keepdim=True)
            else:
                threshold = change.mean(dim=1, keepdim=True)
            cut_positions = torch.arange(1, steps, device=change.device).view(1, -1)
            candidate_mask = (change >= threshold) & (cut_positions >= self.min_stage_len) & ((steps - cut_positions) >= self.min_stage_len)
            if self.nms_radius > 0:
                pooled = torch.nn.functional.max_pool1d(
                    change.unsqueeze(1),
                    kernel_size=2 * self.nms_radius + 1,
                    stride=1,
                    padding=self.nms_radius,
                ).squeeze(1)
                candidate_mask = candidate_mask & (change >= pooled)
            score = change.masked_fill(~candidate_mask, -float("inf"))
            topk = min(self.kmax - 1, score.shape[1])
            if topk > 0:
                top_values, top_indices = torch.topk(score, k=topk, dim=1)
                cut_values = (top_indices + 1).sort(dim=1).values
                cut_valid = torch.isfinite(top_values).gather(1, torch.argsort(top_indices + 1, dim=1))
            else:
                cut_values = torch.empty(batch, 0, device=temporal_features.device, dtype=torch.long)
                cut_valid = torch.empty(batch, 0, device=temporal_features.device, dtype=torch.bool)

        interval_records: List[List[StageInterval]] = []
        for b in range(batch):
            cuts: List[int] = []
            prev = 0
            for idx in range(cut_values.shape[1]):
                if not bool(cut_valid[b, idx].item()):
                    continue
                cut = int(cut_values[b, idx].item())
                if cut - prev < self.min_stage_len:
                    continue
                if steps - cut < self.min_stage_len:
                    continue
                cuts.append(cut)
                prev = cut
                if len(cuts) >= self.kmax - 1:
                    break
            bounds = [0] + cuts + [steps]
            intervals: List[StageInterval] = []
            for k, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
                if k >= self.kmax:
                    break
                start = int(left)
                end = int(right) - 1
                stage_slice = slice(start, end + 1)
                stage_feats[b, k] = temporal_features[b, stage_slice].mean(dim=0)
                stage_mask[b, k] = True
                stage_positions = positions[b, stage_slice].float()
                center = stage_positions.mean()
                duration = stage_positions.max() - stage_positions.min()
                if stage_positions.numel() == 1:
                    duration = torch.ones_like(duration)
                stage_centers[b, k] = center
                stage_durations[b, k] = duration
                stage_to_time[b, stage_slice] = k
                intervals.append(
                    StageInterval(
                        start=start,
                        end=end,
                        center_time=float(center.detach().cpu()),
                        duration=float(duration.detach().cpu()),
                        score=0.0,
                    )
                )
            stage_count[b] = len(intervals)
            interval_records.append(intervals)

        logs = {
            "stage_count_mean": float(stage_count.float().mean().detach().cpu()),
            "stage_count_max": int(stage_count.max().detach().cpu()),
            "stage_count_p90": float(torch.quantile(stage_count.float(), 0.9).detach().cpu()),
        }
        return {
            "stage_feats": stage_feats,
            "stage_mask": stage_mask,
            "stage_centers": stage_centers,
            "stage_durations": stage_durations,
            "stage_to_time": stage_to_time,
            "stage_count": stage_count,
            "intervals": interval_records,
            "logs": logs,
        }

    def partition_curve(self, temporal_curve: torch.Tensor, positions: torch.Tensor) -> List[StageInterval]:
        if temporal_curve.ndim != 2:
            raise ValueError("temporal_curve must be shaped [T,D]")
        if positions.ndim != 1 or positions.shape[0] != temporal_curve.shape[0]:
            raise ValueError("positions must be shaped [T]")
        steps = temporal_curve.shape[0]
        if steps == 0:
            raise ValueError("empty temporal feature curve")
        if steps == 1 or self.kmax == 1:
            return [self._interval(0, steps - 1, positions, 0.0)]

        change = (temporal_curve[1:] - temporal_curve[:-1]).float().pow(2).mean(dim=-1).sqrt()
        if float(change.max().detach().cpu()) <= 1e-12:
            return [self._interval(0, steps - 1, positions, 0.0)]
        threshold = self._threshold(change)
        candidates = torch.nonzero(change >= threshold, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return [self._interval(0, steps - 1, positions, 0.0)]

        order = torch.argsort(change[candidates], descending=True)
        selected: List[int] = []
        for candidate in candidates[order].tolist():
            cut = int(candidate) + 1
            if cut < self.min_stage_len or steps - cut < self.min_stage_len:
                continue
            if any(abs(cut - prev) <= self.nms_radius for prev in selected):
                continue
            selected.append(cut)
            if len(selected) >= self.kmax - 1:
                break
        return self._cuts_to_intervals(sorted(selected), positions, change)

    def _threshold(self, change: torch.Tensor) -> torch.Tensor:
        if self.change_threshold is not None:
            return torch.as_tensor(self.change_threshold, device=change.device, dtype=change.dtype)
        if self.change_quantile is not None:
            return torch.quantile(change, float(self.change_quantile))
        return change.mean()

    def _cuts_to_intervals(
        self,
        cuts: Sequence[int],
        positions: torch.Tensor,
        change: torch.Tensor,
    ) -> List[StageInterval]:
        steps = int(positions.shape[0])
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
            intervals.append(self._interval(start, end, positions, score))
        return intervals or [self._interval(0, steps - 1, positions, 0.0)]

    @staticmethod
    def _interval(start: int, end: int, positions: torch.Tensor, score: float) -> StageInterval:
        stage_positions = positions[start : end + 1].float()
        center = stage_positions.mean()
        duration = stage_positions.max() - stage_positions.min()
        if stage_positions.numel() == 1:
            duration = torch.ones_like(duration)
        return StageInterval(
            start=start,
            end=end,
            center_time=float(center.detach().cpu()),
            duration=float(duration.detach().cpu()),
            score=score,
        )


def pool_stage_features(temporal_features: torch.Tensor, intervals: Sequence[StageInterval]) -> torch.Tensor:
    """Average-pool one temporal curve over variable intervals."""

    if temporal_features.ndim != 2:
        raise ValueError("temporal_features must be shaped [T,D]")
    return torch.stack([temporal_features[stage.start : stage.end + 1].mean(dim=0) for stage in intervals], dim=0)


def partition_target_temporal_features(
    temporal_features: torch.Tensor,
    positions: torch.Tensor,
    partitioner: Optional[BudgetedFeatureChangePartitioner] = None,
) -> Dict[str, torch.Tensor]:
    """Partition target features using the feature-change budgeted rule."""

    partitioner = partitioner or BudgetedFeatureChangePartitioner()
    return partitioner.partition_batch(temporal_features, positions)
