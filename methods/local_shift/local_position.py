"""Local position utilities for v3.2.1 stage-wise residual shift."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from methods.local_shift.target_partition import StageInterval


def compute_stage_residual_shift(
    target_stage_times: torch.Tensor,
    expected_source_times: torch.Tensor,
    clip: Optional[float] = None,
) -> torch.Tensor:
    """Compute residual local shift per target stage.

    Positive values mean the target stage is shifted toward a later source
    reference time.
    """

    shift = expected_source_times - target_stage_times
    if clip is not None:
        shift = shift.clamp(min=-float(clip), max=float(clip))
    return shift


def expand_stage_shift_to_time(
    base_positions: torch.Tensor,
    intervals: Sequence[StageInterval],
    stage_shift: torch.Tensor,
) -> torch.Tensor:
    """Expand stage shifts to per-time positions.

    Args:
        base_positions: ``[T]`` or ``[B,T]``.
        intervals: Target stage intervals.
        stage_shift: ``[K]`` or ``[B,K]``.
    """

    positions = base_positions.clone().float()
    if positions.ndim == 1:
        for idx, interval in enumerate(intervals):
            positions[interval.start : interval.end + 1] += stage_shift[idx]
        return positions

    if positions.ndim == 2:
        for idx, interval in enumerate(intervals):
            positions[:, interval.start : interval.end + 1] += stage_shift[:, idx].unsqueeze(-1)
        return positions

    raise ValueError("base_positions must be [T] or [B,T]")


def check_temporal_positions_in_range(
    positions: torch.Tensor,
    min_position: float = 0.0,
    max_position: Optional[float] = None,
) -> torch.Tensor:
    """Clamp positions to the supported temporal range."""

    if max_position is None:
        return positions.clamp(min=float(min_position))
    return positions.clamp(min=float(min_position), max=float(max_position))
