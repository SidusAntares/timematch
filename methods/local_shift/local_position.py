"""Local position utilities for v3.2.1 stage-wise residual shift."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_stage_residual_shift(
    target_stage_centers: torch.Tensor,
    expected_source_centers: torch.Tensor,
    global_shift,
    local_shift_clip: Optional[float] = None,
    stage_mask: Optional[torch.Tensor] = None,
    return_logs: bool = False,
):
    """Compute stage residual shift.

    Formula:

    ``residual_k = expected_source_center_k - (target_center_k + global_shift)``
    """

    if target_stage_centers.shape != expected_source_centers.shape:
        raise ValueError("target_stage_centers and expected_source_centers must have the same shape")
    batch = target_stage_centers.shape[0]
    global_shift = _expand_global_shift(global_shift, batch, target_stage_centers.device, target_stage_centers.dtype)
    residual = expected_source_centers - (target_stage_centers + global_shift.unsqueeze(1))
    raw_residual = residual
    if local_shift_clip is not None:
        residual = residual.clamp(min=-float(local_shift_clip), max=float(local_shift_clip))
    if not return_logs:
        return residual

    if stage_mask is None:
        valid = torch.ones_like(residual, dtype=torch.bool)
    else:
        valid = stage_mask.bool()
    valid_residual = residual[valid]
    if valid_residual.numel() == 0:
        logs = {
            "residual_mean": 0.0,
            "residual_std": 0.0,
            "residual_abs_mean": 0.0,
            "residual_clip_fraction": 0.0,
        }
    else:
        clipped = (raw_residual != residual) & valid
        logs = {
            "residual_mean": float(valid_residual.mean().item()),
            "residual_std": float(valid_residual.std(unbiased=False).item()),
            "residual_abs_mean": float(valid_residual.abs().mean().item()),
            "residual_clip_fraction": float(clipped.float().sum().item() / valid.float().sum().clamp_min(1.0).item()),
        }
    return residual, logs


def expand_stage_shift_to_time(
    base_positions: torch.Tensor,
    stage_to_time: torch.Tensor,
    stage_shift: torch.Tensor,
    stage_mask: torch.Tensor,
    global_shift,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Expand stage residual shifts to per-time local positions.

    ``positions_local = positions + global_shift + residual_shift_per_time``
    """

    if base_positions.ndim != 2:
        raise ValueError("base_positions must be [B,T]")
    batch, steps = base_positions.shape
    if stage_to_time.shape != (batch, steps):
        raise ValueError("stage_to_time must be [B,T]")
    if stage_shift.ndim != 2 or stage_shift.shape[0] != batch:
        raise ValueError("stage_shift must be [B,K]")
    if stage_mask.shape != stage_shift.shape:
        raise ValueError("stage_mask must match stage_shift")

    global_shift = _expand_global_shift(global_shift, batch, base_positions.device, base_positions.dtype)
    per_time_shift = torch.zeros_like(base_positions, dtype=base_positions.dtype)
    valid_time = stage_to_time >= 0
    for b in range(batch):
        for t in range(steps):
            stage_idx = int(stage_to_time[b, t].item())
            if stage_idx >= 0 and stage_idx < stage_shift.shape[1] and bool(stage_mask[b, stage_idx].item()):
                per_time_shift[b, t] = stage_shift[b, stage_idx]
    local_positions = base_positions.float() + global_shift.unsqueeze(1) + per_time_shift
    valid_residual = stage_shift[stage_mask]
    if valid_residual.numel() == 0:
        residual_mean = residual_std = residual_abs_mean = 0.0
    else:
        residual_mean = float(valid_residual.mean().item())
        residual_std = float(valid_residual.std(unbiased=False).item())
        residual_abs_mean = float(valid_residual.abs().mean().item())
    logs = {
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_abs_mean": residual_abs_mean,
        "valid_time_ratio": float(valid_time.float().mean().item()),
        "residual_clip_fraction": 0.0,
    }
    return local_positions, logs


def check_temporal_positions_in_range(
    positions: torch.Tensor,
    min_position: float = 0.0,
    max_position: Optional[float] = None,
    round_to_long: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clamp positions to positional embedding range and report clamp ratio."""

    before = positions
    if max_position is None:
        after = before.clamp(min=float(min_position))
    else:
        after = before.clamp(min=float(min_position), max=float(max_position))
    clamp_ratio = float((after != before).float().mean().item())
    if round_to_long:
        after = after.round().long()
    return after, {"position_clamp_ratio": clamp_ratio}


def summarize_residual_clip(stage_shift: torch.Tensor, clipped_shift: torch.Tensor, stage_mask: torch.Tensor) -> float:
    if stage_mask.float().sum() <= 0:
        return 0.0
    return float(((stage_shift != clipped_shift) & stage_mask).float().sum().item() / stage_mask.float().sum().item())


def _expand_global_shift(global_shift, batch: int, device, dtype) -> torch.Tensor:
    if torch.is_tensor(global_shift):
        shift = global_shift.to(device=device, dtype=dtype)
        if shift.ndim == 0:
            return shift.view(1).expand(batch)
        if shift.shape == (batch,):
            return shift
        raise ValueError("global_shift tensor must be scalar or [B]")
    return torch.full((batch,), float(global_shift), device=device, dtype=dtype)
