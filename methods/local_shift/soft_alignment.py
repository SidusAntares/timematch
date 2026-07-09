"""Soft stage alignment for v3.2.1 local shift."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class SoftStageAligner:
    """Align target stages to pseudo-label-conditioned source stages.

    The alignment is used to estimate local temporal shift.  It is not a stage
    contrast loss and does not use target true labels.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        top_m: int = 2,
        time_gap_weight: float = 1.0,
        duration_gap_weight: float = 0.25,
        feature_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_m < 1:
            raise ValueError("top_m must be >= 1")
        self.temperature = float(temperature)
        self.top_m = int(top_m)
        self.time_gap_weight = float(time_gap_weight)
        self.duration_gap_weight = float(duration_gap_weight)
        self.feature_weight = float(feature_weight)
        self.eps = float(eps)

    def __call__(
        self,
        target_stage_feats: torch.Tensor,
        target_stage_centers: torch.Tensor,
        target_stage_durations: torch.Tensor,
        target_stage_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
        pseudo_mask: torch.Tensor,
        source_reference: Dict[str, torch.Tensor],
        global_shift,
    ) -> Dict[str, torch.Tensor]:
        if target_stage_feats.ndim != 3:
            raise ValueError("target_stage_feats must be [B,K,D]")
        batch, kmax, dim = target_stage_feats.shape
        if target_stage_centers.shape != (batch, kmax):
            raise ValueError("target_stage_centers must be [B,K]")
        if target_stage_durations.shape != (batch, kmax):
            raise ValueError("target_stage_durations must be [B,K]")
        if target_stage_mask.shape != (batch, kmax):
            raise ValueError("target_stage_mask must be [B,K]")

        source_feats_all = source_reference["class_stage_feats"].to(target_stage_feats.device)
        source_centers_all = source_reference["class_stage_centers"].to(target_stage_feats.device)
        source_durations_all = source_reference["class_stage_durations"].to(target_stage_feats.device)
        source_mask_all = source_reference["class_stage_mask"].to(target_stage_feats.device)
        if source_feats_all.shape[-1] != dim:
            raise ValueError("target/source feature dimensions do not match")

        pseudo_labels = pseudo_labels.to(target_stage_feats.device).long()
        pseudo_mask = pseudo_mask.to(target_stage_feats.device).bool()
        global_shift = _expand_global_shift(global_shift, batch, target_stage_feats.device, target_stage_feats.dtype)

        class_count = source_feats_all.shape[0]
        source_k = source_feats_all.shape[1]
        top_m = min(self.top_m, source_k)
        safe_labels = pseudo_labels.clamp(min=0, max=class_count - 1)
        label_in_range = (pseudo_labels >= 0) & (pseudo_labels < class_count)
        src_feats = source_feats_all[safe_labels]
        src_centers = source_centers_all[safe_labels]
        src_durations = source_durations_all[safe_labels]
        src_mask = source_mask_all[safe_labels]

        target_unit = F.normalize(target_stage_feats, dim=-1, eps=self.eps)
        src_unit = F.normalize(src_feats, dim=-1, eps=self.eps)
        feature_score = torch.einsum("bkd,bsd->bks", target_unit, src_unit)

        src_centers_max = src_centers.masked_fill(~src_mask, -float("inf")).max(dim=1).values
        src_centers_min = src_centers.masked_fill(~src_mask, float("inf")).min(dim=1).values
        source_time_scale = (src_centers_max - src_centers_min)
        source_time_scale = torch.where(torch.isfinite(source_time_scale), source_time_scale, torch.ones_like(source_time_scale))
        source_time_scale = source_time_scale.clamp_min(1.0)
        duration_sum = (src_durations * src_mask.float()).sum(dim=1)
        duration_count = src_mask.float().sum(dim=1).clamp_min(1.0)
        duration_scale = (duration_sum / duration_count).clamp_min(1.0)

        shifted_target_centers = target_stage_centers + global_shift.unsqueeze(1)
        time_penalty = (src_centers.unsqueeze(1) - shifted_target_centers.unsqueeze(2)).abs() / source_time_scale.view(batch, 1, 1)
        duration_penalty = (src_durations.unsqueeze(1) - target_stage_durations.unsqueeze(2)).abs() / duration_scale.view(batch, 1, 1)
        score = (
            self.feature_weight * feature_score
            - self.time_gap_weight * time_penalty
            - self.duration_gap_weight * duration_penalty
        )

        valid_score_mask = (
            target_stage_mask.unsqueeze(2)
            & src_mask.unsqueeze(1)
            & pseudo_mask.view(batch, 1, 1)
            & label_in_range.view(batch, 1, 1)
        )
        score = score.masked_fill(~valid_score_mask, float("-inf"))
        values, source_indices = torch.topk(score, k=top_m, dim=2)
        finite = torch.isfinite(values)
        safe_values = values.masked_fill(~finite, -1e9)
        weights = torch.softmax(safe_values / self.temperature, dim=2)
        weights = torch.where(finite, weights, torch.zeros_like(weights))
        weight_sum = weights.sum(dim=2, keepdim=True).clamp_min(self.eps)
        weights = weights / weight_sum
        valid_mask = finite.any(dim=2)
        gathered_centers = torch.gather(src_centers.unsqueeze(1).expand(-1, kmax, -1), dim=2, index=source_indices)
        expected_centers = (weights * gathered_centers).sum(dim=2)
        source_indices = source_indices.masked_fill(~finite, 0)

        logs = self._logs(weights, valid_mask, target_stage_mask, pseudo_mask)
        return {
            "weights": weights,
            "source_indices": source_indices,
            "expected_source_centers": expected_centers,
            "valid_mask": valid_mask,
            "logs": logs,
        }

    def _score(
        self,
        target_feat: torch.Tensor,
        shifted_target_center: torch.Tensor,
        target_duration: torch.Tensor,
        source_feats: torch.Tensor,
        source_centers: torch.Tensor,
        source_durations: torch.Tensor,
    ) -> torch.Tensor:
        target_unit = F.normalize(target_feat.unsqueeze(0), dim=-1, eps=self.eps)
        source_unit = F.normalize(source_feats, dim=-1, eps=self.eps)
        feature_score = (target_unit * source_unit).sum(dim=-1)

        source_time_scale = source_centers[source_centers > 0]
        if source_time_scale.numel() > 1:
            time_scale = (source_time_scale.max() - source_time_scale.min()).clamp_min(1.0)
        else:
            time_scale = torch.ones((), device=source_centers.device, dtype=source_centers.dtype)
        duration_scale = source_durations[source_durations > 0].mean().clamp_min(1.0)

        time_penalty = (source_centers - shifted_target_center).abs() / time_scale
        duration_penalty = (source_durations - target_duration).abs() / duration_scale
        return (
            self.feature_weight * feature_score
            - self.time_gap_weight * time_penalty
            - self.duration_gap_weight * duration_penalty
        )

    @staticmethod
    def _logs(
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
        target_stage_mask: torch.Tensor,
        pseudo_mask: torch.Tensor,
    ) -> Dict[str, float]:
        valid_weights = weights[valid_mask]
        if valid_weights.numel() == 0:
            entropy = 0.0
            top1_mass = 0.0
        else:
            entropy = float((-(valid_weights.clamp_min(1e-12).log() * valid_weights).sum(dim=-1)).mean().item())
            top1_mass = float(valid_weights.max(dim=-1).values.mean().item())
        possible = target_stage_mask & pseudo_mask.unsqueeze(1)
        possible_count = possible.float().sum().clamp_min(1.0)
        valid_ratio = float((valid_mask.float().sum() / possible_count).item())
        fallback_ratio = float(((possible & ~valid_mask).float().sum() / possible_count).item())
        return {
            "alignment_entropy": entropy,
            "top1_mass": top1_mass,
            "valid_ratio": valid_ratio,
            "fallback_ratio": fallback_ratio,
        }


def _expand_global_shift(global_shift, batch: int, device, dtype) -> torch.Tensor:
    if torch.is_tensor(global_shift):
        shift = global_shift.to(device=device, dtype=dtype)
        if shift.ndim == 0:
            return shift.view(1).expand(batch)
        if shift.shape == (batch,):
            return shift
        raise ValueError("global_shift tensor must be scalar or [B]")
    return torch.full((batch,), float(global_shift), device=device, dtype=dtype)
