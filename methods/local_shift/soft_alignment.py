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

        top_m = min(self.top_m, source_feats_all.shape[1])
        weights = target_stage_feats.new_zeros(batch, kmax, top_m)
        source_indices = torch.zeros(batch, kmax, top_m, device=target_stage_feats.device, dtype=torch.long)
        expected_centers = target_stage_feats.new_zeros(batch, kmax)
        valid_mask = torch.zeros(batch, kmax, device=target_stage_feats.device, dtype=torch.bool)

        for b in range(batch):
            if not pseudo_mask[b]:
                continue
            cls = int(pseudo_labels[b].item())
            if cls < 0 or cls >= source_feats_all.shape[0]:
                continue
            source_valid = source_mask_all[cls]
            if not bool(source_valid.any().item()):
                continue
            source_feats = source_feats_all[cls]
            source_centers = source_centers_all[cls]
            source_durations = source_durations_all[cls]

            for kt in range(kmax):
                if not target_stage_mask[b, kt]:
                    continue
                score = self._score(
                    target_stage_feats[b, kt],
                    target_stage_centers[b, kt] + global_shift[b],
                    target_stage_durations[b, kt],
                    source_feats,
                    source_centers,
                    source_durations,
                )
                score = score.masked_fill(~source_valid, float("-inf"))
                if not torch.isfinite(score).any():
                    continue
                values, indices = torch.topk(score, k=top_m, dim=0)
                probs = torch.softmax(values / self.temperature, dim=0)
                weights[b, kt, :top_m] = probs
                source_indices[b, kt, :top_m] = indices
                expected_centers[b, kt] = (probs * source_centers[indices]).sum()
                valid_mask[b, kt] = True

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
        return feature_score - self.time_gap_weight * time_penalty - self.duration_gap_weight * duration_penalty

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
