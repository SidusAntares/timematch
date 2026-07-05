import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_like_loss(reference):
    return reference.sum() * 0.0


def _segment_cost_matrix(sequence):
    """Return SSE cost for every contiguous segment [start, end)."""
    time_steps = int(sequence.shape[0])
    dtype = sequence.dtype
    device = sequence.device
    prefix = torch.zeros(time_steps + 1, sequence.shape[1], device=device, dtype=dtype)
    prefix_sq = torch.zeros(time_steps + 1, device=device, dtype=dtype)
    prefix[1:] = torch.cumsum(sequence, dim=0)
    prefix_sq[1:] = torch.cumsum(sequence.pow(2).sum(dim=1), dim=0)
    index = torch.arange(time_steps + 1, device=device)
    start = index[:, None]
    end = index[None, :]
    length = (end - start).to(dtype=dtype)
    valid = end > start
    summed = prefix[end] - prefix[start]
    summed_sq = prefix_sq[end] - prefix_sq[start]
    costs = summed_sq - summed.pow(2).sum(dim=-1) / length.clamp_min(1.0)
    return costs.masked_fill(~valid, float("inf"))


def _feature_change_dp_boundaries(sequence, stage_count, min_stage_len):
    """Find contiguous feature-change stages by dynamic programming.

    The boundary search runs on detached features. The returned boundaries are
    then used to pool the original tensor so gradients still flow through the
    selected stage means.
    """
    time_steps = int(sequence.shape[0])
    if time_steps <= 0 or stage_count <= 0:
        return [], float("inf"), "empty_sequence", False

    effective_stages = min(int(stage_count), time_steps)
    requested_min_len = max(1, int(min_stage_len))
    effective_min_len = min(requested_min_len, max(1, time_steps // effective_stages))
    reduced_min_len = effective_min_len < requested_min_len
    if effective_stages * effective_min_len > time_steps:
        effective_min_len = 1
        reduced_min_len = True

    costs = _segment_cost_matrix(sequence.detach())
    inf = float("inf")
    dp = torch.full((effective_stages + 1, time_steps + 1), inf, device=sequence.device, dtype=sequence.dtype)
    prev = torch.full((effective_stages + 1, time_steps + 1), -1, device=sequence.device, dtype=torch.long)
    dp[0, 0] = 0.0

    for stage_idx in range(1, effective_stages + 1):
        min_end = stage_idx * effective_min_len
        for end in range(min_end, time_steps + 1):
            start_min = (stage_idx - 1) * effective_min_len
            start_max = end - effective_min_len
            if start_max < start_min:
                continue
            candidates = dp[stage_idx - 1, start_min : start_max + 1] + costs[start_min : start_max + 1, end]
            best_value, best_offset = torch.min(candidates, dim=0)
            if torch.isfinite(best_value):
                best_start = start_min + int(best_offset.item())
                dp[stage_idx, end] = best_value
                prev[stage_idx, end] = best_start

    if not torch.isfinite(dp[effective_stages, time_steps]):
        return [], float("inf"), "dp_no_solution", reduced_min_len

    boundaries = []
    end = time_steps
    for stage_idx in range(effective_stages, 0, -1):
        start = int(prev[stage_idx, end].item())
        if start < 0:
            return [], float("inf"), "backtrack_failed", reduced_min_len
        boundaries.append((start, end))
        end = start
    boundaries.reverse()
    return boundaries, float(dp[effective_stages, time_steps].detach().item()), "", reduced_min_len


class AdaptiveTemporalStageExtractor(nn.Module):
    """Feature-change dynamic-programming temporal stage extractor."""

    def __init__(self, num_stages=6, min_stage_len=2, mode="feature_change_dp"):
        super().__init__()
        self.num_stages = int(num_stages)
        self.min_stage_len = int(min_stage_len)
        self.mode = str(mode)
        if self.mode != "feature_change_dp":
            raise ValueError(f"Unsupported stage partition mode: {mode}")

    def forward(self, features, positions=None):
        if features.dim() != 3:
            raise ValueError(f"Expected features [B,T,D], got {tuple(features.shape)}")
        batch_size, _, feat_dim = features.shape
        device = features.device
        dtype = features.dtype
        num_stages = max(1, self.num_stages)

        stage_feats = features.new_zeros(batch_size, num_stages, feat_dim)
        stage_mask = torch.zeros(batch_size, num_stages, device=device, dtype=torch.bool)
        stage_start_idx = torch.full((batch_size, num_stages), -1, device=device, dtype=torch.long)
        stage_end_idx = torch.full((batch_size, num_stages), -1, device=device, dtype=torch.long)
        stage_center_pos = features.new_zeros(batch_size, num_stages)

        costs = []
        lengths = []
        boundaries = []
        valid_samples = 0
        reduced_min_len_count = 0
        failed_count = 0

        for batch_idx in range(batch_size):
            sample = features[batch_idx]
            sample_positions = None if positions is None else positions[batch_idx].to(dtype=dtype)
            sample_boundaries, sample_cost, skip_reason, reduced_min_len = _feature_change_dp_boundaries(
                sample,
                num_stages,
                self.min_stage_len,
            )
            if reduced_min_len:
                reduced_min_len_count += 1
            if skip_reason:
                failed_count += 1
                continue
            valid_samples += 1
            costs.append(sample_cost)
            for stage_idx, (start, end) in enumerate(sample_boundaries[:num_stages]):
                segment = sample[start:end]
                stage_feats[batch_idx, stage_idx] = segment.mean(dim=0)
                stage_mask[batch_idx, stage_idx] = True
                stage_start_idx[batch_idx, stage_idx] = int(start)
                stage_end_idx[batch_idx, stage_idx] = int(end)
                if sample_positions is not None:
                    stage_center_pos[batch_idx, stage_idx] = sample_positions[start:end].mean()
                else:
                    stage_center_pos[batch_idx, stage_idx] = features.new_tensor((start + end - 1) * 0.5)
                lengths.append(float(end - start))
                boundaries.append(float(stage_center_pos[batch_idx, stage_idx].detach().item()))

        length_tensor = features.new_tensor(lengths) if lengths else features.new_zeros(1)
        boundary_tensor = features.new_tensor(boundaries) if boundaries else features.new_zeros(1)
        cost_tensor = features.new_tensor(costs) if costs else features.new_zeros(1)
        logs = {
            "stage_lengths_mean": float(length_tensor.mean().detach().item()) if lengths else 0.0,
            "stage_lengths_std": float(length_tensor.std(unbiased=False).detach().item()) if len(lengths) > 1 else 0.0,
            "stage_boundary_positions_mean": float(boundary_tensor.mean().detach().item()) if boundaries else 0.0,
            "stage_dp_cost_mean": float(cost_tensor.mean().detach().item()) if costs else 0.0,
            "stage_valid_ratio": float(valid_samples) / float(max(batch_size, 1)),
            "stage_reduced_min_len_count": float(reduced_min_len_count),
            "stage_failed_count": float(failed_count),
        }
        return {
            "stage_feats": stage_feats,
            "stage_mask": stage_mask,
            "stage_start_idx": stage_start_idx,
            "stage_end_idx": stage_end_idx,
            "stage_center_pos": stage_center_pos,
            "logs": logs,
        }


class ShiftAwareStageCorrespondence(nn.Module):
    """Soft source-stage correspondence around the TimeMatch global shift prior."""

    def __init__(self, stage_time_radius=30.0, stage_time_temperature=10.0, mode="shift_neighbor_soft"):
        super().__init__()
        self.stage_time_radius = float(stage_time_radius)
        self.stage_time_temperature = float(stage_time_temperature)
        self.mode = str(mode)
        if self.mode != "shift_neighbor_soft":
            raise ValueError(f"Unsupported stage correspondence mode: {mode}")

    def forward(
        self,
        source_stage_center_pos,
        target_stage_center_pos,
        source_stage_mask,
        target_stage_mask,
        target_to_source_shift,
    ):
        dtype = source_stage_center_pos.dtype
        shifted_target = target_stage_center_pos[:, None, :, None] + float(target_to_source_shift)
        source_center = source_stage_center_pos[None, :, None, :]
        time_gap = (shifted_target - source_center).abs()

        valid = target_stage_mask[:, None, :, None] & source_stage_mask[None, :, None, :]
        candidates = valid & (time_gap <= self.stage_time_radius)
        fallback_count = 0

        # Fallback to nearest valid source stage per target-stage/source-sample pair.
        no_candidate = (~candidates.any(dim=-1)) & target_stage_mask[:, None, :]
        source_has_stage = source_stage_mask.any(dim=1)
        for target_idx in range(time_gap.shape[0]):
            for source_idx in range(time_gap.shape[1]):
                if not bool(source_has_stage[source_idx].item()):
                    continue
                valid_source = source_stage_mask[source_idx]
                for target_stage_idx in range(time_gap.shape[2]):
                    if not bool(no_candidate[target_idx, source_idx, target_stage_idx].item()):
                        continue
                    if not bool(target_stage_mask[target_idx, target_stage_idx].item()):
                        continue
                    gaps = time_gap[target_idx, source_idx, target_stage_idx].masked_fill(~valid_source, float("inf"))
                    nearest = int(torch.argmin(gaps).item())
                    if torch.isfinite(gaps[nearest]):
                        candidates[target_idx, source_idx, target_stage_idx, nearest] = True
                        fallback_count += 1

        temperature = max(self.stage_time_temperature, 1e-6)
        weights = torch.exp(-time_gap.pow(2) / temperature).to(dtype=dtype)
        weights = weights * candidates.to(dtype=dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = torch.where(denom > 0.0, weights / denom, weights)

        row_mask = candidates.any(dim=-1)
        row_count = int(row_mask.sum().item())
        if row_count > 0:
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)
            candidate_count = candidates.to(dtype=dtype).sum(dim=-1)
            mean_gap = (weights * time_gap).sum(dim=-1)
            max_weight = weights.max(dim=-1).values
            logs = {
                "correspondence_entropy": float(entropy[row_mask].mean().detach().item()),
                "correspondence_valid_candidate_mean": float(candidate_count[row_mask].mean().detach().item()),
                "correspondence_mean_time_gap": float(mean_gap[row_mask].mean().detach().item()),
                "correspondence_fallback_count": float(fallback_count),
                "correspondence_max_weight_mean": float(max_weight[row_mask].mean().detach().item()),
            }
        else:
            logs = {
                "correspondence_entropy": 0.0,
                "correspondence_valid_candidate_mean": 0.0,
                "correspondence_mean_time_gap": 0.0,
                "correspondence_fallback_count": float(fallback_count),
                "correspondence_max_weight_mean": 0.0,
            }
        return weights, logs


class StageContrastiveLoss(nn.Module):
    """Class-conditional source-target stage-level prototype contrast."""

    def __init__(
        self,
        temperature=0.1,
        normalize=True,
        min_positive_count=1,
        min_negative_classes=1,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.min_positive_count = int(min_positive_count)
        self.min_negative_classes = int(min_negative_classes)

    def forward(
        self,
        source_stage_feats,
        source_stage_mask,
        source_labels,
        target_stage_feats,
        target_stage_mask,
        target_pseudo_labels,
        target_conf,
        target_mask,
        correspondence,
    ):
        loss_terms = []
        positive_counts = []
        negative_class_counts = []
        skipped_no_positive = 0
        skipped_no_negative = 0
        valid_queries = 0

        source_labels = source_labels.view(-1)
        target_pseudo_labels = target_pseudo_labels.view(-1)
        target_conf = target_conf.view(-1)
        target_mask = target_mask.view(-1).bool()
        classes = source_labels.unique(sorted=True)
        temperature = max(self.temperature, 1e-6)

        for target_idx in range(target_stage_feats.shape[0]):
            if not bool(target_mask[target_idx].item()):
                continue
            target_class = target_pseudo_labels[target_idx]
            for target_stage_idx in range(target_stage_feats.shape[1]):
                if not bool(target_stage_mask[target_idx, target_stage_idx].item()):
                    continue
                query = target_stage_feats[target_idx, target_stage_idx]
                if self.normalize:
                    query = F.normalize(query, dim=0)

                class_logits = []
                positive_logit_index = None
                pos_count_for_query = 0
                neg_count_for_query = 0

                for class_idx, class_id in enumerate(classes):
                    class_source = source_labels == class_id
                    weights = correspondence[target_idx, :, target_stage_idx, :] * class_source[:, None].to(
                        dtype=correspondence.dtype
                    )
                    weights = weights * source_stage_mask.to(dtype=weights.dtype)
                    weight_sum = weights.sum()
                    count = int((weights > 0).sum().item())
                    if int(class_id.item()) == int(target_class.item()):
                        pos_count_for_query = count
                        if count < self.min_positive_count or weight_sum <= 1e-12:
                            continue
                        positive_logit_index = len(class_logits)
                    else:
                        if weight_sum <= 1e-12:
                            continue
                        neg_count_for_query += 1

                    proto = (weights[..., None] * source_stage_feats).sum(dim=(0, 1)) / weight_sum.clamp_min(1e-12)
                    if self.normalize:
                        proto = F.normalize(proto, dim=0)
                    class_logits.append(torch.dot(query, proto) / temperature)

                if positive_logit_index is None:
                    skipped_no_positive += 1
                    continue
                if neg_count_for_query < self.min_negative_classes:
                    skipped_no_negative += 1
                    continue
                logits = torch.stack(class_logits).unsqueeze(0)
                label = torch.tensor([positive_logit_index], device=logits.device, dtype=torch.long)
                loss_terms.append(F.cross_entropy(logits, label))
                valid_queries += 1
                positive_counts.append(float(pos_count_for_query))
                negative_class_counts.append(float(neg_count_for_query))

        if loss_terms:
            loss = torch.stack(loss_terms).mean()
        else:
            loss = _zero_like_loss(target_stage_feats)

        conf_values = target_conf[target_mask]
        logs = {
            "stage_contrast_loss": float(loss.detach().item()),
            "stage_valid_queries": float(valid_queries),
            "stage_skipped_no_positive": float(skipped_no_positive),
            "stage_skipped_no_negative": float(skipped_no_negative),
            "stage_positive_count_mean": float(sum(positive_counts) / max(len(positive_counts), 1)),
            "stage_negative_class_count_mean": float(sum(negative_class_counts) / max(len(negative_class_counts), 1)),
            "stage_pseudo_coverage": float(target_mask.float().mean().detach().item()) if target_mask.numel() else 0.0,
            "stage_target_conf_mean": float(conf_values.mean().detach().item()) if conf_values.numel() else 0.0,
            "stage_temperature": float(self.temperature),
        }
        return loss, logs


def compute_adaptive_stage_contrast_loss(
    source_features,
    source_positions,
    source_labels,
    target_features,
    target_positions,
    target_pseudo_labels,
    target_conf,
    target_mask,
    target_to_source_shift,
    num_stages=6,
    stage_min_len=2,
    stage_partition_mode="feature_change_dp",
    stage_time_radius=30.0,
    stage_time_temperature=10.0,
    temperature=0.1,
    normalize=True,
):
    extractor = AdaptiveTemporalStageExtractor(
        num_stages=num_stages,
        min_stage_len=stage_min_len,
        mode=stage_partition_mode,
    )
    correspondence_module = ShiftAwareStageCorrespondence(
        stage_time_radius=stage_time_radius,
        stage_time_temperature=stage_time_temperature,
    )
    contrastive_loss = StageContrastiveLoss(
        temperature=temperature,
        normalize=normalize,
    )

    source_stage = extractor(source_features, source_positions)
    target_stage = extractor(target_features, target_positions)
    correspondence, correspondence_logs = correspondence_module(
        source_stage["stage_center_pos"],
        target_stage["stage_center_pos"],
        source_stage["stage_mask"],
        target_stage["stage_mask"],
        target_to_source_shift,
    )
    loss, contrast_logs = contrastive_loss(
        source_stage["stage_feats"],
        source_stage["stage_mask"],
        source_labels,
        target_stage["stage_feats"],
        target_stage["stage_mask"],
        target_pseudo_labels,
        target_conf,
        target_mask,
        correspondence,
    )
    logs = {}
    logs.update({f"source_{key}": value for key, value in source_stage["logs"].items()})
    logs.update({f"target_{key}": value for key, value in target_stage["logs"].items()})
    logs.update(correspondence_logs)
    logs.update(contrast_logs)
    return loss, logs
