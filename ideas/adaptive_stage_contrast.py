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


def build_class_stage_prototypes(
    source_stage_feats,
    source_stage_mask,
    source_stage_center_pos,
    source_labels,
    num_classes,
    eps=1e-6,
):
    if source_stage_feats.dim() != 3:
        raise ValueError(f"Expected source_stage_feats [B,K,D], got {tuple(source_stage_feats.shape)}")
    if source_stage_mask.shape != source_stage_feats.shape[:2]:
        raise ValueError(
            "source_stage_mask must match source_stage_feats[:2], "
            f"got {tuple(source_stage_mask.shape)} vs {tuple(source_stage_feats.shape[:2])}"
        )
    if source_stage_center_pos.shape != source_stage_feats.shape[:2]:
        raise ValueError(
            "source_stage_center_pos must match source_stage_feats[:2], "
            f"got {tuple(source_stage_center_pos.shape)} vs {tuple(source_stage_feats.shape[:2])}"
        )

    source_labels = source_labels.view(-1).long()
    if source_labels.numel() != source_stage_feats.shape[0]:
        raise ValueError("source_labels length must match source batch size")
    if source_labels.numel() > 0:
        if int(source_labels.min().item()) < 0 or int(source_labels.max().item()) >= int(num_classes):
            raise ValueError(
                f"source_labels must be in [0, {int(num_classes) - 1}], "
                f"got min={int(source_labels.min().item())}, max={int(source_labels.max().item())}"
            )

    batch_size, stage_count, feat_dim = source_stage_feats.shape
    dtype = source_stage_feats.dtype
    device = source_stage_feats.device
    num_classes = int(num_classes)
    mask_float = source_stage_mask.to(dtype=dtype)

    proto_sum = source_stage_feats.new_zeros(num_classes, stage_count, feat_dim)
    proto_index = source_labels.view(batch_size, 1, 1).expand(batch_size, stage_count, feat_dim)
    proto_sum.scatter_add_(0, proto_index, source_stage_feats * mask_float[..., None])

    count = source_stage_feats.new_zeros(num_classes, stage_count)
    count_index = source_labels.view(batch_size, 1).expand(batch_size, stage_count)
    count.scatter_add_(0, count_index, mask_float)

    center_sum = source_stage_feats.new_zeros(num_classes, stage_count)
    center_sum.scatter_add_(0, count_index, source_stage_center_pos.to(dtype=dtype) * mask_float)

    source_proto_mask = count > 0
    source_proto = proto_sum / count.clamp_min(eps)[..., None]
    source_center = center_sum / count.clamp_min(eps)
    source_class_count = torch.bincount(source_labels, minlength=num_classes).to(device=device, dtype=dtype)

    return {
        "source_proto": source_proto,
        "source_center": source_center,
        "source_proto_mask": source_proto_mask,
        "source_class_count": source_class_count,
        "source_class_stage_count": count,
    }


class FastShiftAwareClassStageCorrespondence(nn.Module):
    """Vectorized class-stage correspondence A[B_t,K_t,C,K_s]."""

    def __init__(self, stage_time_radius=30.0, stage_time_temperature=10.0, eps=1e-6):
        super().__init__()
        self.stage_time_radius = float(stage_time_radius)
        self.stage_time_temperature = float(stage_time_temperature)
        self.eps = float(eps)

    def forward(
        self,
        target_stage_center_pos,
        target_stage_mask,
        source_center,
        source_proto_mask,
        target_to_source_shift,
    ):
        dtype = source_center.dtype
        shifted_target = target_stage_center_pos.to(dtype=dtype)[:, :, None, None] + float(target_to_source_shift)
        gap = (shifted_target - source_center[None, None, :, :]).abs()

        target_valid = target_stage_mask[:, :, None, None]
        source_valid = source_proto_mask[None, None, :, :]
        candidate = (gap <= self.stage_time_radius) & source_valid & target_valid

        class_has_stage = source_proto_mask.any(dim=-1)
        valid_pair = target_stage_mask[:, :, None] & class_has_stage[None, None, :]
        no_candidate = (~candidate.any(dim=-1)) & valid_pair

        masked_gap = gap.masked_fill(~source_valid, float("inf"))
        nearest_idx = masked_gap.argmin(dim=-1, keepdim=True)
        fallback = torch.zeros_like(candidate)
        fallback.scatter_(-1, nearest_idx, True)
        fallback = fallback & no_candidate[..., None] & source_valid
        candidate_final = candidate | fallback

        scores = -gap.pow(2) / max(self.stage_time_temperature, self.eps)
        scores = scores.masked_fill(~candidate_final, -1.0e9)
        weights = torch.softmax(scores, dim=-1) * candidate_final.to(dtype=dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weights = weights * valid_pair[..., None].to(dtype=dtype)

        valid_entries = candidate_final.any(dim=-1)
        valid_count = int(valid_entries.sum().item())
        fallback_count = float(fallback.sum().detach().item())
        valid_pair_count = float(valid_pair.sum().detach().item())
        if valid_count > 0:
            weights_safe = weights.clamp_min(1e-12)
            entropy = -(weights_safe * weights_safe.log()).sum(dim=-1)
            candidate_count = candidate_final.to(dtype=dtype).sum(dim=-1)
            mean_gap = (weights * gap).sum(dim=-1)
            max_weight = weights.max(dim=-1).values
            valid_class_count = (weights.sum(dim=-1) > 0).to(dtype=dtype).sum(dim=-1)
            logs = {
                "correspondence_entropy": float(entropy[valid_entries].mean().detach().item()),
                "correspondence_valid_candidate_mean": float(candidate_count[valid_entries].mean().detach().item()),
                "correspondence_mean_time_gap": float(mean_gap[valid_entries].mean().detach().item()),
                "correspondence_fallback_count": fallback_count,
                "correspondence_fallback_ratio": fallback_count / max(valid_pair_count, 1.0),
                "correspondence_max_weight_mean": float(max_weight[valid_entries].mean().detach().item()),
                "correspondence_valid_class_mean": float(valid_class_count[target_stage_mask].mean().detach().item())
                if bool(target_stage_mask.any().item())
                else 0.0,
            }
        else:
            logs = {
                "correspondence_entropy": 0.0,
                "correspondence_valid_candidate_mean": 0.0,
                "correspondence_mean_time_gap": 0.0,
                "correspondence_fallback_count": fallback_count,
                "correspondence_fallback_ratio": fallback_count / max(valid_pair_count, 1.0),
                "correspondence_max_weight_mean": 0.0,
                "correspondence_valid_class_mean": 0.0,
            }
        return weights, logs


class FastClassPrototypeStageContrastiveLoss(nn.Module):
    """Vectorized class-prototype stage contrast.

    This keeps the class-conditional stage contrast objective but replaces the
    dense sample-pair correspondence A[B_t,B_s,K,K] with class-stage prototypes
    and A[B_t,K,C,K].
    """

    def __init__(self, temperature=0.1, normalize=True, eps=1e-6):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def forward(
        self,
        source_stage_feats,
        source_stage_mask,
        source_stage_center_pos,
        source_labels,
        target_stage_feats,
        target_stage_mask,
        target_stage_center_pos,
        target_pseudo_labels,
        target_conf,
        target_mask,
        target_to_source_shift,
        num_classes,
        stage_time_radius=30.0,
        stage_time_temperature=10.0,
    ):
        prototypes = build_class_stage_prototypes(
            source_stage_feats,
            source_stage_mask,
            source_stage_center_pos,
            source_labels,
            num_classes=num_classes,
            eps=self.eps,
        )
        correspondence_module = FastShiftAwareClassStageCorrespondence(
            stage_time_radius=stage_time_radius,
            stage_time_temperature=stage_time_temperature,
            eps=self.eps,
        )
        correspondence, correspondence_logs = correspondence_module(
            target_stage_center_pos,
            target_stage_mask,
            prototypes["source_center"],
            prototypes["source_proto_mask"],
            target_to_source_shift,
        )

        proto_for_target = torch.einsum(
            "bkcl,cld->bkcd",
            correspondence,
            prototypes["source_proto"],
        )
        query = target_stage_feats
        if self.normalize:
            query = F.normalize(query, dim=-1, eps=self.eps)
            proto_for_target = F.normalize(proto_for_target, dim=-1, eps=self.eps)
        logits = torch.einsum("bkd,bkcd->bkc", query, proto_for_target) / max(self.temperature, self.eps)

        class_valid = correspondence.sum(dim=-1) > 0
        logits = logits.masked_fill(~class_valid, -1.0e9)

        target_pseudo_labels = target_pseudo_labels.view(-1).long()
        target_conf = target_conf.view(-1)
        target_mask = target_mask.view(-1).bool()
        batch_size, stage_count, num_classes = logits.shape
        label_valid = (target_pseudo_labels >= 0) & (target_pseudo_labels < num_classes)
        safe_labels = target_pseudo_labels.clamp(min=0, max=max(num_classes - 1, 0))
        labels = safe_labels[:, None].expand(batch_size, stage_count)
        positive_valid = class_valid.gather(dim=-1, index=labels[..., None]).squeeze(-1)
        valid_class_count = class_valid.to(dtype=logits.dtype).sum(dim=-1)
        query_valid = (
            target_mask[:, None]
            & target_stage_mask
            & label_valid[:, None]
            & positive_valid
            & (valid_class_count >= 2)
        )

        logits_flat = logits.reshape(batch_size * stage_count, num_classes)
        labels_flat = labels.reshape(batch_size * stage_count)
        valid_flat = query_valid.reshape(batch_size * stage_count)
        if bool(valid_flat.any().item()):
            loss = F.cross_entropy(logits_flat[valid_flat], labels_flat[valid_flat])
        else:
            loss = _zero_like_loss(target_stage_feats)

        invalid_target_queries = (target_mask[:, None] & target_stage_mask & (~label_valid[:, None])).sum()
        no_positive = (target_mask[:, None] & target_stage_mask & label_valid[:, None] & (~positive_valid)).sum()
        no_negative = (
            target_mask[:, None]
            & target_stage_mask
            & label_valid[:, None]
            & positive_valid
            & (valid_class_count < 2)
        ).sum()
        conf_values = target_conf[target_mask]
        source_valid_class_count = (prototypes["source_class_count"] > 0).sum()
        source_valid_class_stage_count = prototypes["source_proto_mask"].sum()

        logs = {
            "stage_contrast_loss": float(loss.detach().item()),
            "stage_valid_queries": float(query_valid.sum().detach().item()),
            "stage_skipped_invalid_target": float(invalid_target_queries.detach().item()),
            "stage_skipped_no_positive": float(no_positive.detach().item()),
            "stage_skipped_no_negative": float(no_negative.detach().item()),
            "stage_valid_class_count_mean": float(valid_class_count[target_stage_mask].mean().detach().item())
            if bool(target_stage_mask.any().item())
            else 0.0,
            "stage_positive_count_mean": float(
                prototypes["source_class_stage_count"][safe_labels].sum(dim=-1)[target_mask].mean().detach().item()
            )
            if bool(target_mask.any().item())
            else 0.0,
            "stage_negative_class_count_mean": 0.0,
            "stage_pseudo_coverage": float(target_mask.to(dtype=logits.dtype).mean().detach().item()) if target_mask.numel() else 0.0,
            "stage_target_conf_mean": float(conf_values.mean().detach().item()) if conf_values.numel() else 0.0,
            "stage_temperature": float(self.temperature),
            "source_valid_class_count": float(source_valid_class_count.detach().item()),
            "source_valid_class_stage_count": float(source_valid_class_stage_count.detach().item()),
        }
        logs.update(correspondence_logs)
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
    num_classes=None,
    backend="class_prototype_fast",
):
    extractor = AdaptiveTemporalStageExtractor(
        num_stages=num_stages,
        min_stage_len=stage_min_len,
        mode=stage_partition_mode,
    )
    source_stage = extractor(source_features, source_positions)
    target_stage = extractor(target_features, target_positions)
    if str(backend) != "class_prototype_fast":
        correspondence_module = ShiftAwareStageCorrespondence(
            stage_time_radius=stage_time_radius,
            stage_time_temperature=stage_time_temperature,
        )
        contrastive_loss = StageContrastiveLoss(
            temperature=temperature,
            normalize=normalize,
        )
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
    else:
        if num_classes is None:
            max_source = int(source_labels.max().item()) if source_labels.numel() else 0
            max_target = int(target_pseudo_labels.max().item()) if target_pseudo_labels.numel() else 0
            num_classes = max(max_source, max_target) + 1
        contrastive_loss = FastClassPrototypeStageContrastiveLoss(
            temperature=temperature,
            normalize=normalize,
        )
        loss, contrast_logs = contrastive_loss(
            source_stage["stage_feats"],
            source_stage["stage_mask"],
            source_stage["stage_center_pos"],
            source_labels,
            target_stage["stage_feats"],
            target_stage["stage_mask"],
            target_stage["stage_center_pos"],
            target_pseudo_labels,
            target_conf,
            target_mask,
            target_to_source_shift,
            num_classes=int(num_classes),
            stage_time_radius=stage_time_radius,
            stage_time_temperature=stage_time_temperature,
        )
    logs = {}
    logs.update({f"source_{key}": value for key, value in source_stage["logs"].items()})
    logs.update({f"target_{key}": value for key, value in target_stage["logs"].items()})
    logs.update(contrast_logs)
    logs["stage_contrast_backend"] = 1.0 if str(backend) == "class_prototype_fast" else 0.0
    return loss, logs
