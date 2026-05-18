import math
from collections import defaultdict

import numpy as np
import torch
import zarr


UNIFORM_PHASE_COUNT = 5
SOURCE_PHASE_COMPACTNESS_LAMBDA = 0.05
SOURCE_STRUCTURE_LOSS_VERSION = "compactness"
SOURCE_STRUCTURE_INTRA_TRADE_OFF = 1.0
SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF = 0.25
SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF = 0.25
SOURCE_STRUCTURE_SHAPE_TRADE_OFF = 0.15
SOURCE_STRUCTURE_TREND_TRADE_OFF = 0.05
SOURCE_STRUCTURE_SEASON_TRADE_OFF = 0.02
SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF = 0.02
SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF = 0.20
SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE = 2
SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF = 0.35
SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF = 0.05
SHAPE_REG_DIRECTION_TRADE_OFF = 0.5
SHAPE_REG_COLLAPSE_TRADE_OFF = 0.5
SHAPE_REG_COLLAPSE_MARGIN = 0.35
SEASON_REG_REDUNDANCY_TRADE_OFF = 0.25
# Source-level dynamic phase-weight construction:
# - keep the phase partition fixed (uniform 5 phases)
# - accumulate source-domain phase structure statistics across batches
# - derive the applied weights from a source-level EMA summary instead of
#   re-deciding them from each batch independently
PHASE_WEIGHT_COMPACTNESS_COEFF = 1.0
PHASE_WEIGHT_MARGIN_COEFF = 1.0
PHASE_WEIGHT_TEMPERATURE = 1.0
PHASE_WEIGHT_EMA_MOMENTUM = 0.9
SOURCE_PHASE_PARTITION_MODE = "uniform"
SOURCE_PHASE_GAP_THRESHOLD = 45
SOURCE_PHASE_MIN_POINTS = 3
SOURCE_PHASE_MAX_POINTS = 8
SOURCE_PHASE_MAX_SPAN = 120
SOURCE_PHASE_MIN_SAMPLE_POINTS = 2
SOURCE_SEGMENT_PARTITION_MODE = SOURCE_PHASE_PARTITION_MODE
SOURCE_SEGMENT_SEMANTIC_QUANTILE = 0.75
SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS = 128
SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF = 0.5
SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF = 0.25
SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF = 0.25
SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE = 2
SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF = 0.5
SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS = SOURCE_PHASE_MIN_POINTS
SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK = 1
SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE = 1.15
SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF = 0.35


def _uniform_phase_slices(sequence_length, phase_count):
    phase_indices = torch.arange(sequence_length, dtype=torch.long)
    return torch.tensor_split(phase_indices, phase_count)


def _sorted_sequence_features(spatial_feats, positions):
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1])
    ordered_feats = torch.gather(spatial_feats, dim=1, index=expanded)
    ordered_positions = torch.gather(positions, dim=1, index=sort_indices)
    return ordered_feats, ordered_positions


def _moving_average_same(sequence_tensor):
    if sequence_tensor.ndim != 2:
        raise ValueError(f"Expected [T, D] tensor, got {tuple(sequence_tensor.shape)}")
    if sequence_tensor.shape[0] <= 1:
        return sequence_tensor
    if sequence_tensor.shape[0] == 2:
        return 0.5 * (sequence_tensor + torch.roll(sequence_tensor, shifts=-1, dims=0))

    padded = torch.cat(
        [sequence_tensor[:1], sequence_tensor, sequence_tensor[-1:]],
        dim=0,
    )
    return (
        padded[:-2]
        + padded[1:-1]
        + padded[2:]
    ) / 3.0


def _compute_trajectory_prototype_losses(ordered_feats, labels, eps=1e-6):
    zero = ordered_feats.sum() * 0.0
    trajectory_intra_loss = zero
    prototype_dynamics_loss = zero
    trajectory_class_count = 0
    dynamics_class_count = 0

    if ordered_feats.ndim != 3 or ordered_feats.shape[1] < 2:
        return trajectory_intra_loss, prototype_dynamics_loss, trajectory_class_count, dynamics_class_count

    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        if int(class_mask.sum().item()) < 2:
            continue

        class_curves = ordered_feats[class_mask]
        class_prototype = class_curves.mean(dim=0, keepdim=True)
        class_trajectory_intra = (class_curves - class_prototype).pow(2).sum(dim=2).mean()
        trajectory_intra_loss = trajectory_intra_loss + class_trajectory_intra
        trajectory_class_count += 1

        class_dynamics = class_curves[:, 1:] - class_curves[:, :-1]
        prototype_dynamics = class_prototype[:, 1:] - class_prototype[:, :-1]
        class_dynamics_loss = (class_dynamics - prototype_dynamics).pow(2).sum(dim=2).mean()
        prototype_dynamics_loss = prototype_dynamics_loss + class_dynamics_loss
        dynamics_class_count += 1

    if trajectory_class_count > 0:
        trajectory_intra_loss = trajectory_intra_loss / (trajectory_class_count + eps)
    if dynamics_class_count > 0:
        prototype_dynamics_loss = prototype_dynamics_loss / (dynamics_class_count + eps)
    return trajectory_intra_loss, prototype_dynamics_loss, trajectory_class_count, dynamics_class_count


def _standardize_temporal_curve(curve, eps=1e-6):
    mean = curve.mean(axis=0, keepdims=True)
    std = curve.std(axis=0, keepdims=True)
    return (curve - mean) / np.clip(std, eps, None)


def _safe_cosine_dissimilarity(left, right, eps=1e-6):
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.clip(left_norm * right_norm, eps, None)
    cosine = np.sum(left * right, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return 1.0 - cosine


def _compute_dataset_semantic_boundary_scores(
    dataset,
    max_samples_per_class=SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS,
    curvature_trade_off=SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF,
    energy_trade_off=SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF,
    similarity_trade_off=SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF,
    eps=1e-6,
):
    if dataset is None or not hasattr(dataset, "samples") or not hasattr(dataset, "date_positions"):
        return None
    sequence_length = len(dataset.date_positions)
    if sequence_length < 2:
        return None

    class_sums = {}
    class_counts = defaultdict(int)
    max_samples_per_class = max(1, int(max_samples_per_class))
    curvature_trade_off = float(curvature_trade_off)
    energy_trade_off = float(energy_trade_off)
    similarity_trade_off = float(similarity_trade_off)

    for path, _parcel_idx, label, _extra in dataset.samples:
        if class_counts[int(label)] >= max_samples_per_class:
            continue
        pixels = zarr.load(path)
        if pixels.ndim != 3 or pixels.shape[0] != sequence_length:
            continue
        temporal_curve = pixels.mean(axis=-1).astype(np.float64, copy=False)
        label = int(label)
        if label not in class_sums:
            class_sums[label] = temporal_curve.copy()
        else:
            class_sums[label] += temporal_curve
        class_counts[label] += 1

    if not class_sums:
        return None

    class_scores = []
    for label, summed_curve in class_sums.items():
        count = max(class_counts[label], 1)
        mean_curve = summed_curve / float(count)
        standardized_curve = _standardize_temporal_curve(mean_curve, eps=eps)
        slope = np.linalg.norm(standardized_curve[1:] - standardized_curve[:-1], axis=1)
        score = slope.copy()

        energy = np.linalg.norm(mean_curve, axis=1)
        energy_change = np.abs(energy[1:] - energy[:-1])
        score = score + energy_trade_off * energy_change

        similarity_drop = _safe_cosine_dissimilarity(
            standardized_curve[:-1],
            standardized_curve[1:],
            eps=eps,
        )
        score = score + similarity_trade_off * similarity_drop

        if standardized_curve.shape[0] >= 3:
            curvature = np.linalg.norm(
                standardized_curve[2:] - 2.0 * standardized_curve[1:-1] + standardized_curve[:-2],
                axis=1,
            )
            boundary_curvature = np.zeros_like(slope)
            boundary_curvature_counts = np.zeros_like(slope)
            boundary_curvature[:-1] += curvature
            boundary_curvature[1:] += curvature
            boundary_curvature_counts[:-1] += 1.0
            boundary_curvature_counts[1:] += 1.0
            boundary_curvature = boundary_curvature / np.clip(boundary_curvature_counts, 1.0, None)
            score = score + curvature_trade_off * boundary_curvature

        class_scores.append(score)

    if not class_scores:
        return None
    return np.mean(np.stack(class_scores, axis=0), axis=0)


def _compute_dataset_semantic_time_embeddings(
    dataset,
    max_samples_per_class=SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS,
    eps=1e-6,
):
    if dataset is None or not hasattr(dataset, "samples") or not hasattr(dataset, "date_positions"):
        return None
    sequence_length = len(dataset.date_positions)
    if sequence_length < 1:
        return None

    class_sums = {}
    class_counts = defaultdict(int)
    max_samples_per_class = max(1, int(max_samples_per_class))

    for path, _parcel_idx, label, _extra in dataset.samples:
        if class_counts[int(label)] >= max_samples_per_class:
            continue
        pixels = zarr.load(path)
        if pixels.ndim != 3 or pixels.shape[0] != sequence_length:
            continue
        temporal_curve = pixels.mean(axis=-1).astype(np.float64, copy=False)
        label = int(label)
        if label not in class_sums:
            class_sums[label] = temporal_curve.copy()
        else:
            class_sums[label] += temporal_curve
        class_counts[label] += 1

    if not class_sums:
        return None

    embeddings = []
    for label, summed_curve in sorted(class_sums.items(), key=lambda item: item[0]):
        count = max(class_counts[label], 1)
        mean_curve = summed_curve / float(count)
        standardized_curve = _standardize_temporal_curve(mean_curve, eps=eps)
        energy = np.linalg.norm(mean_curve, axis=1, keepdims=True)
        energy = _standardize_temporal_curve(energy, eps=eps)
        embeddings.append(np.concatenate([standardized_curve, energy], axis=1))

    if not embeddings:
        return None
    return np.concatenate(embeddings, axis=1)


def _select_semantic_cut_indices(boundary_scores, quantile, base_segments, sorted_positions, min_points, max_extra_cuts_per_base):
    if boundary_scores is None:
        return []

    boundary_scores = np.asarray(boundary_scores, dtype=np.float64)
    if boundary_scores.ndim != 1 or boundary_scores.size == 0:
        return []
    finite_scores = boundary_scores[np.isfinite(boundary_scores)]
    if finite_scores.size == 0:
        return []

    quantile = float(np.clip(quantile, 0.0, 1.0))
    threshold = float(np.quantile(finite_scores, quantile))
    sorted_positions = np.asarray(sorted_positions, dtype=np.int64)
    min_points = max(1, int(min_points))
    max_extra_cuts_per_base = max(0, int(max_extra_cuts_per_base))
    cut_indices = []
    for segment in base_segments:
        segment = np.asarray(segment, dtype=np.int64)
        if len(segment) < (2 * min_points + 1) or max_extra_cuts_per_base <= 0:
            continue
        start_idx = int(np.searchsorted(sorted_positions, segment[0]))
        end_idx = int(np.searchsorted(sorted_positions, segment[-1]))
        candidates = []
        for cut_idx in range(start_idx + min_points, end_idx - min_points + 2):
            boundary_idx = cut_idx - 1
            if boundary_idx < 0 or boundary_idx >= boundary_scores.size:
                continue
            score = boundary_scores[boundary_idx]
            if not np.isfinite(score) or score <= 0.0 or score < threshold:
                continue
            left = boundary_scores[boundary_idx - 1] if boundary_idx > 0 else -np.inf
            right = boundary_scores[boundary_idx + 1] if boundary_idx < boundary_scores.size - 1 else -np.inf
            if score >= left and score >= right:
                candidates.append((float(score), int(cut_idx)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = sorted(cut_idx for _score, cut_idx in candidates[:max_extra_cuts_per_base])
        cut_indices.extend(selected)
    return cut_indices


def _segments_from_cut_indices(sorted_positions, cut_indices):
    cut_indices = sorted({int(idx) for idx in cut_indices if 0 < int(idx) < len(sorted_positions)})
    segments = []
    start_idx = 0
    for cut_idx in cut_indices:
        segments.append(sorted_positions[start_idx:cut_idx])
        start_idx = cut_idx
    segments.append(sorted_positions[start_idx:])
    return [segment for segment in segments if len(segment) > 0]


def _allocate_block_segment_targets(block_lengths, block_caps, target_count):
    num_blocks = len(block_lengths)
    if num_blocks == 0:
        return []
    target_count = int(target_count)
    target_count = max(num_blocks, min(target_count, int(sum(block_caps))))
    allocations = np.ones(num_blocks, dtype=np.int64)
    remaining = target_count - num_blocks
    if remaining <= 0:
        return allocations.tolist()

    capacities = np.maximum(np.asarray(block_caps, dtype=np.int64) - 1, 0)
    weights = np.asarray(block_lengths, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    while remaining > 0 and capacities.sum() > 0:
        scores = np.where(capacities > 0, weights / np.clip(allocations, 1, None), -np.inf)
        chosen = int(np.argmax(scores))
        if not np.isfinite(scores[chosen]):
            break
        allocations[chosen] += 1
        capacities[chosen] -= 1
        remaining -= 1
    return allocations.tolist()


def _build_semantic_agglomerative_segments(
    sorted_positions,
    boundary_scores,
    time_embeddings,
    mandatory_cut_indices,
    target_count,
    min_points,
    max_points,
    max_span,
    merge_boundary_trade_off,
    aggl_min_points,
    aggl_target_slack,
    aggl_merge_cost_tolerance,
    aggl_dynamics_trade_off,
):
    sorted_positions = np.asarray(sorted_positions, dtype=np.int64)
    time_embeddings = None if time_embeddings is None else np.asarray(time_embeddings, dtype=np.float64)
    boundary_scores = None if boundary_scores is None else np.asarray(boundary_scores, dtype=np.float64)
    min_points = max(1, int(min_points))
    max_points = max(1, int(max_points))
    max_span = max(1, int(max_span))
    merge_boundary_trade_off = float(merge_boundary_trade_off)
    aggl_min_points = max(1, int(aggl_min_points))
    aggl_target_slack = max(0, int(aggl_target_slack))
    aggl_merge_cost_tolerance = max(0.0, float(aggl_merge_cost_tolerance))
    aggl_dynamics_trade_off = max(0.0, float(aggl_dynamics_trade_off))

    base_segments = _segments_from_cut_indices(sorted_positions, mandatory_cut_indices)
    if time_embeddings is None or time_embeddings.shape[0] != sorted_positions.size:
        return _merge_small_segments(base_segments, min_points)

    block_lengths = [len(segment) for segment in base_segments]
    block_caps = [
        max(1, min(len(segment), len(segment) // aggl_min_points if aggl_min_points > 1 else len(segment)))
        for segment in base_segments
    ]
    allocations = _allocate_block_segment_targets(block_lengths, block_caps, target_count)

    def segment_repr(start_idx, end_idx):
        return time_embeddings[start_idx:end_idx].mean(axis=0)

    def segment_dynamics(start_idx, end_idx):
        segment_embed = time_embeddings[start_idx:end_idx]
        if segment_embed.shape[0] <= 1:
            return np.zeros(segment_embed.shape[1], dtype=np.float64)
        diffs = segment_embed[1:] - segment_embed[:-1]
        return diffs.mean(axis=0)

    def merge_cost(left, right):
        _, left_end = left
        boundary_idx = left_end - 1
        boundary_penalty = 0.0
        if boundary_scores is not None and 0 <= boundary_idx < boundary_scores.size and np.isfinite(boundary_scores[boundary_idx]):
            boundary_penalty = float(boundary_scores[boundary_idx])
        semantic_distance = float(np.linalg.norm(segment_repr(*left) - segment_repr(*right)))
        dynamics_distance = float(np.linalg.norm(segment_dynamics(*left) - segment_dynamics(*right)))
        return (
            semantic_distance
            + aggl_dynamics_trade_off * dynamics_distance
            + merge_boundary_trade_off * boundary_penalty
        )

    refined_segments = []
    base_start_idx = 0
    for segment, block_target in zip(base_segments, allocations):
        block_size = len(segment)
        block_target = max(1, min(int(block_target), block_size))
        min_segment_count = max(1, min(block_size, block_target - aggl_target_slack))
        max_segment_count = max(min_segment_count, min(block_size, block_target + aggl_target_slack))
        block_segments = [(base_start_idx + idx, base_start_idx + idx + 1) for idx in range(block_size)]
        initial_pair_costs = [
            merge_cost(block_segments[idx], block_segments[idx + 1])
            for idx in range(len(block_segments) - 1)
        ]
        merge_cost_threshold = (
            float(np.median(initial_pair_costs)) * aggl_merge_cost_tolerance
            if initial_pair_costs
            else np.inf
        )

        while len(block_segments) > max_segment_count:
            best_pair_idx = None
            best_cost = np.inf
            for idx in range(len(block_segments) - 1):
                left = block_segments[idx]
                right = block_segments[idx + 1]
                merged_points = right[1] - left[0]
                merged_span = int(sorted_positions[right[1] - 1] - sorted_positions[left[0]])
                if merged_points > max_points or merged_span > max_span:
                    continue
                cost = merge_cost(left, right)
                if cost < best_cost:
                    best_cost = cost
                    best_pair_idx = idx

            if best_pair_idx is None:
                for idx in range(len(block_segments) - 1):
                    cost = merge_cost(block_segments[idx], block_segments[idx + 1])
                    if cost < best_cost:
                        best_cost = cost
                        best_pair_idx = idx
            if best_pair_idx is None:
                break

            merged = (block_segments[best_pair_idx][0], block_segments[best_pair_idx + 1][1])
            block_segments = block_segments[:best_pair_idx] + [merged] + block_segments[best_pair_idx + 2:]

        while len(block_segments) > min_segment_count:
            best_pair_idx = None
            best_cost = np.inf
            for idx in range(len(block_segments) - 1):
                left = block_segments[idx]
                right = block_segments[idx + 1]
                merged_points = right[1] - left[0]
                merged_span = int(sorted_positions[right[1] - 1] - sorted_positions[left[0]])
                if merged_points > max_points or merged_span > max_span:
                    continue
                cost = merge_cost(left, right)
                if cost < best_cost:
                    best_cost = cost
                    best_pair_idx = idx
            if best_pair_idx is None or not np.isfinite(best_cost) or best_cost > merge_cost_threshold:
                break
            merged = (block_segments[best_pair_idx][0], block_segments[best_pair_idx + 1][1])
            block_segments = block_segments[:best_pair_idx] + [merged] + block_segments[best_pair_idx + 2:]

        merged_block = [sorted_positions[start:end] for start, end in block_segments]
        merged_block = _merge_small_segments(merged_block, min_points)
        refined_segments.extend(merged_block)
        base_start_idx += block_size

    return refined_segments


def _split_large_segments_by_semantics(
    segments,
    sorted_positions,
    boundary_scores,
    max_points,
    max_span,
    min_points,
):
    sorted_positions = np.asarray(sorted_positions, dtype=np.int64)
    boundary_scores = None if boundary_scores is None else np.asarray(boundary_scores, dtype=np.float64)
    max_points = max(1, int(max_points))
    max_span = max(1, int(max_span))
    min_points = max(1, int(min_points))

    def split_segment(segment):
        segment = np.asarray(segment, dtype=np.int64)
        if len(segment) == 0:
            return []
        span = int(segment[-1] - segment[0]) if len(segment) > 0 else 0
        if len(segment) <= max_points and span <= max_span:
            return [segment]

        start_idx = int(np.searchsorted(sorted_positions, segment[0]))
        end_idx = int(np.searchsorted(sorted_positions, segment[-1]))
        valid_cut_indices = []
        for cut_idx in range(start_idx + min_points, end_idx - min_points + 2):
            if cut_idx <= start_idx or cut_idx > end_idx:
                continue
            valid_cut_indices.append(cut_idx)

        if not valid_cut_indices:
            midpoint = start_idx + max(1, len(segment) // 2)
            valid_cut_indices = [min(max(midpoint, start_idx + 1), end_idx)]

        best_cut_idx = None
        best_score = -np.inf
        if boundary_scores is not None:
            for cut_idx in valid_cut_indices:
                score_idx = cut_idx - 1
                score = boundary_scores[score_idx] if 0 <= score_idx < boundary_scores.size else -np.inf
                if score > best_score:
                    best_score = score
                    best_cut_idx = cut_idx

        if best_cut_idx is None:
            best_cut_idx = valid_cut_indices[len(valid_cut_indices) // 2]

        left = sorted_positions[start_idx:best_cut_idx]
        right = sorted_positions[best_cut_idx:end_idx + 1]
        return split_segment(left) + split_segment(right)

    refined_segments = []
    for segment in segments:
        refined_segments.extend(split_segment(segment))
    return refined_segments


def _merge_small_segments(segments, min_points):
    segments = [np.asarray(segment, dtype=np.int64) for segment in segments if len(segment) > 0]
    if not segments:
        return segments

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for idx, segment in enumerate(list(segments)):
            if len(segment) >= min_points:
                continue
            if idx == 0:
                segments[1] = np.concatenate([segment, segments[1]])
                del segments[0]
            elif idx == len(segments) - 1:
                segments[idx - 1] = np.concatenate([segments[idx - 1], segment])
                del segments[idx]
            else:
                gap_prev = int(segment[0] - segments[idx - 1][-1])
                gap_next = int(segments[idx + 1][0] - segment[-1])
                if gap_prev <= gap_next:
                    segments[idx - 1] = np.concatenate([segments[idx - 1], segment])
                    del segments[idx]
                else:
                    segments[idx + 1] = np.concatenate([segment, segments[idx + 1]])
                    del segments[idx]
            changed = True
            break
    return segments


def build_source_phase_partition_spec(
    date_positions,
    mode=SOURCE_PHASE_PARTITION_MODE,
    phase_count=UNIFORM_PHASE_COUNT,
    gap_threshold=SOURCE_PHASE_GAP_THRESHOLD,
    min_points=SOURCE_PHASE_MIN_POINTS,
    max_points=SOURCE_PHASE_MAX_POINTS,
    max_span=SOURCE_PHASE_MAX_SPAN,
    dataset=None,
    semantic_quantile=SOURCE_SEGMENT_SEMANTIC_QUANTILE,
    semantic_max_samples_per_class=SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS,
    semantic_curvature_trade_off=SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF,
    semantic_energy_trade_off=SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF,
    semantic_similarity_trade_off=SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF,
    semantic_max_extra_cuts_per_base=SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE,
    semantic_merge_boundary_trade_off=SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF,
    semantic_aggl_min_points=SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS,
    semantic_aggl_target_slack=SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK,
    semantic_aggl_merge_cost_tolerance=SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE,
    semantic_aggl_dynamics_trade_off=SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF,
):
    mode = str(mode).lower()
    sorted_positions = np.asarray(sorted({int(pos) for pos in date_positions}), dtype=np.int64)
    if sorted_positions.size == 0:
        raise ValueError("Cannot build source phase partition without date positions.")

    if mode == "uniform":
        return {
            "mode": "uniform",
            "phase_count": int(max(1, min(int(phase_count), int(sorted_positions.size)))),
            "intervals": None,
            "date_positions": sorted_positions.tolist(),
        }

    if mode not in {"doy_gap", "semantic_doy_gap", "semantic_doy", "semantic_gap", "semantic_agglomerative"}:
        raise ValueError(f"Unsupported source phase partition mode: {mode}")

    mandatory_cut_indices = []
    for idx in range(1, sorted_positions.size):
        if int(sorted_positions[idx] - sorted_positions[idx - 1]) > int(gap_threshold):
            mandatory_cut_indices.append(idx)

    semantic_boundary_scores = None
    semantic_cut_indices = []
    base_segments = _segments_from_cut_indices(sorted_positions, mandatory_cut_indices)
    if mode in {"semantic_doy_gap", "semantic_doy", "semantic_gap", "semantic_agglomerative"}:
        semantic_boundary_scores = _compute_dataset_semantic_boundary_scores(
            dataset,
            max_samples_per_class=semantic_max_samples_per_class,
            curvature_trade_off=semantic_curvature_trade_off,
            energy_trade_off=semantic_energy_trade_off,
            similarity_trade_off=semantic_similarity_trade_off,
        )
        if mode in {"semantic_doy_gap", "semantic_doy", "semantic_gap"}:
            semantic_cut_indices = _select_semantic_cut_indices(
                semantic_boundary_scores,
                quantile=semantic_quantile,
                base_segments=base_segments,
                sorted_positions=sorted_positions,
                min_points=min_points,
                max_extra_cuts_per_base=semantic_max_extra_cuts_per_base,
            )

    if mode == "semantic_agglomerative":
        semantic_time_embeddings = _compute_dataset_semantic_time_embeddings(
            dataset,
            max_samples_per_class=semantic_max_samples_per_class,
        )
        merged_segments = _build_semantic_agglomerative_segments(
            sorted_positions=sorted_positions,
            boundary_scores=semantic_boundary_scores,
            time_embeddings=semantic_time_embeddings,
            mandatory_cut_indices=mandatory_cut_indices,
            target_count=int(max(1, min(int(phase_count), int(sorted_positions.size)))),
            min_points=min_points,
            max_points=max_points,
            max_span=max_span,
            merge_boundary_trade_off=semantic_merge_boundary_trade_off,
            aggl_min_points=semantic_aggl_min_points,
            aggl_target_slack=semantic_aggl_target_slack,
            aggl_merge_cost_tolerance=semantic_aggl_merge_cost_tolerance,
            aggl_dynamics_trade_off=semantic_aggl_dynamics_trade_off,
        )
        intervals = [(int(segment[0]), int(segment[-1])) for segment in merged_segments]
        spec = {
            "mode": mode,
            "phase_count": len(intervals),
            "segment_count": len(intervals),
            "intervals": intervals,
            "date_positions": sorted_positions.tolist(),
            "gap_threshold": int(gap_threshold),
            "min_points": int(min_points),
            "max_points": int(max_points),
            "max_span": int(max_span),
            "semantic_mode": True,
            "semantic_quantile": float(semantic_quantile),
            "semantic_max_samples_per_class": int(semantic_max_samples_per_class),
            "semantic_curvature_trade_off": float(semantic_curvature_trade_off),
            "semantic_energy_trade_off": float(semantic_energy_trade_off),
            "semantic_similarity_trade_off": float(semantic_similarity_trade_off),
            "semantic_max_extra_cuts_per_base": int(semantic_max_extra_cuts_per_base),
            "semantic_merge_boundary_trade_off": float(semantic_merge_boundary_trade_off),
            "semantic_aggl_min_points": int(semantic_aggl_min_points),
            "semantic_aggl_target_slack": int(semantic_aggl_target_slack),
            "semantic_aggl_merge_cost_tolerance": float(semantic_aggl_merge_cost_tolerance),
            "semantic_aggl_dynamics_trade_off": float(semantic_aggl_dynamics_trade_off),
            "semantic_target_count": int(max(1, min(int(phase_count), int(sorted_positions.size)))),
        }
        if semantic_boundary_scores is not None:
            spec["semantic_boundary_scores"] = [float(x) for x in semantic_boundary_scores.tolist()]
        return spec

    base_segments = _segments_from_cut_indices(
        sorted_positions,
        mandatory_cut_indices + semantic_cut_indices,
    )
    split_segments = _split_large_segments_by_semantics(
        base_segments,
        sorted_positions=sorted_positions,
        boundary_scores=semantic_boundary_scores,
        max_points=max_points,
        max_span=max_span,
        min_points=min_points,
    )
    merged_segments = _merge_small_segments(split_segments, max(1, int(min_points)))
    intervals = [(int(segment[0]), int(segment[-1])) for segment in merged_segments]
    spec = {
        "mode": mode,
        "phase_count": len(intervals),
        "segment_count": len(intervals),
        "intervals": intervals,
        "date_positions": sorted_positions.tolist(),
        "gap_threshold": int(gap_threshold),
        "min_points": int(min_points),
        "max_points": int(max_points),
        "max_span": int(max_span),
        "semantic_mode": mode in {"semantic_doy_gap", "semantic_doy", "semantic_gap", "semantic_agglomerative"},
        "semantic_quantile": float(semantic_quantile),
        "semantic_max_samples_per_class": int(semantic_max_samples_per_class),
        "semantic_curvature_trade_off": float(semantic_curvature_trade_off),
        "semantic_energy_trade_off": float(semantic_energy_trade_off),
        "semantic_similarity_trade_off": float(semantic_similarity_trade_off),
        "semantic_max_extra_cuts_per_base": int(semantic_max_extra_cuts_per_base),
        "semantic_merge_boundary_trade_off": float(semantic_merge_boundary_trade_off),
        "semantic_aggl_min_points": int(semantic_aggl_min_points),
        "semantic_aggl_target_slack": int(semantic_aggl_target_slack),
        "semantic_aggl_merge_cost_tolerance": float(semantic_aggl_merge_cost_tolerance),
        "semantic_aggl_dynamics_trade_off": float(semantic_aggl_dynamics_trade_off),
    }
    if semantic_boundary_scores is not None:
        spec["semantic_boundary_scores"] = [float(x) for x in semantic_boundary_scores.tolist()]
    return spec


def describe_source_phase_partition_spec(spec):
    if spec["mode"] == "uniform":
        return f"mode=uniform, phase_count={spec['phase_count']}"
    interval_text = ", ".join([f"[{start},{end}]" for start, end in spec["intervals"]])
    semantic_suffix = ""
    if spec.get("semantic_mode", False):
        semantic_suffix = (
            f", semantic_quantile={spec['semantic_quantile']:.2f}, "
            f"semantic_max_samples_per_class={spec['semantic_max_samples_per_class']}, "
            f"semantic_curvature_trade_off={spec['semantic_curvature_trade_off']:.2f}, "
            f"semantic_energy_trade_off={spec['semantic_energy_trade_off']:.2f}, "
            f"semantic_similarity_trade_off={spec['semantic_similarity_trade_off']:.2f}, "
            f"semantic_max_extra_cuts_per_base={spec['semantic_max_extra_cuts_per_base']}, "
            f"semantic_merge_boundary_trade_off={spec['semantic_merge_boundary_trade_off']:.2f}"
        )
        if "semantic_target_count" in spec:
            semantic_suffix += f", semantic_target_count={spec['semantic_target_count']}"
        if spec["mode"] == "semantic_agglomerative":
            semantic_suffix += (
                f", semantic_aggl_min_points={spec['semantic_aggl_min_points']}, "
                f"semantic_aggl_target_slack={spec['semantic_aggl_target_slack']}, "
                f"semantic_aggl_merge_cost_tolerance={spec['semantic_aggl_merge_cost_tolerance']:.2f}, "
                f"semantic_aggl_dynamics_trade_off={spec['semantic_aggl_dynamics_trade_off']:.2f}"
            )
    return (
        f"mode={spec['mode']}, phase_count={spec['phase_count']}, "
        f"gap_threshold={spec['gap_threshold']}, min_points={spec['min_points']}, "
        f"max_points={spec['max_points']}, max_span={spec['max_span']}{semantic_suffix}, intervals={interval_text}"
    )


def build_source_segment_partition_spec(
    date_positions,
    mode=SOURCE_SEGMENT_PARTITION_MODE,
    segment_count=UNIFORM_PHASE_COUNT,
    gap_threshold=SOURCE_PHASE_GAP_THRESHOLD,
    min_points=SOURCE_PHASE_MIN_POINTS,
    max_points=SOURCE_PHASE_MAX_POINTS,
    max_span=SOURCE_PHASE_MAX_SPAN,
    dataset=None,
    semantic_quantile=SOURCE_SEGMENT_SEMANTIC_QUANTILE,
    semantic_max_samples_per_class=SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS,
    semantic_curvature_trade_off=SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF,
    semantic_energy_trade_off=SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF,
    semantic_similarity_trade_off=SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF,
    semantic_max_extra_cuts_per_base=SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE,
    semantic_merge_boundary_trade_off=SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF,
    semantic_aggl_min_points=SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS,
    semantic_aggl_target_slack=SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK,
    semantic_aggl_merge_cost_tolerance=SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE,
    semantic_aggl_dynamics_trade_off=SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF,
):
    spec = build_source_phase_partition_spec(
        date_positions=date_positions,
        mode=mode,
        phase_count=segment_count,
        gap_threshold=gap_threshold,
        min_points=min_points,
        max_points=max_points,
        max_span=max_span,
        dataset=dataset,
        semantic_quantile=semantic_quantile,
        semantic_max_samples_per_class=semantic_max_samples_per_class,
        semantic_curvature_trade_off=semantic_curvature_trade_off,
        semantic_energy_trade_off=semantic_energy_trade_off,
        semantic_similarity_trade_off=semantic_similarity_trade_off,
        semantic_max_extra_cuts_per_base=semantic_max_extra_cuts_per_base,
        semantic_merge_boundary_trade_off=semantic_merge_boundary_trade_off,
        semantic_aggl_min_points=semantic_aggl_min_points,
        semantic_aggl_target_slack=semantic_aggl_target_slack,
        semantic_aggl_merge_cost_tolerance=semantic_aggl_merge_cost_tolerance,
        semantic_aggl_dynamics_trade_off=semantic_aggl_dynamics_trade_off,
    )
    spec["segment_count"] = int(spec["phase_count"])
    return spec


def describe_source_segment_partition_spec(spec):
    text = describe_source_phase_partition_spec(spec)
    return text.replace("phase_count", "segment_count")


def _phase_masks_from_spec(ordered_positions, phase_partition_spec):
    batch_size, sequence_length = ordered_positions.shape
    if phase_partition_spec is None or phase_partition_spec["mode"] == "uniform":
        phase_count = UNIFORM_PHASE_COUNT if phase_partition_spec is None else int(phase_partition_spec["phase_count"])
        phase_slices = _uniform_phase_slices(sequence_length, phase_count)
        phase_masks = []
        for phase_indices in phase_slices:
            mask = torch.zeros((batch_size, sequence_length), device=ordered_positions.device, dtype=torch.bool)
            mask[:, phase_indices] = True
            phase_masks.append(mask)
        return phase_masks

    phase_masks = []
    for start, end in phase_partition_spec["intervals"]:
        phase_masks.append((ordered_positions >= start) & (ordered_positions <= end))
    return phase_masks


def _segment_masks_from_spec(ordered_positions, segment_partition_spec):
    return _phase_masks_from_spec(ordered_positions, segment_partition_spec)


def _zscore_or_zero(values, eps):
    if values.numel() <= 1:
        return values.new_zeros(values.shape)
    centered = values - values.mean()
    std = centered.std(unbiased=False)
    if float(std.item()) < eps:
        return values.new_zeros(values.shape)
    return centered / (std + eps)


def _compute_phase_weights(phase_structures, reference_tensor, eps):
    compactness_scores = []
    margin_scores = []
    valid_mask = []

    for stats in phase_structures:
        valid = stats["valid_class_count"] > 0
        valid_mask.append(valid)
        if valid:
            compactness_scores.append(stats["compactness_score"])
            margin_scores.append(stats["margin_score"])
        else:
            compactness_scores.append(reference_tensor.new_tensor(0.0))
            margin_scores.append(reference_tensor.new_tensor(0.0))

    compactness_scores = torch.stack(compactness_scores)
    margin_scores = torch.stack(margin_scores)
    valid_mask_tensor = torch.tensor(valid_mask, device=reference_tensor.device, dtype=torch.bool)

    if not bool(valid_mask_tensor.any()):
        return reference_tensor.new_full((len(phase_structures),), 1.0 / max(len(phase_structures), 1))

    compactness_z = _zscore_or_zero(compactness_scores[valid_mask_tensor], eps)
    margin_z = _zscore_or_zero(margin_scores[valid_mask_tensor], eps)
    combined = (
        PHASE_WEIGHT_COMPACTNESS_COEFF * compactness_z
        + PHASE_WEIGHT_MARGIN_COEFF * margin_z
    )
    valid_weights = torch.softmax(combined / PHASE_WEIGHT_TEMPERATURE, dim=0)

    weights = reference_tensor.new_zeros(len(phase_structures))
    weights[valid_mask_tensor] = valid_weights
    return weights


class SourcePhaseWeightTracker:
    def __init__(
        self,
        phase_count,
        ema_momentum=PHASE_WEIGHT_EMA_MOMENTUM,
        phase_partition_spec=None,
        min_sample_points_per_phase=SOURCE_PHASE_MIN_SAMPLE_POINTS,
    ):
        self.phase_count = phase_count
        self.ema_momentum = ema_momentum
        self.phase_partition_spec = phase_partition_spec
        self.min_sample_points_per_phase = int(min_sample_points_per_phase)
        self.running_compactness = None
        self.running_margin = None
        self.valid_counts = None

    def update(self, phase_structures):
        compactness_scores = []
        margin_scores = []
        valid_mask = []

        for stats in phase_structures:
            valid = stats["valid_class_count"] > 0
            valid_mask.append(valid)
            compactness_scores.append(stats["compactness_score"].detach().cpu())
            margin_scores.append(stats["margin_score"].detach().cpu())

        compactness_scores = torch.stack(compactness_scores)
        margin_scores = torch.stack(margin_scores)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        if self.running_compactness is None:
            self.running_compactness = compactness_scores.clone()
            self.running_margin = margin_scores.clone()
            self.valid_counts = valid_mask_tensor.to(torch.long)
            return

        for phase_idx in range(self.phase_count):
            if not bool(valid_mask_tensor[phase_idx].item()):
                continue

            if int(self.valid_counts[phase_idx].item()) == 0:
                self.running_compactness[phase_idx] = compactness_scores[phase_idx]
                self.running_margin[phase_idx] = margin_scores[phase_idx]
            else:
                momentum = self.ema_momentum
                self.running_compactness[phase_idx] = (
                    momentum * self.running_compactness[phase_idx]
                    + (1.0 - momentum) * compactness_scores[phase_idx]
                )
                self.running_margin[phase_idx] = (
                    momentum * self.running_margin[phase_idx]
                    + (1.0 - momentum) * margin_scores[phase_idx]
                )

            self.valid_counts[phase_idx] += 1

    def get_weights(self, reference_tensor, eps):
        if self.running_compactness is None or self.valid_counts is None:
            return reference_tensor.new_full((self.phase_count,), 1.0 / max(self.phase_count, 1))

        valid_mask = self.valid_counts > 0
        if not bool(valid_mask.any().item()):
            return reference_tensor.new_full((self.phase_count,), 1.0 / max(self.phase_count, 1))

        compactness_scores = self.running_compactness.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        margin_scores = self.running_margin.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        valid_mask_device = valid_mask.to(device=reference_tensor.device)

        compactness_z = _zscore_or_zero(compactness_scores[valid_mask_device], eps)
        margin_z = _zscore_or_zero(margin_scores[valid_mask_device], eps)
        combined = (
            PHASE_WEIGHT_COMPACTNESS_COEFF * compactness_z
            + PHASE_WEIGHT_MARGIN_COEFF * margin_z
        )
        valid_weights = torch.softmax(combined / PHASE_WEIGHT_TEMPERATURE, dim=0)

        weights = reference_tensor.new_zeros(self.phase_count)
        weights[valid_mask_device] = valid_weights
        return weights

    def get_logs(self):
        logs = {}
        if self.running_compactness is None or self.running_margin is None or self.valid_counts is None:
            return logs

        for phase_idx in range(self.phase_count):
            logs[f"source_compactness_score_p{phase_idx + 1}"] = float(self.running_compactness[phase_idx].item())
            logs[f"source_margin_score_p{phase_idx + 1}"] = float(self.running_margin[phase_idx].item())
            logs[f"source_valid_count_p{phase_idx + 1}"] = float(self.valid_counts[phase_idx].item())
        if self.phase_partition_spec is not None:
            logs["source_phase_count"] = float(self.phase_partition_spec["phase_count"])
            logs["source_segment_count"] = float(self.phase_partition_spec["phase_count"])
        return logs


class SourceSegmentWeightTracker(SourcePhaseWeightTracker):
    pass


def compute_source_phase_compactness_loss(spatial_feats, positions, labels, weight_tracker=None, eps=1e-6):
    """
    Compute a source-domain phase compactness regularizer on top of per-time-step
    PSE features.

    - uniform 5 phases
    - phase weights come from a source-level running summary
    - compactness remains the optimized quantity
    - source-level weights are built from detached reliability estimates:
      - inverse within-class scatter
      - nearest-class phase margin
    """
    if spatial_feats.ndim != 3:
        raise ValueError(f"Expected spatial_feats to have shape [B, T, D], got {tuple(spatial_feats.shape)}")

    batch_size, sequence_length, _ = spatial_feats.shape
    if batch_size < 2 or sequence_length < 2:
        zero = spatial_feats.sum() * 0.0
        return zero, {"compactness_loss": 0.0}

    phase_partition_spec = getattr(weight_tracker, "phase_partition_spec", None) if weight_tracker is not None else None
    min_sample_points = (
        getattr(weight_tracker, "min_sample_points_per_phase", SOURCE_PHASE_MIN_SAMPLE_POINTS)
        if weight_tracker is not None
        else SOURCE_PHASE_MIN_SAMPLE_POINTS
    )
    ordered_feats, ordered_positions = _sorted_sequence_features(spatial_feats, positions)
    phase_masks = _phase_masks_from_spec(ordered_positions, phase_partition_spec)
    total_loss = spatial_feats.sum() * 0.0
    phase_logs = {}
    phase_structures = []

    for phase_idx, phase_mask in enumerate(phase_masks):
        phase_counts = phase_mask.sum(dim=1)
        valid_sample_mask = phase_counts >= int(min_sample_points)
        if not bool(valid_sample_mask.any().item()):
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_valid_samples_p{phase_idx + 1}"] = 0.0
            phase_structures.append({
                "phase_loss": total_loss,
                "valid_class_count": 0,
                "compactness_score": spatial_feats.new_tensor(0.0),
                "margin_score": spatial_feats.new_tensor(0.0),
            })
            continue

        phase_mask_float = phase_mask.unsqueeze(-1).to(dtype=ordered_feats.dtype)
        phase_feats = (ordered_feats * phase_mask_float).sum(dim=1) / phase_counts.clamp_min(1).unsqueeze(-1)
        phase_loss = phase_feats.sum() * 0.0
        valid_class_count = 0
        class_centers = []
        class_withins = []

        for class_id in labels.unique(sorted=True):
            class_mask = (labels == class_id) & valid_sample_mask
            class_count = int(class_mask.sum().item())
            if class_count < 2:
                continue

            class_phase_feats = phase_feats[class_mask]
            class_center = class_phase_feats.mean(dim=0, keepdim=True)
            class_within = (class_phase_feats - class_center).pow(2).sum(dim=1).mean()
            phase_loss = phase_loss + class_within
            valid_class_count += 1
            class_centers.append(class_center.squeeze(0))
            class_withins.append(class_within)

        if valid_class_count > 0:
            phase_loss = phase_loss / (valid_class_count + eps)
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = float(phase_loss.detach().item())
            phase_logs[f"phase_valid_samples_p{phase_idx + 1}"] = float(valid_sample_mask.sum().item())
            compactness_score = 1.0 / (phase_loss.detach() + eps)

            if len(class_centers) >= 2:
                centers = torch.stack(class_centers, dim=0)
                center_distances = torch.cdist(centers, centers, p=2)
                center_distances.fill_diagonal_(float("inf"))
                margin_score = center_distances.min(dim=1).values.mean().detach()
            else:
                margin_score = spatial_feats.new_tensor(0.0)

            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = float(margin_score.item())
            phase_structures.append({
                "phase_loss": phase_loss,
                "valid_class_count": valid_class_count,
                "compactness_score": compactness_score,
                "margin_score": margin_score,
            })
        else:
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_valid_samples_p{phase_idx + 1}"] = float(valid_sample_mask.sum().item())
            phase_structures.append({
                "phase_loss": phase_loss,
                "valid_class_count": 0,
                "compactness_score": spatial_feats.new_tensor(0.0),
                "margin_score": spatial_feats.new_tensor(0.0),
            })

    if weight_tracker is None:
        weights = _compute_phase_weights(phase_structures, spatial_feats, eps)
    else:
        weight_tracker.update(phase_structures)
        weights = weight_tracker.get_weights(spatial_feats, eps)

    for phase_idx, stats in enumerate(phase_structures):
        if stats["valid_class_count"] > 0:
            total_loss = total_loss + weights[phase_idx] * stats["phase_loss"]
        phase_logs[f"phase_weight_p{phase_idx + 1}"] = float(weights[phase_idx].detach().item())

    if weight_tracker is not None:
        phase_logs.update(weight_tracker.get_logs())

    active_intra_loss = (
        trajectory_intra_loss
        if version in {"trajectory_prototype_dynamics", "whole_curve_prototype_dynamics", "prototype_dynamics", "v244"}
        else intra_loss
    )
    total_loss = SOURCE_PHASE_COMPACTNESS_LAMBDA * total_loss
    phase_logs["compactness_loss"] = float(total_loss.detach().item())
    return total_loss, phase_logs


def compute_source_structure_loss(
    spatial_feats,
    positions,
    labels,
    weight_tracker=None,
    eps=1e-6,
    version=SOURCE_STRUCTURE_LOSS_VERSION,
    intra_trade_off=SOURCE_STRUCTURE_INTRA_TRADE_OFF,
    amplitude_trade_off=SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF,
    interphase_trade_off=SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF,
    shape_trade_off=SOURCE_STRUCTURE_SHAPE_TRADE_OFF,
    trend_trade_off=SOURCE_STRUCTURE_TREND_TRADE_OFF,
    season_trade_off=SOURCE_STRUCTURE_SEASON_TRADE_OFF,
    segment_inter_trade_off=SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF,
    boundary_window_trade_off=SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF,
    boundary_window_size=SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE,
    warp_invariant_trade_off=SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF,
    prototype_dynamics_trade_off=SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF,
    anchor_spatial_feats=None,
    anchor_positions=None,
):
    """
    Generic source-side structural loss.

    Versions:
    - compactness:
        original v2.2 / v2.3.1 phase-weighted phase-compactness objective
    - multi_component:
        v2.3.2 fixed-weight combination of:
          * intra-phase compactness
          * class-curve amplitude spread across phases
          * adjacent inter-phase smoothness of class centers
    - profiled_components:
        v2.3.3 fixed-weight combination of:
          * intra-phase compactness
          * shape regularization on reshaped class curves
            - discourage disorderly temporal deformation
            - discourage fragmented transition patterns
            - discourage anomalous phase-energy collapse
    - trend_residual:
        v2.3.4 decomposition-inspired objective:
          * residual/noise suppression via intra-phase compactness
          * low-frequency trend regularization on phase-level class-center sequences
    - trend_seasonal_residual:
        v2.3.5 source-only decomposition-inspired objective:
          * residual/noise suppression via intra-phase compactness
          * low-frequency trend regularization
          * seasonal-pattern regularization
            - discourage fragmented seasonal transitions
            - discourage redundant class-wise seasonal patterns
    - segment_trend_residual:
        v2.4.0 framework-refactor alias of v2.3.4:
          * keep residual/noise suppression + trend regularization unchanged
          * switch the code path onto the new temporal-segment abstraction
    - segment_transition_residual:
        v2.4.1 first segment-aware structural split:
          * keep residual/noise suppression as the main intra-segment term
          * keep trend regularization
          * add a weak adjacent inter-segment transition regularizer
    - segment_transition_semantic:
        v2.4.2 semantic-segment refinement:
          * keep the v2.4.1 loss unchanged
          * upgrade the segment partition builder from geometry-only DOY-gap rules
            to semantic-enhanced source-driven segmentation
    - segment_boundary_window_residual:
        v2.4.3b boundary-centered sliding-window weighting:
          * keep the v2.4.1 segment-aware residual + trend + weak inter-segment loss
          * use local boundary windows only to modulate the strength of adjacent
            inter-segment regularization
    - segment_boundary_window_warp_residual:
        GTW-inspired relaxed transition objective:
          * keep v2.4.3b terms
          * add a weak local monotonic-warp-invariant transition consistency term
    - trajectory_prototype_dynamics:
        v2.4.4 whole-curve structural objective:
          * class-wise full-trajectory compactness against source prototypes
          * class-wise first-difference consistency against prototype dynamics
          * does not require or assume meaningful segment boundaries
    """
    version = str(version).lower()
    if version == "compactness":
        return compute_source_phase_compactness_loss(
            spatial_feats,
            positions,
            labels,
            weight_tracker=weight_tracker,
            eps=eps,
        )

    if version not in {
        "multi_component",
        "multicomponent",
        "v232",
        "profiled_components",
        "profiled",
        "v233",
        "trend_residual",
        "trend",
        "v234",
        "segment_trend_residual",
        "segment_trend",
        "v240",
        "segment_transition_residual",
        "segment_transition",
        "segment_inter",
        "v241",
        "segment_transition_semantic",
        "v242",
        "segment_boundary_window_residual",
        "segment_boundary_window_warp_residual",
        "segment_boundary_window",
        "boundary_window_segment",
        "warp_boundary_window_segment",
        "v243",
        "trend_seasonal_residual",
        "trend_season",
        "season_pattern",
        "v235",
        "trajectory_prototype_dynamics",
        "whole_curve_prototype_dynamics",
        "prototype_dynamics",
        "v244",
    }:
        raise ValueError(f"Unsupported source structure loss version: {version}")

    if spatial_feats.ndim != 3:
        raise ValueError(f"Expected spatial_feats to have shape [B, T, D], got {tuple(spatial_feats.shape)}")

    batch_size, sequence_length, _ = spatial_feats.shape
    if batch_size < 2 or sequence_length < 2:
        zero = spatial_feats.sum() * 0.0
        return zero, {"structure_loss": 0.0, "compactness_loss": 0.0}

    phase_partition_spec = getattr(weight_tracker, "phase_partition_spec", None) if weight_tracker is not None else None
    min_sample_points = (
        getattr(weight_tracker, "min_sample_points_per_phase", SOURCE_PHASE_MIN_SAMPLE_POINTS)
        if weight_tracker is not None
        else SOURCE_PHASE_MIN_SAMPLE_POINTS
    )
    ordered_feats, ordered_positions = _sorted_sequence_features(spatial_feats, positions)
    phase_masks = _segment_masks_from_spec(ordered_positions, phase_partition_spec)
    zero = spatial_feats.sum() * 0.0
    phase_logs = {}
    phase_structures = []

    for phase_idx, phase_mask in enumerate(phase_masks):
        phase_counts = phase_mask.sum(dim=1)
        valid_sample_mask = phase_counts >= int(min_sample_points)
        if not bool(valid_sample_mask.any().item()):
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_valid_samples_p{phase_idx + 1}"] = 0.0
            phase_structures.append({
                "phase_loss": zero,
                "valid_class_count": 0,
                "compactness_score": spatial_feats.new_tensor(0.0),
                "margin_score": spatial_feats.new_tensor(0.0),
                "class_centers": {},
            })
            continue

        phase_mask_float = phase_mask.unsqueeze(-1).to(dtype=ordered_feats.dtype)
        phase_feats = (ordered_feats * phase_mask_float).sum(dim=1) / phase_counts.clamp_min(1).unsqueeze(-1)
        phase_loss = zero
        valid_class_count = 0
        class_centers = {}

        for class_id in labels.unique(sorted=True):
            class_mask = (labels == class_id) & valid_sample_mask
            class_count = int(class_mask.sum().item())
            if class_count < 2:
                continue

            class_phase_feats = phase_feats[class_mask]
            class_center = class_phase_feats.mean(dim=0, keepdim=True)
            class_within = (class_phase_feats - class_center).pow(2).sum(dim=1).mean()
            phase_loss = phase_loss + class_within
            valid_class_count += 1
            class_centers[int(class_id.item())] = class_center.squeeze(0)

        if valid_class_count > 0:
            phase_loss = phase_loss / (valid_class_count + eps)
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = float(phase_loss.detach().item())
            phase_logs[f"phase_valid_samples_p{phase_idx + 1}"] = float(valid_sample_mask.sum().item())
            compactness_score = 1.0 / (phase_loss.detach() + eps)

            if len(class_centers) >= 2:
                centers = torch.stack(list(class_centers.values()), dim=0)
                center_distances = torch.cdist(centers, centers, p=2)
                center_distances.fill_diagonal_(float("inf"))
                margin_score = center_distances.min(dim=1).values.mean().detach()
            else:
                margin_score = spatial_feats.new_tensor(0.0)

            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = float(margin_score.item())
            phase_structures.append({
                "phase_loss": phase_loss,
                "valid_class_count": valid_class_count,
                "compactness_score": compactness_score,
                "margin_score": margin_score,
                "class_centers": class_centers,
            })
        else:
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_valid_samples_p{phase_idx + 1}"] = float(valid_sample_mask.sum().item())
            phase_structures.append({
                "phase_loss": phase_loss,
                "valid_class_count": 0,
                "compactness_score": spatial_feats.new_tensor(0.0),
                "margin_score": spatial_feats.new_tensor(0.0),
                "class_centers": {},
            })

    if weight_tracker is None:
        weights = _compute_phase_weights(phase_structures, spatial_feats, eps)
    else:
        weight_tracker.update(phase_structures)
        weights = weight_tracker.get_weights(spatial_feats, eps)

    intra_loss = zero
    for phase_idx, stats in enumerate(phase_structures):
        if stats["valid_class_count"] > 0:
            intra_loss = intra_loss + weights[phase_idx] * stats["phase_loss"]
        phase_logs[f"phase_weight_p{phase_idx + 1}"] = float(weights[phase_idx].detach().item())

    class_ids = sorted({
        class_id
        for stats in phase_structures
        for class_id in stats["class_centers"].keys()
    })
    amplitude_loss = zero
    interphase_loss = zero
    shape_loss = zero
    shape_disorder_loss = zero
    shape_fragment_loss = zero
    shape_collapse_loss = zero
    trajectory_intra_loss = zero
    prototype_dynamics_loss = zero
    trend_loss = zero
    season_loss = zero
    season_coherence_loss = zero
    season_redundancy_loss = zero
    segment_inter_loss = zero
    boundary_window_weight_signal = zero
    warp_invariant_loss = zero
    amplitude_class_count = 0
    interphase_class_count = 0
    shape_class_count = 0
    trajectory_class_count = 0
    prototype_dynamics_class_count = 0
    trend_class_count = 0
    season_class_count = 0
    segment_inter_class_count = 0
    boundary_window_class_count = 0
    warp_invariant_class_count = 0

    if version in {"trajectory_prototype_dynamics", "whole_curve_prototype_dynamics", "prototype_dynamics", "v244"}:
        (
            trajectory_intra_loss,
            prototype_dynamics_loss,
            trajectory_class_count,
            prototype_dynamics_class_count,
        ) = _compute_trajectory_prototype_losses(ordered_feats, labels, eps=eps)
    elif version in {"profiled_components", "profiled", "v233"}:
        for class_id in class_ids:
            valid_phase_indices = [
                phase_idx
                for phase_idx, stats in enumerate(phase_structures)
                if class_id in stats["class_centers"]
            ]
            if len(valid_phase_indices) < 3:
                continue

            centers = torch.stack(
                [phase_structures[phase_idx]["class_centers"][class_id] for phase_idx in valid_phase_indices],
                dim=0,
            )
            class_weights = weights[torch.tensor(valid_phase_indices, device=weights.device)]
            class_weights = class_weights / class_weights.sum().clamp_min(eps)

            diffs = centers[1:] - centers[:-1]
            pair_weights = 0.5 * (class_weights[1:] + class_weights[:-1])
            pair_weights = pair_weights / pair_weights.sum().clamp_min(eps)
            diff_energy = diffs.pow(2).sum(dim=1)

            second_diffs = diffs[1:] - diffs[:-1]
            second_weights = 0.5 * (pair_weights[1:] + pair_weights[:-1])
            second_weights = second_weights / second_weights.sum().clamp_min(eps)
            class_disorder = (
                second_diffs.pow(2).sum(dim=1) * second_weights
            ).sum() / (diff_energy.mul(pair_weights).sum().clamp_min(eps))

            cosine = torch.nn.functional.cosine_similarity(diffs[:-1], diffs[1:], dim=1, eps=eps)
            class_fragment = ((1.0 - cosine) * second_weights).sum()

            center_energy = centers.norm(dim=1)
            center_profile = center_energy / center_energy.sum().clamp_min(eps)
            concentration = center_profile.pow(2).sum()
            uniform_floor = 1.0 / max(center_profile.numel(), 1)
            normalized_collapse = (concentration - uniform_floor) / max(1.0 - uniform_floor, eps)
            class_collapse = torch.relu(
                normalized_collapse - SHAPE_REG_COLLAPSE_MARGIN
            ).pow(2)

            class_shape = (
                class_disorder
                + SHAPE_REG_DIRECTION_TRADE_OFF * class_fragment
                + SHAPE_REG_COLLAPSE_TRADE_OFF * class_collapse
            )
            shape_disorder_loss = shape_disorder_loss + class_disorder
            shape_fragment_loss = shape_fragment_loss + class_fragment
            shape_collapse_loss = shape_collapse_loss + class_collapse
            shape_loss = shape_loss + class_shape
            shape_class_count += 1
    elif version in {
        "trend_residual",
        "trend",
        "v234",
        "segment_trend_residual",
        "segment_trend",
        "v240",
        "segment_transition_residual",
        "segment_transition",
        "segment_inter",
        "v241",
        "segment_transition_semantic",
        "v242",
        "segment_boundary_window_residual",
        "segment_boundary_window_warp_residual",
        "segment_boundary_window",
        "boundary_window_segment",
        "warp_boundary_window_segment",
        "v243",
        "trend_seasonal_residual",
        "trend_season",
        "season_pattern",
        "v235",
    }:
        season_vectors = []
        for class_id in class_ids:
            class_mask = (labels == class_id)
            valid_phase_indices = [
                phase_idx
                for phase_idx, stats in enumerate(phase_structures)
                if class_id in stats["class_centers"]
            ]
            if len(valid_phase_indices) < 3:
                continue

            centers = torch.stack(
                [phase_structures[phase_idx]["class_centers"][class_id] for phase_idx in valid_phase_indices],
                dim=0,
            )
            trend_centers = _moving_average_same(centers)
            if trend_centers.shape[0] < 3:
                continue

            second_diffs = trend_centers[2:] - 2.0 * trend_centers[1:-1] + trend_centers[:-2]
            class_weights = weights[torch.tensor(valid_phase_indices, device=weights.device)]
            interior_weights = class_weights[1:-1]
            interior_weights = interior_weights / interior_weights.sum().clamp_min(eps)
            class_trend = (second_diffs.pow(2).sum(dim=1) * interior_weights).sum()
            trend_loss = trend_loss + class_trend
            trend_class_count += 1

            if version in {"segment_transition_residual", "segment_transition", "segment_inter", "v241", "segment_transition_semantic", "v242", "segment_boundary_window_residual", "segment_boundary_window_warp_residual", "segment_boundary_window", "boundary_window_segment", "warp_boundary_window_segment", "v243"}:
                seasonal_centers = centers - trend_centers
                if seasonal_centers.shape[0] >= 3:
                    seasonal_diffs = seasonal_centers[1:] - seasonal_centers[:-1]
                    if seasonal_diffs.shape[0] >= 2:
                        seasonal_cos = torch.nn.functional.cosine_similarity(
                            seasonal_diffs[:-1],
                            seasonal_diffs[1:],
                            dim=1,
                            eps=eps,
                        )
                        seasonal_pair_weights = 0.5 * (class_weights[1:] + class_weights[:-1])
                        seasonal_transition_weights = 0.5 * (
                            seasonal_pair_weights[1:] + seasonal_pair_weights[:-1]
                        )
                        seasonal_transition_weights = (
                            seasonal_transition_weights
                            / seasonal_transition_weights.sum().clamp_min(eps)
                        )
                        boundary_pair_modulator = None

                        if version in {"segment_boundary_window_warp_residual", "warp_boundary_window_segment"} and seasonal_diffs.shape[0] >= 3:
                            adjacent_mismatch = (
                                seasonal_diffs[1:] - seasonal_diffs[:-1]
                            ).pow(2).sum(dim=1)
                            skip_mismatch = (
                                seasonal_diffs[2:] - seasonal_diffs[:-2]
                            ).pow(2).sum(dim=1)
                            relaxed_mismatch = torch.cat(
                                [
                                    torch.minimum(adjacent_mismatch[:-1], skip_mismatch),
                                    adjacent_mismatch[-1:],
                                ],
                                dim=0,
                            )
                            relaxed_weights = seasonal_transition_weights
                            relaxed_weights = relaxed_weights / relaxed_weights.sum().clamp_min(eps)
                            warp_invariant_loss = warp_invariant_loss + (
                                relaxed_mismatch * relaxed_weights
                            ).sum()
                            warp_invariant_class_count += 1

                if version in {"segment_boundary_window_residual", "segment_boundary_window_warp_residual", "segment_boundary_window", "boundary_window_segment", "warp_boundary_window_segment", "v243"} and phase_partition_spec is not None:
                    if class_mask.sum().item() >= 2:
                        class_time_curve = ordered_feats[class_mask].mean(dim=0)
                        class_time_trend = _moving_average_same(class_time_curve)
                        class_time_seasonal = class_time_curve - class_time_trend
                        phase_index_tensor = torch.tensor(valid_phase_indices, device=weights.device, dtype=torch.long)
                        if phase_index_tensor.numel() >= 2:
                            consecutive_pairs = []
                            for local_idx in range(phase_index_tensor.numel() - 1):
                                left_idx = int(phase_index_tensor[local_idx].item())
                                right_idx = int(phase_index_tensor[local_idx + 1].item())
                                if right_idx == left_idx + 1:
                                    consecutive_pairs.append((left_idx, right_idx, local_idx))
                            if consecutive_pairs:
                                spec_positions = phase_partition_spec.get("date_positions")
                                spec_intervals = phase_partition_spec.get("intervals")
                                if spec_positions is not None and spec_intervals is not None:
                                    spec_positions = np.asarray(spec_positions, dtype=np.int64)
                                    boundary_terms = []
                                    boundary_weights = []
                                    boundary_local_indices = []
                                    window_size = max(1, int(boundary_window_size))
                                    for left_phase_idx, right_phase_idx, local_idx in consecutive_pairs:
                                        left_end_pos = int(spec_intervals[left_phase_idx][1])
                                        boundary_candidates = np.where(spec_positions == left_end_pos)[0]
                                        if boundary_candidates.size == 0:
                                            continue
                                        boundary_idx = int(boundary_candidates[-1])
                                        left_start = max(0, boundary_idx - window_size + 1)
                                        right_end = min(class_time_seasonal.shape[0], boundary_idx + 1 + window_size)
                                        left_window = class_time_seasonal[left_start:boundary_idx + 1]
                                        right_window = class_time_seasonal[boundary_idx + 1:right_end]
                                        if left_window.shape[0] == 0 or right_window.shape[0] == 0:
                                            continue
                                        window_transition = right_window.mean(dim=0) - left_window.mean(dim=0)
                                        boundary_terms.append(window_transition.norm())
                                        boundary_weights.append(
                                            0.5 * (
                                                class_weights[local_idx]
                                                + class_weights[local_idx + 1]
                                            )
                                        )
                                        boundary_local_indices.append(local_idx)
                                    if boundary_terms:
                                        boundary_terms = torch.stack(boundary_terms)
                                        boundary_weights = torch.stack(boundary_weights)
                                        boundary_weights = boundary_weights / boundary_weights.sum().clamp_min(eps)
                                        normalized_boundary_terms = boundary_terms / boundary_terms.mean().clamp_min(eps)
                                        class_boundary_window = (normalized_boundary_terms * boundary_weights).sum()
                                        boundary_window_weight_signal = boundary_window_weight_signal + class_boundary_window
                                        boundary_window_class_count += 1
                                        if seasonal_centers.shape[0] >= 3 and seasonal_diffs.shape[0] >= 2:
                                            boundary_strengths = centers.new_ones(seasonal_diffs.shape[0])
                                            for local_idx, normalized_term in zip(boundary_local_indices, normalized_boundary_terms):
                                                if 0 <= local_idx < boundary_strengths.shape[0]:
                                                    boundary_strengths[local_idx] = normalized_term
                                            boundary_pair_modulator = 0.5 * (
                                                boundary_strengths[:-1] + boundary_strengths[1:]
                                            )

                if seasonal_centers.shape[0] >= 3 and seasonal_diffs.shape[0] >= 2:
                    if boundary_pair_modulator is not None:
                        seasonal_transition_weights = (
                            seasonal_transition_weights
                            * (1.0 + float(boundary_window_trade_off) * boundary_pair_modulator)
                        )
                        seasonal_transition_weights = (
                            seasonal_transition_weights
                            / seasonal_transition_weights.sum().clamp_min(eps)
                        )
                    class_segment_inter = (
                        (1.0 - seasonal_cos) * seasonal_transition_weights
                    ).sum()
                    segment_inter_loss = segment_inter_loss + class_segment_inter
                    segment_inter_class_count += 1

            if version in {"trend_seasonal_residual", "trend_season", "season_pattern", "v235"}:
                seasonal_centers = centers - trend_centers
                if seasonal_centers.shape[0] >= 3:
                    seasonal_diffs = seasonal_centers[1:] - seasonal_centers[:-1]
                    seasonal_cos = torch.nn.functional.cosine_similarity(
                        seasonal_diffs[:-1],
                        seasonal_diffs[1:],
                        dim=1,
                        eps=eps,
                    )
                    seasonal_pair_weights = 0.5 * (class_weights[1:] + class_weights[:-1])
                    seasonal_transition_weights = 0.5 * (
                        seasonal_pair_weights[1:] + seasonal_pair_weights[:-1]
                    )
                    seasonal_transition_weights = seasonal_transition_weights / seasonal_transition_weights.sum().clamp_min(eps)
                    class_season_coherence = ((1.0 - seasonal_cos) * seasonal_transition_weights).sum()
                    season_coherence_loss = season_coherence_loss + class_season_coherence

                seasonal_full = centers.new_zeros((len(phase_structures), centers.shape[-1]))
                seasonal_full[torch.tensor(valid_phase_indices, device=centers.device)] = seasonal_centers
                season_vectors.append(seasonal_full.reshape(-1))
                season_class_count += 1

        if version in {"trend_seasonal_residual", "trend_season", "season_pattern", "v235"} and len(season_vectors) >= 2:
            stacked = torch.stack(season_vectors, dim=0)
            normalized = torch.nn.functional.normalize(stacked, dim=1, eps=eps)
            cosine_matrix = normalized @ normalized.transpose(0, 1)
            redundancy_matrix = cosine_matrix.pow(2)
            redundancy_matrix.fill_diagonal_(0.0)
            denom = max(redundancy_matrix.numel() - redundancy_matrix.shape[0], 1)
            season_redundancy_loss = redundancy_matrix.sum() / float(denom)
    else:
        for class_id in class_ids:
            valid_phase_indices = [
                phase_idx
                for phase_idx, stats in enumerate(phase_structures)
                if class_id in stats["class_centers"]
            ]
            if len(valid_phase_indices) < 2:
                continue

            centers = torch.stack(
                [phase_structures[phase_idx]["class_centers"][class_id] for phase_idx in valid_phase_indices],
                dim=0,
            )
            class_weights = weights[torch.tensor(valid_phase_indices, device=weights.device)]
            class_weights = class_weights / class_weights.sum().clamp_min(eps)

            temporal_center = (class_weights.unsqueeze(-1) * centers).sum(dim=0, keepdim=True)
            class_amplitude = ((centers - temporal_center).pow(2).sum(dim=1) * class_weights).sum()
            amplitude_loss = amplitude_loss + class_amplitude
            amplitude_class_count += 1

            if centers.shape[0] >= 2:
                diffs = centers[1:] - centers[:-1]
                diff_energy = diffs.pow(2).sum(dim=1)
                pair_weights = 0.5 * (class_weights[1:] + class_weights[:-1])
                pair_weights = pair_weights / pair_weights.sum().clamp_min(eps)
                class_interphase = (diff_energy * pair_weights).sum()
                interphase_loss = interphase_loss + class_interphase
                interphase_class_count += 1

    if amplitude_class_count > 0:
        amplitude_loss = amplitude_loss / amplitude_class_count
    if interphase_class_count > 0:
        interphase_loss = interphase_loss / interphase_class_count
    if shape_class_count > 0:
        shape_loss = shape_loss / shape_class_count
        shape_disorder_loss = shape_disorder_loss / shape_class_count
        shape_fragment_loss = shape_fragment_loss / shape_class_count
        shape_collapse_loss = shape_collapse_loss / shape_class_count
    if trend_class_count > 0:
        trend_loss = trend_loss / trend_class_count
    if season_class_count > 0:
        season_coherence_loss = season_coherence_loss / season_class_count
    if segment_inter_class_count > 0:
        segment_inter_loss = segment_inter_loss / segment_inter_class_count
    if boundary_window_class_count > 0:
        boundary_window_weight_signal = boundary_window_weight_signal / boundary_window_class_count
    if warp_invariant_class_count > 0:
        warp_invariant_loss = warp_invariant_loss / warp_invariant_class_count
    if version in {"trend_seasonal_residual", "trend_season", "season_pattern", "v235"}:
        season_loss = (
            season_coherence_loss
            + SEASON_REG_REDUNDANCY_TRADE_OFF * season_redundancy_loss
        )

    if version in {"trajectory_prototype_dynamics", "whole_curve_prototype_dynamics", "prototype_dynamics", "v244"}:
        total_loss = (
            float(intra_trade_off) * trajectory_intra_loss
            + float(prototype_dynamics_trade_off) * prototype_dynamics_loss
        )
    elif version in {"profiled_components", "profiled", "v233"}:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(shape_trade_off) * shape_loss
        )
    elif version in {"trend_seasonal_residual", "trend_season", "season_pattern", "v235"}:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(trend_trade_off) * trend_loss
            + float(season_trade_off) * season_loss
        )
    elif version in {"segment_transition_residual", "segment_transition", "segment_inter", "v241", "segment_transition_semantic", "v242"}:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(trend_trade_off) * trend_loss
            + float(segment_inter_trade_off) * segment_inter_loss
        )
    elif version in {"segment_boundary_window_residual", "segment_boundary_window", "boundary_window_segment", "v243"}:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(trend_trade_off) * trend_loss
            + float(segment_inter_trade_off) * segment_inter_loss
        )
    elif version in {"segment_boundary_window_warp_residual", "warp_boundary_window_segment"}:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(trend_trade_off) * trend_loss
            + float(segment_inter_trade_off) * segment_inter_loss
            + float(warp_invariant_trade_off) * warp_invariant_loss
        )
    elif version in {"trend_residual", "trend", "v234", "segment_trend_residual", "segment_trend", "v240"}:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(trend_trade_off) * trend_loss
        )
    else:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(amplitude_trade_off) * amplitude_loss
            + float(interphase_trade_off) * interphase_loss
        )

    if weight_tracker is not None:
        phase_logs.update(weight_tracker.get_logs())

    total_loss = SOURCE_PHASE_COMPACTNESS_LAMBDA * total_loss
    phase_logs["source_structure_loss_version"] = (
        1.0 if version in {"multi_component", "multicomponent", "v232"} else 2.0
    )
    phase_logs["source_structure_intra_loss"] = float((SOURCE_PHASE_COMPACTNESS_LAMBDA * active_intra_loss).detach().item())
    phase_logs["source_structure_amplitude_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * amplitude_loss).detach().item()
    )
    phase_logs["source_structure_interphase_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * interphase_loss).detach().item()
    )
    phase_logs["source_structure_shape_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * shape_loss).detach().item()
    )
    phase_logs["source_structure_shape_disorder_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * shape_disorder_loss).detach().item()
    )
    phase_logs["source_structure_shape_fragment_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * shape_fragment_loss).detach().item()
    )
    phase_logs["source_structure_shape_collapse_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * shape_collapse_loss).detach().item()
    )
    phase_logs["source_structure_trajectory_intra_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * trajectory_intra_loss).detach().item()
    )
    phase_logs["source_structure_prototype_dynamics_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * prototype_dynamics_loss).detach().item()
    )
    phase_logs["source_structure_trend_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * trend_loss).detach().item()
    )
    phase_logs["source_structure_season_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * season_loss).detach().item()
    )
    phase_logs["source_structure_season_coherence_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * season_coherence_loss).detach().item()
    )
    phase_logs["source_structure_season_redundancy_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * season_redundancy_loss).detach().item()
    )
    phase_logs["source_structure_segment_inter_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * segment_inter_loss).detach().item()
    )
    phase_logs["source_structure_boundary_window_weight_signal"] = float(
        boundary_window_weight_signal.detach().item()
    )
    phase_logs["source_structure_warp_invariant_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * warp_invariant_loss).detach().item()
    )
    phase_logs["source_structure_amplitude_classes"] = float(amplitude_class_count)
    phase_logs["source_structure_interphase_classes"] = float(interphase_class_count)
    phase_logs["source_structure_shape_classes"] = float(shape_class_count)
    phase_logs["source_structure_trajectory_classes"] = float(trajectory_class_count)
    phase_logs["source_structure_prototype_dynamics_classes"] = float(prototype_dynamics_class_count)
    phase_logs["source_structure_trend_classes"] = float(trend_class_count)
    phase_logs["source_structure_season_classes"] = float(season_class_count)
    phase_logs["source_structure_segment_inter_classes"] = float(segment_inter_class_count)
    phase_logs["source_structure_boundary_window_classes"] = float(boundary_window_class_count)
    phase_logs["source_structure_warp_invariant_classes"] = float(warp_invariant_class_count)
    phase_logs["source_structure_segment_count"] = float(len(phase_structures))
    phase_logs["structure_loss"] = float(total_loss.detach().item())
    phase_logs["compactness_loss"] = phase_logs["structure_loss"]
    return total_loss, phase_logs
