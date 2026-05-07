import math

import numpy as np
import torch


UNIFORM_PHASE_COUNT = 5
SOURCE_PHASE_COMPACTNESS_LAMBDA = 0.05
SOURCE_STRUCTURE_LOSS_VERSION = "compactness"
SOURCE_STRUCTURE_INTRA_TRADE_OFF = 1.0
SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF = 0.25
SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF = 0.25
SOURCE_STRUCTURE_SHAPE_TRADE_OFF = 0.15
SOURCE_STRUCTURE_TREND_TRADE_OFF = 0.05
SHAPE_REG_DIRECTION_TRADE_OFF = 0.5
SHAPE_REG_COLLAPSE_TRADE_OFF = 0.5
SHAPE_REG_COLLAPSE_MARGIN = 0.35
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

    if mode != "doy_gap":
        raise ValueError(f"Unsupported source phase partition mode: {mode}")

    base_segments = []
    start_idx = 0
    for idx in range(1, sorted_positions.size):
        if int(sorted_positions[idx] - sorted_positions[idx - 1]) > int(gap_threshold):
            base_segments.append(sorted_positions[start_idx:idx])
            start_idx = idx
    base_segments.append(sorted_positions[start_idx:])

    split_segments = []
    max_points = max(1, int(max_points))
    max_span = max(1, int(max_span))
    for segment in base_segments:
        span = int(segment[-1] - segment[0]) if len(segment) > 0 else 0
        split_count = max(
            1,
            int(math.ceil(len(segment) / max_points)),
            int(math.ceil(max(span, 1) / max_span)),
        )
        for split_segment in np.array_split(segment, split_count):
            if len(split_segment) > 0:
                split_segments.append(split_segment)

    merged_segments = _merge_small_segments(split_segments, max(1, int(min_points)))
    intervals = [(int(segment[0]), int(segment[-1])) for segment in merged_segments]
    return {
        "mode": "doy_gap",
        "phase_count": len(intervals),
        "intervals": intervals,
        "date_positions": sorted_positions.tolist(),
        "gap_threshold": int(gap_threshold),
        "min_points": int(min_points),
        "max_points": int(max_points),
        "max_span": int(max_span),
    }


def describe_source_phase_partition_spec(spec):
    if spec["mode"] == "uniform":
        return f"mode=uniform, phase_count={spec['phase_count']}"
    interval_text = ", ".join([f"[{start},{end}]" for start, end in spec["intervals"]])
    return (
        f"mode=doy_gap, phase_count={spec['phase_count']}, "
        f"gap_threshold={spec['gap_threshold']}, min_points={spec['min_points']}, "
        f"max_points={spec['max_points']}, max_span={spec['max_span']}, intervals={interval_text}"
    )


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
        return logs


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
    phase_masks = _phase_masks_from_spec(ordered_positions, phase_partition_spec)
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
    trend_loss = zero
    amplitude_class_count = 0
    interphase_class_count = 0
    shape_class_count = 0
    trend_class_count = 0

    if version in {"profiled_components", "profiled", "v233"}:
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
    elif version in {"trend_residual", "trend", "v234"}:
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

    if version in {"profiled_components", "profiled", "v233"}:
        total_loss = (
            float(intra_trade_off) * intra_loss
            + float(shape_trade_off) * shape_loss
        )
    elif version in {"trend_residual", "trend", "v234"}:
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
    phase_logs["source_structure_intra_loss"] = float((SOURCE_PHASE_COMPACTNESS_LAMBDA * intra_loss).detach().item())
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
    phase_logs["source_structure_trend_loss"] = float(
        (SOURCE_PHASE_COMPACTNESS_LAMBDA * trend_loss).detach().item()
    )
    phase_logs["source_structure_amplitude_classes"] = float(amplitude_class_count)
    phase_logs["source_structure_interphase_classes"] = float(interphase_class_count)
    phase_logs["source_structure_shape_classes"] = float(shape_class_count)
    phase_logs["source_structure_trend_classes"] = float(trend_class_count)
    phase_logs["structure_loss"] = float(total_loss.detach().item())
    phase_logs["compactness_loss"] = phase_logs["structure_loss"]
    return total_loss, phase_logs
