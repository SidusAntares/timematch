import torch

from ideas.source_phase_compactness import _moving_average_same, _segment_masks_from_spec, _sorted_sequence_features


def _safe_factor(reliability, strength, min_factor, max_factor, baseline=0.65):
    factor = 1.0 + float(strength) * (float(reliability) - float(baseline))
    return max(float(min_factor), min(float(max_factor), factor))


def _energy_reliability(samples, zeta=0.90, eps=1e-6):
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        return None

    centered = samples - samples.mean(dim=0, keepdim=True)
    if torch.all(centered.detach().abs().sum() <= eps):
        return 1.0

    values = torch.linalg.svdvals(centered.float()).pow(2)
    total = values.sum()
    if not bool((total > eps).item()):
        return 1.0

    cumulative = torch.cumsum(values, dim=0) / total.clamp_min(eps)
    reached = torch.nonzero(cumulative >= float(zeta), as_tuple=False)
    if reached.numel() == 0:
        effective_k = int(values.numel())
    else:
        effective_k = int(reached[0].item()) + 1

    max_k = max(int(values.numel()), 1)
    if max_k <= 1:
        return 1.0
    reliability = 1.0 - float(effective_k - 1) / float(max_k - 1)
    return max(0.0, min(1.0, reliability))


def _mean_or_default(values, default=0.65):
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def compute_svd_structure_reliability_factors(
    spatial_feats,
    positions,
    labels,
    phase_partition_spec=None,
    min_sample_points=2,
    zeta=0.90,
    strength=0.35,
    min_factor=0.70,
    max_factor=1.20,
    eps=1e-6,
):
    if spatial_feats.ndim != 3:
        raise ValueError(f"Expected spatial_feats to have shape [B, T, D], got {tuple(spatial_feats.shape)}")

    ordered_feats, ordered_positions = _sorted_sequence_features(spatial_feats.detach(), positions)
    phase_masks = _segment_masks_from_spec(ordered_positions, phase_partition_spec)

    phase_structures = []
    intra_reliabilities = []
    for phase_mask in phase_masks:
        phase_counts = phase_mask.sum(dim=1)
        valid_sample_mask = phase_counts >= int(min_sample_points)
        phase_mask_float = phase_mask.unsqueeze(-1).to(dtype=ordered_feats.dtype)
        phase_feats = (ordered_feats * phase_mask_float).sum(dim=1) / phase_counts.clamp_min(1).unsqueeze(-1)
        class_centers = {}

        for class_id in labels.unique(sorted=True):
            class_mask = (labels == class_id) & valid_sample_mask
            if int(class_mask.sum().item()) < 2:
                continue

            class_phase_feats = phase_feats[class_mask]
            reliability = _energy_reliability(class_phase_feats, zeta=zeta, eps=eps)
            if reliability is not None:
                intra_reliabilities.append(reliability)
            class_centers[int(class_id.item())] = class_phase_feats.mean(dim=0)

        phase_structures.append({"class_centers": class_centers})

    class_ids = sorted({
        class_id
        for stats in phase_structures
        for class_id in stats["class_centers"].keys()
    })
    trend_reliabilities = []
    transition_reliabilities = []

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
        trend_reliability = _energy_reliability(trend_centers, zeta=zeta, eps=eps)
        if trend_reliability is not None:
            trend_reliabilities.append(trend_reliability)

        seasonal_centers = centers - trend_centers
        if seasonal_centers.shape[0] >= 3:
            seasonal_diffs = seasonal_centers[1:] - seasonal_centers[:-1]
            transition_reliability = _energy_reliability(seasonal_diffs, zeta=zeta, eps=eps)
            if transition_reliability is not None:
                transition_reliabilities.append(transition_reliability)

    intra_reliability = _mean_or_default(intra_reliabilities)
    trend_reliability = _mean_or_default(trend_reliabilities)
    transition_reliability = _mean_or_default(transition_reliabilities)

    intra_factor = _safe_factor(intra_reliability, strength, min_factor, max_factor)
    trend_factor = _safe_factor(trend_reliability, strength, min_factor, max_factor)
    transition_factor = _safe_factor(transition_reliability, strength, min_factor, max_factor)

    factors = {
        "intra": intra_factor,
        "trend": trend_factor,
        "segment_inter": transition_factor,
        "boundary_window": transition_factor,
    }
    logs = {
        "source_structure_svd_intra_reliability": intra_reliability,
        "source_structure_svd_trend_reliability": trend_reliability,
        "source_structure_svd_transition_reliability": transition_reliability,
        "source_structure_svd_intra_factor": intra_factor,
        "source_structure_svd_trend_factor": trend_factor,
        "source_structure_svd_segment_inter_factor": transition_factor,
        "source_structure_svd_boundary_window_factor": transition_factor,
        "source_structure_svd_intra_units": float(len(intra_reliabilities)),
        "source_structure_svd_trend_units": float(len(trend_reliabilities)),
        "source_structure_svd_transition_units": float(len(transition_reliabilities)),
    }
    return factors, logs
