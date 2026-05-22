import torch
import torch.nn.functional as F


def build_temporal_window_weights(
    count,
    mode="none",
    center=0.5,
    width=0.35,
    min_weight=0.15,
    device=None,
    dtype=None,
):
    """Build white-box temporal weights for phase/segment-level structure loss.

    The returned weights are relative importance scores. Callers should combine
    them with validity-aware phase weights and renormalize to keep the total
    structure-loss scale stable.
    """
    count = int(count)
    if count <= 0:
        return torch.empty(0, device=device, dtype=dtype or torch.float32)

    mode = str(mode or "none").lower()
    weights = torch.ones(count, device=device, dtype=dtype or torch.float32)
    if mode in {"none", "full", "uniform", "source_target_mask", "source_target_static_mask", "source_target_soft_support"}:
        return weights

    positions = torch.linspace(0.0, 1.0, steps=count, device=device, dtype=weights.dtype)
    min_weight = float(min_weight)
    min_weight = max(0.0, min(1.0, min_weight))

    if mode == "early":
        center = 0.20
    elif mode in {"middle", "mid"}:
        center = 0.50
    elif mode == "late":
        center = 0.80
    elif mode == "gaussian":
        center = float(center)
    else:
        raise ValueError(f"Unsupported source temporal window mode: {mode}")

    width = max(float(width), 1e-3)
    gaussian = torch.exp(-0.5 * ((positions - float(center)) / width).pow(2))
    if gaussian.max().item() > 0:
        gaussian = gaussian / gaussian.max().clamp_min(1e-6)
    return min_weight + (1.0 - min_weight) * gaussian


def apply_temporal_window_to_phase_weights(phase_weights, window_weights, eps=1e-6):
    if window_weights is None:
        return phase_weights
    if phase_weights.numel() != window_weights.numel():
        raise ValueError(
            "Temporal window length must match phase weight length: "
            f"{window_weights.numel()} vs {phase_weights.numel()}"
        )
    adjusted = phase_weights * window_weights.to(device=phase_weights.device, dtype=phase_weights.dtype)
    if float(adjusted.sum().detach().item()) <= eps:
        return phase_weights
    return adjusted / adjusted.sum().clamp_min(eps)


def _relative_index_masks(batch_size, sequence_length, count, device):
    count = max(1, int(count))
    indices = torch.arange(sequence_length, device=device)
    masks = []
    for phase_idx in range(count):
        start = int(round(phase_idx * sequence_length / count))
        end = int(round((phase_idx + 1) * sequence_length / count))
        if phase_idx == count - 1:
            end = sequence_length
        mask_1d = (indices >= start) & (indices < max(end, start + 1))
        masks.append(mask_1d.unsqueeze(0).expand(batch_size, -1))
    return masks


def _source_phase_masks_from_spec(ordered_positions, phase_partition_spec, count):
    batch_size, sequence_length = ordered_positions.shape
    if phase_partition_spec is None or phase_partition_spec.get("intervals") is None:
        return _relative_index_masks(batch_size, sequence_length, count, ordered_positions.device)

    masks = []
    for start, end in phase_partition_spec.get("intervals"):
        masks.append((ordered_positions >= int(start)) & (ordered_positions <= int(end)))
    if not masks:
        return _relative_index_masks(batch_size, sequence_length, count, ordered_positions.device)
    return masks


def _masked_mean(features, mask, eps=1e-6):
    mask_float = mask.unsqueeze(-1).to(dtype=features.dtype)
    denom = mask_float.sum(dim=1).clamp_min(eps)
    return (features * mask_float).sum(dim=1) / denom


def _minmax(values, eps=1e-6):
    if values.numel() <= 1:
        return torch.ones_like(values)
    lo = values.min()
    hi = values.max()
    if float((hi - lo).detach().item()) <= eps:
        return torch.ones_like(values)
    return (values - lo) / (hi - lo + eps)


def _smooth_1d(values, kernel_size=5, eps=1e-6):
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or values.numel() <= 2:
        return values
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(float(kernel_size) / 3.0, eps)
    offsets = torch.arange(kernel_size, device=values.device, dtype=values.dtype)
    offsets = offsets - float(kernel_size // 2)
    kernel = torch.exp(-0.5 * (offsets / sigma).pow(2))
    kernel = kernel / kernel.sum().clamp_min(eps)
    padded = F.pad(values.view(1, 1, -1), (kernel_size // 2, kernel_size // 2), mode="replicate")
    return F.conv1d(padded, kernel.view(1, 1, -1)).view(-1)


def _sort_temporal_features(features, positions):
    if positions is None:
        return features, positions
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    return torch.gather(features, dim=1, index=expanded), torch.gather(positions, dim=1, index=sort_indices)


def _listify(values):
    return [float(value.detach().item()) for value in values]


def _weight_diagnostics(weights, eps=1e-6):
    total = weights.sum().clamp_min(eps)
    probs = weights / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    max_entropy = torch.log(weights.new_tensor(float(max(weights.numel(), 1)))).clamp_min(eps)
    effective_size = total.pow(2) / weights.pow(2).sum().clamp_min(eps)
    return {
        "entropy": entropy,
        "entropy_norm": entropy / max_entropy,
        "effective_size": effective_size,
        "effective_ratio": effective_size / float(max(weights.numel(), 1)),
        "std": weights.std(unbiased=False) if weights.numel() > 1 else weights.new_tensor(0.0),
    }


def compute_source_target_temporal_mask(
    source_features,
    labels,
    target_features,
    source_positions=None,
    phase_partition_spec=None,
    phase_count=None,
    min_sample_points=1,
    min_weight=0.15,
    reliability_gate=True,
    reliability_low=5e-4,
    reliability_high=4e-2,
    eps=1e-6,
):
    """Estimate which temporal phases are worth structuring.

    This is a white-box v2.6.2b mask:
    - source discriminativeness: classes separate inside this phase
    - target explainability: target phase features have a clear nearest source
      prototype instead of lying between classes
    - mismatch penalty: nearest target-source prototype distance should not be
      too large relative to the source inter-class scale
    """
    if source_features.ndim != 3 or target_features.ndim != 3:
        raise ValueError("Expected [B, T, D] source and target features")

    source_features, source_positions = _sort_temporal_features(source_features, source_positions)

    if target_features.shape[1] != source_features.shape[1]:
        min_len = min(source_features.shape[1], target_features.shape[1])
        source_features = source_features[:, :min_len]
        target_features = target_features[:, :min_len]
        if source_positions is not None:
            source_positions = source_positions[:, :min_len]

    batch_size, sequence_length, _ = source_features.shape
    if phase_partition_spec is not None and phase_partition_spec.get("intervals") is not None:
        count = len(phase_partition_spec.get("intervals"))
    elif phase_count is not None:
        count = int(phase_count)
    else:
        count = 1
    source_masks = _source_phase_masks_from_spec(source_positions, phase_partition_spec, count) if source_positions is not None else _relative_index_masks(batch_size, sequence_length, count, source_features.device)
    target_masks = _relative_index_masks(target_features.shape[0], sequence_length, len(source_masks), target_features.device)

    scores = []
    source_disc_scores = []
    target_margin_scores = []
    mismatch_scores = []
    valid_flags = []

    for phase_idx, source_mask in enumerate(source_masks):
        target_mask = target_masks[phase_idx]
        source_counts = source_mask.sum(dim=1)
        valid_source = source_counts >= int(min_sample_points)
        centers = []
        compactness = []
        for class_id in labels.unique(sorted=True):
            class_mask = (labels == class_id) & valid_source
            if int(class_mask.sum().item()) < 2:
                continue
            class_phase_features = _masked_mean(source_features[class_mask], source_mask[class_mask], eps=eps)
            center = class_phase_features.mean(dim=0)
            centers.append(center)
            compactness.append((class_phase_features - center).pow(2).sum(dim=1).sqrt().mean())

        if len(centers) < 2:
            scores.append(source_features.new_tensor(1.0))
            source_disc_scores.append(source_features.new_tensor(0.0))
            target_margin_scores.append(source_features.new_tensor(0.0))
            mismatch_scores.append(source_features.new_tensor(0.0))
            valid_flags.append(False)
            continue

        centers = torch.stack(centers, dim=0)
        compact = torch.stack(compactness).mean()
        center_distances = torch.cdist(centers, centers, p=2)
        center_distances.fill_diagonal_(float("inf"))
        separability = center_distances.min(dim=1).values.mean()
        source_disc = separability / compact.clamp_min(eps)

        target_counts = target_mask.sum(dim=1)
        valid_target = target_counts >= 1
        if not bool(valid_target.any().item()):
            target_margin = source_features.new_tensor(0.0)
            nearest_distance = separability
        else:
            target_phase_features = _masked_mean(target_features[valid_target], target_mask[valid_target], eps=eps)
            distances = torch.cdist(target_phase_features, centers, p=2)
            nearest = distances.topk(k=2, largest=False, dim=1).values
            margin = nearest[:, 1] - nearest[:, 0]
            target_margin = (margin / nearest[:, 1].clamp_min(eps)).mean()
            nearest_distance = nearest[:, 0].mean()

        mismatch_ratio = nearest_distance / separability.clamp_min(eps)
        explainability = torch.relu(target_margin) / (1.0 + mismatch_ratio)
        score = torch.relu(source_disc) * explainability

        scores.append(score)
        source_disc_scores.append(source_disc)
        target_margin_scores.append(target_margin)
        mismatch_scores.append(mismatch_ratio)
        valid_flags.append(True)

    scores = torch.stack(scores)
    if not any(valid_flags) or float(scores.max().detach().item()) <= eps:
        weights = torch.ones_like(scores)
        raw_weights = weights
        reliability = scores.new_tensor(0.0)
        gate_rho = scores.new_tensor(0.0)
    else:
        normalized = _minmax(scores, eps=eps)
        min_weight = max(0.0, min(1.0, float(min_weight)))
        raw_weights = min_weight + (1.0 - min_weight) * normalized
        reliability = scores.max()
        if reliability_gate:
            low = float(reliability_low)
            high = max(float(reliability_high), low + eps)
            gate_rho = ((reliability - low) / (high - low)).clamp(0.0, 1.0)
            weights = 1.0 + gate_rho * (raw_weights - 1.0)
        else:
            gate_rho = scores.new_tensor(1.0)
            weights = raw_weights

    logs = {
        "source_structure_mask_score_mean": float(scores.mean().detach().item()),
        "source_structure_mask_score_max": float(scores.max().detach().item()),
        "source_structure_mask_gate_rho": float(gate_rho.detach().item()),
        "source_structure_mask_reliability_gate": float(bool(reliability_gate)),
        "source_structure_mask_reliability_low": float(reliability_low),
        "source_structure_mask_reliability_high": float(reliability_high),
        "source_structure_mask_reliability": float(reliability.detach().item()),
        "source_structure_mask_source_disc_mean": float(torch.stack(source_disc_scores).mean().detach().item()),
        "source_structure_mask_target_margin_mean": float(torch.stack(target_margin_scores).mean().detach().item()),
        "source_structure_mask_mismatch_mean": float(torch.stack(mismatch_scores).mean().detach().item()),
    }
    for idx, weight in enumerate(weights):
        logs[f"source_structure_mask_weight_p{idx + 1}"] = float(weight.detach().item())
    for idx, weight in enumerate(raw_weights):
        logs[f"source_structure_mask_raw_weight_p{idx + 1}"] = float(weight.detach().item())
    return weights.detach(), logs


def compute_source_target_soft_support_mask(
    source_features,
    labels,
    target_features,
    source_positions=None,
    target_positions=None,
    min_weight=0.15,
    smooth_kernel_size=5,
    reliability_gate=True,
    reliability_low=5e-4,
    reliability_high=4e-2,
    eps=1e-6,
):
    """Estimate a time-point soft support mask for true location adaptivity.

    This is the v2.6.2e variant. Unlike the phase mask, it does not assume a
    fixed segment is the atomic support. It computes a white-box reliability
    score for every ordered time index and then smooths the resulting curve.
    """
    if source_features.ndim != 3 or target_features.ndim != 3:
        raise ValueError("Expected [B, T, D] source and target features")

    source_features, source_positions = _sort_temporal_features(source_features, source_positions)
    target_features, target_positions = _sort_temporal_features(target_features, target_positions)

    if target_features.shape[1] != source_features.shape[1]:
        min_len = min(source_features.shape[1], target_features.shape[1])
        source_features = source_features[:, :min_len]
        target_features = target_features[:, :min_len]
        if source_positions is not None:
            source_positions = source_positions[:, :min_len]
        if target_positions is not None:
            target_positions = target_positions[:, :min_len]

    _, sequence_length, _ = source_features.shape
    scores = []
    source_disc_scores = []
    target_margin_scores = []
    mismatch_scores = []
    valid_flags = []

    for time_idx in range(sequence_length):
        source_t = source_features[:, time_idx, :]
        target_t = target_features[:, time_idx, :]
        centers = []
        compactness = []
        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            if int(class_mask.sum().item()) < 2:
                continue
            class_features = source_t[class_mask]
            center = class_features.mean(dim=0)
            centers.append(center)
            compactness.append((class_features - center).pow(2).sum(dim=1).sqrt().mean())

        if len(centers) < 2:
            scores.append(source_features.new_tensor(0.0))
            source_disc_scores.append(source_features.new_tensor(0.0))
            target_margin_scores.append(source_features.new_tensor(0.0))
            mismatch_scores.append(source_features.new_tensor(0.0))
            valid_flags.append(False)
            continue

        centers = torch.stack(centers, dim=0)
        compact = torch.stack(compactness).mean()
        center_distances = torch.cdist(centers, centers, p=2)
        center_distances.fill_diagonal_(float("inf"))
        separability = center_distances.min(dim=1).values.mean()
        source_disc = separability / compact.clamp_min(eps)

        distances = torch.cdist(target_t, centers, p=2)
        nearest = distances.topk(k=2, largest=False, dim=1).values
        margin = nearest[:, 1] - nearest[:, 0]
        target_margin = (margin / nearest[:, 1].clamp_min(eps)).mean()
        nearest_distance = nearest[:, 0].mean()
        mismatch_ratio = nearest_distance / separability.clamp_min(eps)
        explainability = torch.relu(target_margin) / (1.0 + mismatch_ratio)
        score = torch.relu(source_disc) * explainability

        scores.append(score)
        source_disc_scores.append(source_disc)
        target_margin_scores.append(target_margin)
        mismatch_scores.append(mismatch_ratio)
        valid_flags.append(True)

    scores = torch.stack(scores)
    smoothed_scores = _smooth_1d(scores, kernel_size=smooth_kernel_size, eps=eps)
    if not any(valid_flags) or float(smoothed_scores.max().detach().item()) <= eps:
        weights = torch.ones_like(smoothed_scores)
        raw_weights = weights
        reliability = smoothed_scores.new_tensor(0.0)
        gate_rho = smoothed_scores.new_tensor(0.0)
    else:
        normalized = _minmax(smoothed_scores, eps=eps)
        min_weight = max(0.0, min(1.0, float(min_weight)))
        raw_weights = min_weight + (1.0 - min_weight) * normalized
        reliability = smoothed_scores.max()
        if reliability_gate:
            low = float(reliability_low)
            high = max(float(reliability_high), low + eps)
            gate_rho = ((reliability - low) / (high - low)).clamp(0.0, 1.0)
            weights = 1.0 + gate_rho * (raw_weights - 1.0)
        else:
            gate_rho = smoothed_scores.new_tensor(1.0)
            weights = raw_weights

    weight_diagnostics = _weight_diagnostics(weights, eps=eps)
    logs = {
        "source_structure_support_score_mean": float(scores.mean().detach().item()),
        "source_structure_support_score_max": float(scores.max().detach().item()),
        "source_structure_support_smooth_score_mean": float(smoothed_scores.mean().detach().item()),
        "source_structure_support_smooth_score_max": float(smoothed_scores.max().detach().item()),
        "source_structure_support_gate_rho": float(gate_rho.detach().item()),
        "source_structure_support_reliability": float(reliability.detach().item()),
        "source_structure_support_min": float(weights.min().detach().item()),
        "source_structure_support_max": float(weights.max().detach().item()),
        "source_structure_support_std": float(weight_diagnostics["std"].detach().item()),
        "source_structure_support_entropy": float(weight_diagnostics["entropy"].detach().item()),
        "source_structure_support_entropy_norm": float(weight_diagnostics["entropy_norm"].detach().item()),
        "source_structure_support_effective_size": float(weight_diagnostics["effective_size"].detach().item()),
        "source_structure_support_effective_ratio": float(weight_diagnostics["effective_ratio"].detach().item()),
        "source_structure_support_source_disc_mean": float(torch.stack(source_disc_scores).mean().detach().item()),
        "source_structure_support_target_margin_mean": float(torch.stack(target_margin_scores).mean().detach().item()),
        "source_structure_support_mismatch_mean": float(torch.stack(mismatch_scores).mean().detach().item()),
        "source_structure_support_smooth_kernel_size": float(max(1, int(smooth_kernel_size))),
        "source_structure_support_scores": _listify(scores),
        "source_structure_support_smooth_scores": _listify(smoothed_scores),
        "source_structure_support_raw_weights": _listify(raw_weights),
        "source_structure_support_weights": _listify(weights),
    }
    probe_count = min(12, weights.numel())
    if probe_count > 0:
        probe_indices = torch.linspace(0, weights.numel() - 1, steps=probe_count, device=weights.device).round().long()
        for probe_idx, time_idx in enumerate(probe_indices):
            logs[f"source_structure_support_weight_t{probe_idx + 1}"] = float(weights[time_idx].detach().item())
            logs[f"source_structure_support_raw_weight_t{probe_idx + 1}"] = float(raw_weights[time_idx].detach().item())
            logs[f"source_structure_support_index_t{probe_idx + 1}"] = float(time_idx.detach().item())
    return weights.detach(), logs
