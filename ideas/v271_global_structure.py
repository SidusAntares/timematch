import torch
import torch.nn.functional as F

from ideas.v271_decomposition import (
    bounded_residual_energy,
    classwise_curve_variance,
    classwise_dynamics_consistency,
    infer_time_smoothing_bandwidth,
    split_trend_residual,
    temporal_differences,
)


def _mean_time_grid(positions):
    if positions is None:
        return None
    if positions.ndim == 1:
        return positions
    return positions.to(dtype=torch.float32).median(dim=0).values.to(device=positions.device, dtype=positions.dtype)


def _class_prototypes(curves, labels):
    prototypes = []
    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        if bool(class_mask.any().item()):
            prototypes.append(curves[class_mask].mean(dim=0))
    if not prototypes:
        return None
    return torch.stack(prototypes, dim=0)


def _event_centers_from_source_trend(trend, positions, labels, event_count=2, random_centers=False, eps=1e-6):
    if trend.shape[1] < 2:
        return trend.new_empty((0,))

    time_grid = _mean_time_grid(positions)
    if time_grid is None:
        time_grid = torch.arange(trend.shape[1], device=trend.device, dtype=trend.dtype)
    time_grid = time_grid.to(device=trend.device, dtype=trend.dtype)
    midpoint_grid = 0.5 * (time_grid[1:] + time_grid[:-1])

    count = max(0, min(int(event_count), int(midpoint_grid.numel())))
    if count <= 0:
        return trend.new_empty((0,))

    if random_centers:
        start = time_grid.min()
        span = (time_grid.max() - start).clamp_min(eps)
        # Deterministic matched-random centers: continuous in time, independent of labels/features.
        idx = torch.arange(count, device=trend.device, dtype=trend.dtype)
        fractions = torch.remainder(0.137 + 0.61803398875 * (idx + 1.0), 1.0)
        return start + fractions * span

    prototypes = _class_prototypes(trend.detach(), labels)
    if prototypes is None or prototypes.shape[0] == 0:
        return trend.new_empty((0,))

    proto_diffs = prototypes[:, 1:] - prototypes[:, :-1]
    delta_t = (time_grid[1:] - time_grid[:-1]).abs().clamp_min(1.0)
    proto_diffs = proto_diffs / delta_t.view(1, -1, 1)
    event_scores = proto_diffs.norm(dim=2).mean(dim=0)
    if event_scores.numel() == 0:
        return trend.new_empty((0,))
    topk = torch.topk(event_scores, k=count, largest=True).indices.sort().values
    return midpoint_grid[topk]


def _gaussian_support_weights(positions, centers, sigma, eps=1e-6):
    if centers.numel() == 0:
        return None
    sigma = positions.new_tensor(float(sigma)).clamp_min(eps)
    distances = positions.unsqueeze(0) - centers.view(-1, 1, 1)
    weights = torch.exp(-0.5 * (distances / sigma).pow(2))
    return weights


def _interpolate_curve_at(curve, grid, query, eps=1e-6):
    if curve.ndim != 2:
        raise ValueError(f"Expected curve [T, D], got {tuple(curve.shape)}")
    if grid.ndim != 1:
        raise ValueError(f"Expected grid [T], got {tuple(grid.shape)}")
    if curve.shape[0] != grid.shape[0]:
        raise ValueError(f"Curve/grid length mismatch: {tuple(curve.shape)} vs {tuple(grid.shape)}")
    if curve.shape[0] == 1:
        return curve[:1].expand(*query.shape, curve.shape[-1])

    grid = grid.to(device=curve.device, dtype=curve.dtype).contiguous()
    query = query.to(device=curve.device, dtype=curve.dtype).clamp(min=grid[0], max=grid[-1])
    right = torch.searchsorted(grid, query.contiguous(), right=False).clamp(min=1, max=grid.numel() - 1)
    left = right - 1
    left_t = grid[left]
    right_t = grid[right]
    alpha = ((query - left_t) / (right_t - left_t).clamp_min(eps)).clamp(0.0, 1.0)
    left_value = curve[left]
    right_value = curve[right]
    return left_value * (1.0 - alpha.unsqueeze(-1)) + right_value * alpha.unsqueeze(-1)


def _shift_bank_from_positions(positions, radius_steps=2.0, shift_count=5, eps=1e-6):
    shift_count = max(1, int(shift_count))
    radius_steps = max(0.0, float(radius_steps))
    if positions.shape[-1] <= 1 or radius_steps <= 0.0 or shift_count <= 1:
        return positions.new_zeros((1,)), 0.0
    gaps = (positions[:, 1:] - positions[:, :-1]).abs()
    positive_gaps = gaps[gaps > eps]
    median_gap = float(positive_gaps.median().detach().item()) if positive_gaps.numel() > 0 else 1.0
    radius = radius_steps * median_gap
    shifts = torch.linspace(-radius, radius, steps=shift_count, device=positions.device, dtype=positions.dtype)
    return shifts, radius


def _soft_shift_distance(distances, shifts, temperature=0.05):
    if distances.shape[-1] == 1 or float(temperature) <= 0.0:
        values, indices = distances.min(dim=-1)
        selected = shifts[indices]
        return values, selected.abs().mean()
    temp = distances.new_tensor(float(temperature)).clamp_min(1e-6)
    weights = torch.softmax(-distances / temp, dim=-1)
    values = (weights * distances).sum(dim=-1)
    selected = (weights * shifts.view(1, -1)).sum(dim=-1)
    return values, selected.abs().mean()


def classwise_warp_tolerant_curve_variance(
    curves,
    labels,
    positions=None,
    shift_radius_steps=2.0,
    shift_count=5,
    temperature=0.05,
    eps=1e-6,
):
    zero = curves.sum() * 0.0
    loss = zero
    class_count = 0
    shift_sum = zero
    time_grid = _mean_time_grid(positions)
    if time_grid is None:
        time_grid = torch.arange(curves.shape[1], device=curves.device, dtype=curves.dtype)
        positions = time_grid.view(1, -1).expand(curves.shape[0], -1)
    positions = positions.to(device=curves.device, dtype=curves.dtype)
    time_grid = time_grid.to(device=curves.device, dtype=curves.dtype)
    shifts, radius = _shift_bank_from_positions(
        positions,
        radius_steps=shift_radius_steps,
        shift_count=shift_count,
        eps=eps,
    )

    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        if int(class_mask.sum().item()) < 2:
            continue
        class_curves = curves[class_mask]
        class_positions = positions[class_mask]
        prototype = class_curves.mean(dim=0)
        distances = []
        for shift in shifts:
            shifted_proto = _interpolate_curve_at(prototype, time_grid, class_positions + shift, eps=eps)
            distances.append((class_curves - shifted_proto).pow(2).sum(dim=2).mean(dim=1))
        distance_bank = torch.stack(distances, dim=1)
        class_loss, class_shift = _soft_shift_distance(distance_bank, shifts, temperature=temperature)
        loss = loss + class_loss.mean()
        shift_sum = shift_sum + class_shift
        class_count += 1

    if class_count > 0:
        loss = loss / (class_count + eps)
        shift_sum = shift_sum / (class_count + eps)
    return loss, class_count, shift_sum, float(radius)


def classwise_warp_tolerant_dynamics_consistency(
    curves,
    labels,
    positions=None,
    mode="cosine",
    shift_radius_steps=2.0,
    shift_count=5,
    temperature=0.05,
    eps=1e-6,
):
    zero = curves.sum() * 0.0
    if curves.shape[1] < 2:
        return zero, 0, zero, 0.0
    if positions is None:
        positions = torch.arange(curves.shape[1], device=curves.device, dtype=curves.dtype).view(1, -1).expand(
            curves.shape[0],
            -1,
        )
    positions = positions.to(device=curves.device, dtype=curves.dtype)
    dynamics = temporal_differences(curves, positions=positions)
    midpoint_positions = 0.5 * (positions[:, 1:] + positions[:, :-1])
    midpoint_grid = _mean_time_grid(midpoint_positions).to(device=curves.device, dtype=curves.dtype)
    shifts, radius = _shift_bank_from_positions(
        midpoint_positions,
        radius_steps=shift_radius_steps,
        shift_count=shift_count,
        eps=eps,
    )

    loss = zero
    class_count = 0
    shift_sum = zero
    mode = str(mode or "cosine").lower()
    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        if int(class_mask.sum().item()) < 2:
            continue
        class_dyn = dynamics[class_mask]
        class_midpoints = midpoint_positions[class_mask]
        prototype = class_dyn.mean(dim=0)
        distances = []
        for shift in shifts:
            shifted_proto = _interpolate_curve_at(prototype, midpoint_grid, class_midpoints + shift, eps=eps)
            if mode == "mse":
                distance = (class_dyn - shifted_proto).pow(2).sum(dim=2).mean(dim=1)
            elif mode in {"cosine", "direction"}:
                distance = 1.0 - F.cosine_similarity(class_dyn, shifted_proto, dim=2, eps=eps)
                distance = distance.mean(dim=1)
            else:
                raise ValueError(f"Unsupported dynamics mode: {mode}")
            distances.append(distance)
        distance_bank = torch.stack(distances, dim=1)
        class_loss, class_shift = _soft_shift_distance(distance_bank, shifts, temperature=temperature)
        loss = loss + class_loss.mean()
        shift_sum = shift_sum + class_shift
        class_count += 1

    if class_count > 0:
        loss = loss / (class_count + eps)
        shift_sum = shift_sum / (class_count + eps)
    return loss, class_count, shift_sum, float(radius)


def _weighted_curve_variance(curves, labels, support_weights, eps=1e-6):
    zero = curves.sum() * 0.0
    if support_weights is None:
        return zero, 0.0
    total = zero
    valid_terms = 0
    for support_idx in range(support_weights.shape[0]):
        weights = support_weights[support_idx].to(dtype=curves.dtype)
        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            if int(class_mask.sum().item()) < 2:
                continue
            class_curves = curves[class_mask]
            class_weights = weights[class_mask].unsqueeze(-1)
            weight_sum = class_weights.sum(dim=1, keepdim=True).clamp_min(eps)
            proto = (class_curves * class_weights).sum(dim=0) / class_weights.sum(dim=0).clamp_min(eps)
            proto = proto.unsqueeze(0)
            point_loss = (class_curves - proto).pow(2).sum(dim=2, keepdim=True)
            sample_loss = (point_loss * class_weights).sum(dim=1) / weight_sum.squeeze(1)
            total = total + sample_loss.mean()
            valid_terms += 1
    if valid_terms > 0:
        total = total / (valid_terms + eps)
    return total, float(valid_terms)


def _weighted_dynamics_consistency(curves, labels, positions, support_centers, sigma, mode="cosine", eps=1e-6):
    zero = curves.sum() * 0.0
    if support_centers.numel() == 0 or curves.shape[1] < 2:
        return zero, 0.0
    dynamics = temporal_differences(curves, positions=positions)
    midpoint_positions = 0.5 * (positions[:, 1:] + positions[:, :-1])
    weights = _gaussian_support_weights(midpoint_positions, support_centers, sigma, eps=eps)
    if weights is None:
        return zero, 0.0

    total = zero
    valid_terms = 0
    mode = str(mode or "cosine").lower()
    for support_idx in range(weights.shape[0]):
        support_weights = weights[support_idx].to(dtype=curves.dtype)
        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            if int(class_mask.sum().item()) < 2:
                continue
            class_dyn = dynamics[class_mask]
            class_weights = support_weights[class_mask].unsqueeze(-1)
            proto_dyn = (class_dyn * class_weights).sum(dim=0) / class_weights.sum(dim=0).clamp_min(eps)
            proto_dyn = proto_dyn.unsqueeze(0)
            proto_expanded = proto_dyn.expand_as(class_dyn)
            if mode == "mse":
                point_loss = (class_dyn - proto_expanded).pow(2).sum(dim=2, keepdim=True)
            elif mode in {"cosine", "direction"}:
                cosine_loss = 1.0 - F.cosine_similarity(class_dyn, proto_expanded, dim=2, eps=eps)
                point_loss = cosine_loss.unsqueeze(-1)
            else:
                raise ValueError(f"Unsupported dynamics mode: {mode}")
            weight_sum = class_weights.sum(dim=1, keepdim=True).clamp_min(eps)
            sample_loss = (point_loss * class_weights).sum(dim=1) / weight_sum.squeeze(1)
            total = total + sample_loss.mean()
            valid_terms += 1
    if valid_terms > 0:
        total = total / (valid_terms + eps)
    return total, float(valid_terms)


def compute_v271_global_structure_loss(
    ordered_feats,
    ordered_positions,
    labels,
    trend_kernel_size=5,
    trend_smoothing_mode="time",
    trend_bandwidth=0.0,
    trend_kernel="gaussian",
    trend_cohesion_trade_off=1.0,
    trend_dynamics_trade_off=0.05,
    residual_variance_trade_off=0.10,
    residual_energy_trade_off=0.05,
    residual_energy_margin=1.0,
    dynamics_mode="cosine",
):
    trend, residual = split_trend_residual(
        ordered_feats,
        positions=ordered_positions,
        kernel_size=trend_kernel_size,
        mode=trend_smoothing_mode,
        bandwidth=trend_bandwidth,
        kernel=trend_kernel,
    )
    trend_cohesion, trend_class_count = classwise_curve_variance(trend, labels)
    trend_dynamics, dynamics_class_count = classwise_dynamics_consistency(
        trend,
        labels,
        positions=ordered_positions,
        mode=dynamics_mode,
    )
    residual_variance, residual_class_count = classwise_curve_variance(residual, labels)
    residual_energy_loss, residual_energy = bounded_residual_energy(
        residual,
        margin=residual_energy_margin,
    )

    total = (
        float(trend_cohesion_trade_off) * trend_cohesion
        + float(trend_dynamics_trade_off) * trend_dynamics
        + float(residual_variance_trade_off) * residual_variance
        + float(residual_energy_trade_off) * residual_energy_loss
    )
    logs = {
        "v271_global_trend_cohesion_loss": trend_cohesion,
        "v271_global_trend_dynamics_loss": trend_dynamics,
        "v271_global_residual_variance_loss": residual_variance,
        "v271_global_residual_energy_loss": residual_energy_loss,
        "v271_global_residual_energy": residual_energy,
        "v271_global_trend_class_count": float(trend_class_count),
        "v271_global_dynamics_class_count": float(dynamics_class_count),
        "v271_global_residual_class_count": float(residual_class_count),
        "v271_global_trend_kernel_size": float(trend_kernel_size),
        "v271_global_trend_smoothing_mode": str(trend_smoothing_mode),
        "v271_global_trend_bandwidth": float(trend_bandwidth),
        "v271_global_trend_bandwidth_effective": float(trend_bandwidth)
        if float(trend_bandwidth) > 0.0
        else infer_time_smoothing_bandwidth(ordered_positions, kernel_size=trend_kernel_size),
        "v271_global_trend_kernel": str(trend_kernel),
        "v271_global_residual_energy_margin": float(residual_energy_margin),
        "v271_global_trend_cohesion_trade_off": float(trend_cohesion_trade_off),
        "v271_global_trend_dynamics_trade_off": float(trend_dynamics_trade_off),
        "v271_global_residual_variance_trade_off": float(residual_variance_trade_off),
        "v271_global_residual_energy_trade_off": float(residual_energy_trade_off),
    }
    return total, logs


def compute_v271_global_gtw_structure_loss(
    ordered_feats,
    ordered_positions,
    labels,
    trend_kernel_size=5,
    trend_smoothing_mode="time",
    trend_bandwidth=0.0,
    trend_kernel="gaussian",
    trend_cohesion_trade_off=1.0,
    trend_dynamics_trade_off=0.05,
    residual_variance_trade_off=0.10,
    residual_energy_trade_off=0.05,
    residual_energy_margin=1.0,
    dynamics_mode="cosine",
    gtw_shift_radius_steps=2.0,
    gtw_shift_count=5,
    gtw_temperature=0.05,
):
    trend, residual = split_trend_residual(
        ordered_feats,
        positions=ordered_positions,
        kernel_size=trend_kernel_size,
        mode=trend_smoothing_mode,
        bandwidth=trend_bandwidth,
        kernel=trend_kernel,
    )
    trend_cohesion, trend_class_count, trend_shift, trend_shift_radius = classwise_warp_tolerant_curve_variance(
        trend,
        labels,
        positions=ordered_positions,
        shift_radius_steps=gtw_shift_radius_steps,
        shift_count=gtw_shift_count,
        temperature=gtw_temperature,
    )
    trend_dynamics, dynamics_class_count, dynamics_shift, dynamics_shift_radius = (
        classwise_warp_tolerant_dynamics_consistency(
            trend,
            labels,
            positions=ordered_positions,
            mode=dynamics_mode,
            shift_radius_steps=gtw_shift_radius_steps,
            shift_count=gtw_shift_count,
            temperature=gtw_temperature,
        )
    )
    residual_variance, residual_class_count = classwise_curve_variance(residual, labels)
    residual_energy_loss, residual_energy = bounded_residual_energy(
        residual,
        margin=residual_energy_margin,
    )

    total = (
        float(trend_cohesion_trade_off) * trend_cohesion
        + float(trend_dynamics_trade_off) * trend_dynamics
        + float(residual_variance_trade_off) * residual_variance
        + float(residual_energy_trade_off) * residual_energy_loss
    )
    logs = {
        "v271_gtw_trend_cohesion_loss": trend_cohesion,
        "v271_gtw_trend_dynamics_loss": trend_dynamics,
        "v271_gtw_residual_variance_loss": residual_variance,
        "v271_gtw_residual_energy_loss": residual_energy_loss,
        "v271_gtw_residual_energy": residual_energy,
        "v271_gtw_trend_class_count": float(trend_class_count),
        "v271_gtw_dynamics_class_count": float(dynamics_class_count),
        "v271_gtw_residual_class_count": float(residual_class_count),
        "v271_gtw_trend_abs_shift": trend_shift,
        "v271_gtw_dynamics_abs_shift": dynamics_shift,
        "v271_gtw_shift_radius": float(max(trend_shift_radius, dynamics_shift_radius)),
        "v271_gtw_shift_radius_steps": float(gtw_shift_radius_steps),
        "v271_gtw_shift_count": float(gtw_shift_count),
        "v271_gtw_temperature": float(gtw_temperature),
        "v271_gtw_trend_kernel_size": float(trend_kernel_size),
        "v271_gtw_trend_smoothing_mode": str(trend_smoothing_mode),
        "v271_gtw_trend_bandwidth": float(trend_bandwidth),
        "v271_gtw_trend_bandwidth_effective": float(trend_bandwidth)
        if float(trend_bandwidth) > 0.0
        else infer_time_smoothing_bandwidth(ordered_positions, kernel_size=trend_kernel_size),
        "v271_gtw_trend_kernel": str(trend_kernel),
        "v271_gtw_residual_energy_margin": float(residual_energy_margin),
        "v271_gtw_trend_cohesion_trade_off": float(trend_cohesion_trade_off),
        "v271_gtw_trend_dynamics_trade_off": float(trend_dynamics_trade_off),
        "v271_gtw_residual_variance_trade_off": float(residual_variance_trade_off),
        "v271_gtw_residual_energy_trade_off": float(residual_energy_trade_off),
    }
    return total, logs


def compute_v271_event_support_structure_loss(
    ordered_feats,
    ordered_positions,
    labels,
    trend_kernel_size=5,
    trend_smoothing_mode="time",
    trend_bandwidth=0.0,
    trend_kernel="gaussian",
    trend_cohesion_trade_off=1.0,
    trend_dynamics_trade_off=0.05,
    residual_variance_trade_off=0.10,
    residual_energy_trade_off=0.05,
    residual_energy_margin=1.0,
    dynamics_mode="cosine",
    event_count=2,
    support_sigma_ratio=0.20,
    random_centers=False,
    local_trade_off=1.0,
    eps=1e-6,
):
    global_loss, global_logs = compute_v271_global_structure_loss(
        ordered_feats,
        ordered_positions,
        labels,
        trend_kernel_size=trend_kernel_size,
        trend_smoothing_mode=trend_smoothing_mode,
        trend_bandwidth=trend_bandwidth,
        trend_kernel=trend_kernel,
        trend_cohesion_trade_off=trend_cohesion_trade_off,
        trend_dynamics_trade_off=trend_dynamics_trade_off,
        residual_variance_trade_off=residual_variance_trade_off,
        residual_energy_trade_off=residual_energy_trade_off,
        residual_energy_margin=residual_energy_margin,
        dynamics_mode=dynamics_mode,
    )
    trend, residual = split_trend_residual(
        ordered_feats,
        positions=ordered_positions,
        kernel_size=trend_kernel_size,
        mode=trend_smoothing_mode,
        bandwidth=trend_bandwidth,
        kernel=trend_kernel,
    )
    centers = _event_centers_from_source_trend(
        trend,
        ordered_positions,
        labels,
        event_count=event_count,
        random_centers=random_centers,
        eps=eps,
    )
    time_grid = _mean_time_grid(ordered_positions).to(device=ordered_feats.device, dtype=ordered_feats.dtype)
    time_span = (time_grid.max() - time_grid.min()).abs().clamp_min(1.0)
    sigma = float(support_sigma_ratio) * float(time_span.detach().item())
    if sigma <= 0.0:
        sigma = infer_time_smoothing_bandwidth(ordered_positions, kernel_size=trend_kernel_size, eps=eps)
    support_weights = _gaussian_support_weights(ordered_positions, centers, sigma, eps=eps)

    local_trend_cohesion, trend_terms = _weighted_curve_variance(trend, labels, support_weights, eps=eps)
    local_trend_dynamics, dynamics_terms = _weighted_dynamics_consistency(
        trend,
        labels,
        ordered_positions,
        centers,
        sigma,
        mode=dynamics_mode,
        eps=eps,
    )
    local_residual_variance, residual_terms = _weighted_curve_variance(residual, labels, support_weights, eps=eps)
    local_residual_energy_loss, local_residual_energy = bounded_residual_energy(
        residual * support_weights.mean(dim=0).unsqueeze(-1) if support_weights is not None else residual,
        margin=residual_energy_margin,
    )

    local_loss = (
        float(trend_cohesion_trade_off) * local_trend_cohesion
        + float(trend_dynamics_trade_off) * local_trend_dynamics
        + float(residual_variance_trade_off) * local_residual_variance
        + float(residual_energy_trade_off) * local_residual_energy_loss
    )
    total = global_loss + float(local_trade_off) * local_loss
    logs = dict(global_logs)
    logs.update(
        {
            "v271_event_local_loss": local_loss,
            "v271_event_total_loss": total,
            "v271_event_trend_cohesion_loss": local_trend_cohesion,
            "v271_event_trend_dynamics_loss": local_trend_dynamics,
            "v271_event_residual_variance_loss": local_residual_variance,
            "v271_event_residual_energy_loss": local_residual_energy_loss,
            "v271_event_residual_energy": local_residual_energy,
            "v271_event_count": float(centers.numel()),
            "v271_event_support_sigma": float(sigma),
            "v271_event_support_sigma_ratio": float(support_sigma_ratio),
            "v271_event_local_trade_off": float(local_trade_off),
            "v271_event_random_centers": 1.0 if random_centers else 0.0,
            "v271_event_trend_terms": float(trend_terms),
            "v271_event_dynamics_terms": float(dynamics_terms),
            "v271_event_residual_terms": float(residual_terms),
        }
    )
    for idx in range(min(int(centers.numel()), 4)):
        logs[f"v271_event_center_{idx + 1}"] = float(centers[idx].detach().item())
    return total, logs
