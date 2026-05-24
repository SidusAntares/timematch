import torch
import torch.nn.functional as F


def _smooth_index_moving_average(curves, kernel_size=5):
    if curves.ndim != 3:
        raise ValueError(f"Expected [B, T, D] curves, got {tuple(curves.shape)}")
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or curves.shape[1] <= 1:
        return curves
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    x = curves.transpose(1, 2)
    x = F.pad(x, (pad, pad), mode="replicate")
    trend = F.avg_pool1d(x, kernel_size=kernel_size, stride=1)
    return trend.transpose(1, 2)


def _prepare_temporal_positions(curves, positions=None):
    if positions is None:
        return torch.arange(curves.shape[1], device=curves.device, dtype=curves.dtype).view(1, -1).expand(
            curves.shape[0],
            -1,
        )
    if positions.ndim == 1:
        positions = positions.view(1, -1).expand(curves.shape[0], -1)
    if positions.shape[:2] != curves.shape[:2]:
        raise ValueError(
            f"Expected positions to have shape [B, T] or [T], got {tuple(positions.shape)} "
            f"for curves {tuple(curves.shape)}"
        )
    return positions.to(device=curves.device, dtype=curves.dtype)


def infer_time_smoothing_bandwidth(positions, kernel_size=5, eps=1e-6):
    kernel_size = max(1, int(kernel_size))
    if positions is None or positions.shape[-1] <= 1:
        return 1.0
    gaps = (positions[:, 1:] - positions[:, :-1]).abs()
    positive_gaps = gaps[gaps > eps]
    if positive_gaps.numel() == 0:
        return 1.0
    half_window = max(1.0, float(kernel_size - 1) / 2.0)
    return float(positive_gaps.median().detach().item()) * half_window


def _time_kernel_weights(distances, bandwidth, kernel="gaussian"):
    scaled = distances / bandwidth.clamp_min(1e-6)
    kernel = str(kernel or "gaussian").lower()
    if kernel in {"gaussian", "rbf"}:
        return torch.exp(-0.5 * scaled.pow(2))
    if kernel in {"boxcar", "uniform"}:
        return (scaled <= 1.0).to(dtype=distances.dtype)
    if kernel in {"triangular", "linear"}:
        return torch.relu(1.0 - scaled)
    raise ValueError(f"Unsupported trend smoothing kernel: {kernel}")


def smooth_temporal_trend(
    curves,
    positions=None,
    kernel_size=5,
    mode="time",
    bandwidth=0.0,
    kernel="gaussian",
    eps=1e-6,
):
    if curves.ndim != 3:
        raise ValueError(f"Expected [B, T, D] curves, got {tuple(curves.shape)}")
    if curves.shape[1] <= 1:
        return curves

    mode = str(mode or "time").lower()
    if mode in {"index", "moving_average", "ma"}:
        return _smooth_index_moving_average(curves, kernel_size=kernel_size)
    if mode not in {"time", "time_aware", "kernel"}:
        raise ValueError(f"Unsupported trend smoothing mode: {mode}")

    temporal_positions = _prepare_temporal_positions(curves, positions)
    distances = (temporal_positions.unsqueeze(2) - temporal_positions.unsqueeze(1)).abs()
    if float(bandwidth) > 0.0:
        bandwidth_tensor = curves.new_tensor(float(bandwidth))
    else:
        bandwidth_tensor = curves.new_tensor(
            infer_time_smoothing_bandwidth(temporal_positions, kernel_size=kernel_size, eps=eps)
        )
    weights = _time_kernel_weights(distances, bandwidth_tensor, kernel=kernel)
    weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(eps)
    return torch.bmm(weights, curves)


def split_trend_residual(
    curves,
    positions=None,
    kernel_size=5,
    mode="time",
    bandwidth=0.0,
    kernel="gaussian",
):
    trend = smooth_temporal_trend(
        curves,
        positions=positions,
        kernel_size=kernel_size,
        mode=mode,
        bandwidth=bandwidth,
        kernel=kernel,
    )
    residual = curves - trend
    return trend, residual


def classwise_curve_variance(curves, labels, eps=1e-6):
    zero = curves.sum() * 0.0
    loss = zero
    class_count = 0
    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        if int(class_mask.sum().item()) < 2:
            continue
        class_curves = curves[class_mask]
        center = class_curves.mean(dim=0, keepdim=True)
        loss = loss + (class_curves - center).pow(2).sum(dim=2).mean()
        class_count += 1
    if class_count > 0:
        loss = loss / (class_count + eps)
    return loss, class_count


def bounded_residual_energy(residual, margin=1.0):
    energy = residual.pow(2).sum(dim=2).mean()
    margin_tensor = residual.new_tensor(float(margin))
    loss = torch.relu(energy - margin_tensor).pow(2)
    return loss, energy


def temporal_differences(curves, positions=None):
    diffs = curves[:, 1:] - curves[:, :-1]
    if positions is None:
        return diffs
    delta_t = (positions[:, 1:] - positions[:, :-1]).abs().clamp_min(1.0)
    return diffs / delta_t.to(dtype=curves.dtype).unsqueeze(-1)


def classwise_dynamics_consistency(curves, labels, positions=None, mode="cosine", eps=1e-6):
    zero = curves.sum() * 0.0
    loss = zero
    class_count = 0
    if curves.shape[1] < 2:
        return loss, class_count

    dynamics = temporal_differences(curves, positions=positions)
    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        if int(class_mask.sum().item()) < 2:
            continue
        class_dyn = dynamics[class_mask]
        proto_dyn = class_dyn.mean(dim=0, keepdim=True)
        mode = str(mode or "cosine").lower()
        if mode == "mse":
            class_loss = (class_dyn - proto_dyn).pow(2).sum(dim=2).mean()
        elif mode in {"cosine", "direction"}:
            proto_expanded = proto_dyn.expand_as(class_dyn)
            sample_norm = class_dyn.norm(dim=2)
            proto_norm = proto_expanded.norm(dim=2)
            valid = (sample_norm > eps) & (proto_norm > eps)
            cosine_loss = 1.0 - F.cosine_similarity(class_dyn, proto_expanded, dim=2, eps=eps)
            class_loss = cosine_loss[valid].mean() if bool(valid.any().item()) else zero
        else:
            raise ValueError(f"Unsupported dynamics mode: {mode}")
        loss = loss + class_loss
        class_count += 1

    if class_count > 0:
        loss = loss / (class_count + eps)
    return loss, class_count
