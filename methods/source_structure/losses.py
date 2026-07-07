import math

import torch
import torch.nn.functional as F


RAW_GLOBAL_COMPACTNESS_VERSIONS = {
    "v275_raw_global_compactness",
    "raw_global_compactness",
    "source_raw_global_compactness",
    "v276_raw_timepoint_compactness",
    "raw_timepoint_compactness",
    "source_raw_timepoint_compactness",
    "v276_raw_smoothed_timepoint_compactness",
    "raw_smoothed_timepoint_compactness",
    "source_raw_smoothed_timepoint_compactness",
    "v303_time_permuted_smoothed_timepoint_compactness",
    "time_permuted_smoothed_timepoint_compactness",
    "source_time_permuted_smoothed_timepoint_compactness",
    "v276_raw_trimmed_global_compactness",
    "raw_trimmed_global_compactness",
    "source_raw_trimmed_global_compactness",
    "v277_raw_lowfreq_dct_k2_compactness",
    "v277_raw_lowfreq_dct_k4_compactness",
    "v277_raw_lowfreq_dct_k8_compactness",
    "v283a_umsc_dual_075_025_compactness",
    "v283a_umsc_dual_050_050_compactness",
    "v283b_umsc_triscale_060_020_020_compactness",
    "v284_elastic_smoothed_timepoint_compactness",
}


def is_raw_global_compactness_version(version):
    return str(version or "").lower() in RAW_GLOBAL_COMPACTNESS_VERSIONS


def _compactness_distance(class_feats, class_center, mode="mse", eps=1e-6):
    mode = str(mode or "mse").lower()
    if mode in {"mse", "euclidean", "l2"}:
        return (class_feats - class_center).pow(2).sum(dim=1).mean()
    if mode in {"normalized_mse", "l2_normalized_mse", "unit_mse"}:
        normalized_feats = F.normalize(class_feats, dim=1, eps=eps)
        normalized_center = F.normalize(class_center, dim=1, eps=eps)
        return (normalized_feats - normalized_center).pow(2).sum(dim=1).mean()
    raise ValueError(f"Unsupported raw compactness distance: {mode}")


def _class_norm_preserve_loss(class_feats, target="min_mean", value=1.0, eps=1e-6):
    target = str(target or "batch_mean").lower()
    norms = class_feats.norm(dim=1)
    if target in {"none", "off", "disabled"}:
        return class_feats.sum() * 0.0
    if target in {"batch_mean", "class_mean", "detached_mean"}:
        anchor = norms.detach().mean().clamp_min(eps)
        return ((norms - anchor) / anchor).pow(2).mean()
    if target in {"min_mean", "floor", "hinge"}:
        anchor = norms.new_tensor(float(value)).clamp_min(eps)
        return torch.relu(anchor - norms.mean()).div(anchor).pow(2)
    if target in {"fixed", "constant"}:
        anchor = norms.new_tensor(float(value)).clamp_min(eps)
        return ((norms - anchor) / anchor).pow(2).mean()
    raise ValueError(f"Unsupported raw compactness norm preserve target: {target}")


def _trimmed_mean(class_feats, trim_ratio=0.10):
    if class_feats.shape[0] < 3 or float(trim_ratio) <= 0.0:
        return class_feats.mean(dim=0, keepdim=True)
    trim_count = int(class_feats.shape[0] * float(trim_ratio))
    if trim_count <= 0 or class_feats.shape[0] - 2 * trim_count < 1:
        return class_feats.mean(dim=0, keepdim=True)
    sorted_feats, _ = class_feats.sort(dim=0)
    return sorted_feats[trim_count:-trim_count].mean(dim=0, keepdim=True)


def _compute_global_compactness(
    pooled_feats,
    labels,
    compact_distance="mse",
    center_mode="mean",
    eps=1e-6,
):
    zero = pooled_feats.sum() * 0.0
    compact_loss = zero
    valid_class_count = 0
    valid_sample_count = 0

    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        class_feats = pooled_feats[class_mask]
        if class_feats.shape[0] < 2:
            continue
        if str(center_mode).lower() == "trimmed":
            class_center = _trimmed_mean(class_feats)
        else:
            class_center = class_feats.mean(dim=0, keepdim=True)
        compact_loss = compact_loss + _compactness_distance(
            class_feats,
            class_center,
            mode=compact_distance,
            eps=eps,
        )
        valid_class_count += 1
        valid_sample_count += int(class_feats.shape[0])

    if valid_class_count > 0:
        compact_loss = compact_loss / (valid_class_count + eps)
    return compact_loss, valid_class_count, valid_sample_count


def _compute_timepoint_compactness(spatial_feats, labels, compact_distance="mse", eps=1e-6):
    zero = spatial_feats.sum() * 0.0
    compact_loss = zero
    valid_class_count = 0
    valid_sample_count = 0

    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        class_feats = spatial_feats[class_mask]
        if class_feats.shape[0] < 2:
            continue
        class_center = class_feats.mean(dim=0, keepdim=True)
        if str(compact_distance or "mse").lower() in {
            "normalized_mse",
            "l2_normalized_mse",
            "unit_mse",
        }:
            class_feats_cmp = F.normalize(class_feats, dim=2, eps=eps)
            class_center_cmp = F.normalize(class_center, dim=2, eps=eps)
            compact_loss = compact_loss + (class_feats_cmp - class_center_cmp).pow(2).sum(dim=2).mean()
        else:
            compact_loss = compact_loss + (class_feats - class_center).pow(2).sum(dim=2).mean()
        valid_class_count += 1
        valid_sample_count += int(class_feats.shape[0])

    if valid_class_count > 0:
        compact_loss = compact_loss / (valid_class_count + eps)
    return compact_loss, valid_class_count, valid_sample_count


def _dct_lowfreq_basis(time_steps, components, device, dtype):
    components = max(1, min(int(components), int(time_steps)))
    time_index = torch.arange(time_steps, device=device, dtype=dtype)
    basis_rows = [torch.ones(time_steps, device=device, dtype=dtype) / float(time_steps)]
    if components > 1:
        scale = (2.0 ** 0.5) / float(time_steps)
        for freq in range(1, components):
            row = scale * torch.cos(
                math.pi * (time_index + 0.5) * float(freq) / float(time_steps)
            )
            basis_rows.append(row)
    return torch.stack(basis_rows, dim=0)


def _lowfreq_dct_components(version):
    version = str(version or "").lower()
    if "lowfreq_dct_k2" in version:
        return 2
    if "lowfreq_dct_k4" in version:
        return 4
    if "lowfreq_dct_k8" in version:
        return 8
    return None


def _umsc_weights(version):
    version = str(version or "").lower()
    if version == "v283a_umsc_dual_075_025_compactness":
        return {"l3": 0.75, "l5": 0.0, "linf": 0.25}
    if version == "v283a_umsc_dual_050_050_compactness":
        return {"l3": 0.50, "l5": 0.0, "linf": 0.50}
    if version == "v283b_umsc_triscale_060_020_020_compactness":
        return {"l3": 0.60, "l5": 0.20, "linf": 0.20}
    return None


def _compute_lowfreq_compactness(spatial_feats, labels, components, compact_distance="mse", eps=1e-6):
    basis = _dct_lowfreq_basis(
        spatial_feats.shape[1],
        components,
        device=spatial_feats.device,
        dtype=spatial_feats.dtype,
    )
    lowfreq_feats = torch.einsum("kt,btd->bkd", basis, spatial_feats)
    zero = spatial_feats.sum() * 0.0
    compact_loss = zero
    valid_class_count = 0
    valid_sample_count = 0

    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        class_feats = lowfreq_feats[class_mask]
        if class_feats.shape[0] < 2:
            continue
        class_center = class_feats.mean(dim=0, keepdim=True)
        if str(compact_distance or "mse").lower() in {
            "normalized_mse",
            "l2_normalized_mse",
            "unit_mse",
        }:
            class_feats_cmp = F.normalize(class_feats, dim=2, eps=eps)
            class_center_cmp = F.normalize(class_center, dim=2, eps=eps)
            compact_loss = compact_loss + (
                class_feats_cmp - class_center_cmp
            ).pow(2).sum(dim=2).mean()
        else:
            compact_loss = compact_loss + (class_feats - class_center).pow(2).sum(dim=2).mean()
        valid_class_count += 1
        valid_sample_count += int(class_feats.shape[0])

    if valid_class_count > 0:
        compact_loss = compact_loss / (valid_class_count + eps)
    return compact_loss, valid_class_count, valid_sample_count, lowfreq_feats


def _smooth_time_axis(spatial_feats, kernel_size=3):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return spatial_feats
    if kernel_size % 2 == 0:
        raise ValueError(f"time smoothing kernel_size must be odd, got {kernel_size}")
    padding = kernel_size // 2
    batch_size, time_steps, feat_dim = spatial_feats.shape
    feats = spatial_feats.transpose(1, 2).reshape(batch_size * feat_dim, 1, time_steps)
    feats = F.pad(feats, (padding, padding), mode="replicate")
    kernel = spatial_feats.new_ones(1, 1, kernel_size) / float(kernel_size)
    smoothed = F.conv1d(feats, kernel)
    return smoothed.reshape(batch_size, feat_dim, time_steps).transpose(1, 2)


def _time_permuted_smooth_time_axis(spatial_feats, kernel_size=3, permutation_seed=0):
    time_steps = spatial_feats.shape[1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(permutation_seed) + 1009 * int(time_steps))
    perm = torch.randperm(time_steps, generator=generator).to(spatial_feats.device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(time_steps, device=spatial_feats.device)
    permuted = spatial_feats[:, perm, :]
    smoothed = _smooth_time_axis(permuted, kernel_size=kernel_size)
    return smoothed[:, inv_perm, :]


def _compute_elastic_smoothed_timepoint_compactness(
    spatial_feats,
    labels,
    kernel_size=3,
    radius=0,
    eta=0.1,
    softmin_tau=0.1,
    detach_center=False,
    compact_distance="mse",
    eps=1e-6,
):
    smoothed_feats = _smooth_time_axis(spatial_feats, kernel_size=kernel_size)
    radius = int(radius)
    if radius <= 0:
        compact_loss, valid_class_count, valid_sample_count = _compute_timepoint_compactness(
            smoothed_feats,
            labels,
            compact_distance=compact_distance,
            eps=eps,
        )
        return compact_loss, valid_class_count, valid_sample_count, {
            "mean_abs_offset": 0.0,
            "center_weight": 1.0 if valid_class_count > 0 else 0.0,
            "boundary_weight": 0.0,
            "scale": float(compact_loss.detach().clamp_min(eps).item()),
        }

    zero = spatial_feats.sum() * 0.0
    compact_loss = zero
    valid_class_count = 0
    valid_sample_count = 0
    mean_abs_offset_sum = 0.0
    center_weight_sum = 0.0
    boundary_weight_sum = 0.0
    scale_sum = 0.0
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=spatial_feats.device,
        dtype=torch.long,
    )
    offset_abs = offsets.abs().to(dtype=spatial_feats.dtype)
    offset_norm = offset_abs / float(max(radius, 1))
    center_offset_index = int(radius)
    boundary_mask = offset_abs == float(radius)
    time_index = torch.arange(
        smoothed_feats.shape[1],
        device=spatial_feats.device,
        dtype=torch.long,
    )
    normalized_distance = str(compact_distance or "mse").lower() in {
        "normalized_mse",
        "l2_normalized_mse",
        "unit_mse",
    }

    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        class_feats = smoothed_feats[class_mask]
        if class_feats.shape[0] < 2:
            continue
        class_center = class_feats.mean(dim=0, keepdim=True)
        if detach_center:
            class_center = class_center.detach()
        compare_feats = F.normalize(class_feats, dim=2, eps=eps) if normalized_distance else class_feats
        compare_center = (
            F.normalize(class_center, dim=2, eps=eps) if normalized_distance else class_center
        )

        dist_terms = []
        for offset in offsets:
            shifted_index = (time_index + int(offset.item())).clamp(
                min=0,
                max=smoothed_feats.shape[1] - 1,
            )
            center_shift = compare_center[:, shifted_index, :]
            dist_terms.append((compare_feats - center_shift).pow(2).sum(dim=2))
        dist_stack = torch.stack(dist_terms, dim=-1)
        scale = dist_stack.detach().mean().clamp_min(eps)
        penalty = float(eta) * scale * offset_norm.pow(2)
        cost_stack = dist_stack + penalty.view(1, 1, -1)
        temperature = (float(softmin_tau) * scale).clamp_min(eps)
        weights = torch.softmax(-cost_stack / temperature, dim=-1)
        elastic_dist = (weights * cost_stack).sum(dim=-1)
        compact_loss = compact_loss + elastic_dist.mean()
        valid_class_count += 1
        valid_sample_count += int(class_feats.shape[0])
        with torch.no_grad():
            mean_abs_offset_sum += float((weights * offset_abs.view(1, 1, -1)).sum(dim=-1).mean().item())
            center_weight_sum += float(weights[..., center_offset_index].mean().item())
            boundary_weight_sum += float(weights[..., boundary_mask].sum(dim=-1).mean().item())
            scale_sum += float(scale.item())

    if valid_class_count > 0:
        compact_loss = compact_loss / (valid_class_count + eps)
    return compact_loss, valid_class_count, valid_sample_count, {
        "mean_abs_offset": mean_abs_offset_sum / max(valid_class_count, 1),
        "center_weight": center_weight_sum / max(valid_class_count, 1),
        "boundary_weight": boundary_weight_sum / max(valid_class_count, 1),
        "scale": scale_sum / max(valid_class_count, 1),
    }


def compute_source_raw_global_compactness_loss(
    spatial_feats,
    labels,
    version="v275_raw_global_compactness",
    intra_trade_off=1.0,
    compact_distance="mse",
    time_smooth_kernel_size=3,
    time_permutation_seed=0,
    elastic_radius=0,
    elastic_eta=0.1,
    elastic_softmin_tau=0.1,
    elastic_detach_center=False,
    norm_preserve_trade_off=0.0,
    norm_preserve_target="min_mean",
    norm_preserve_value=1.0,
    eps=1e-6,
):
    """v2.7.5 source-stage raw encoder whole-curve compactness.

    This loss intentionally ignores temporal partitions. It pools each source
    feature curve into one raw encoder representation and pulls same-class
    samples toward their batch class prototype.
    """
    if spatial_feats.dim() != 3:
        raise ValueError(
            "source raw global compactness expects spatial_feats with shape [B, T, D], "
            f"got {tuple(spatial_feats.shape)}"
        )

    zero = spatial_feats.sum() * 0.0
    pooled_feats = spatial_feats.mean(dim=1)
    labels = labels.view(-1)

    version = str(version or "v275_raw_global_compactness").lower()
    lowfreq_components = _lowfreq_dct_components(version)
    umsc_weights = _umsc_weights(version)
    umsc_component_losses = {}
    elastic_logs = {
        "mean_abs_offset": 0.0,
        "center_weight": 0.0,
        "boundary_weight": 0.0,
        "scale": 0.0,
    }
    if version == "v284_elastic_smoothed_timepoint_compactness":
        compact_loss, valid_class_count, valid_sample_count, elastic_logs = (
            _compute_elastic_smoothed_timepoint_compactness(
                spatial_feats,
                labels,
                kernel_size=time_smooth_kernel_size,
                radius=elastic_radius,
                eta=elastic_eta,
                softmin_tau=elastic_softmin_tau,
                detach_center=elastic_detach_center,
                compact_distance=compact_distance,
                eps=eps,
            )
        )
        center_mode = "elastic_smoothed_timepoint"
    elif umsc_weights is not None:
        compact_loss = zero
        valid_class_count = 0
        valid_sample_count = 0
        if umsc_weights["l3"] != 0.0:
            l3_loss, l3_classes, l3_samples = _compute_timepoint_compactness(
                _smooth_time_axis(spatial_feats, kernel_size=3),
                labels,
                compact_distance=compact_distance,
                eps=eps,
            )
            compact_loss = compact_loss + float(umsc_weights["l3"]) * l3_loss
            valid_class_count = max(valid_class_count, l3_classes)
            valid_sample_count = max(valid_sample_count, l3_samples)
            umsc_component_losses["l3"] = l3_loss
        if umsc_weights["l5"] != 0.0:
            l5_loss, l5_classes, l5_samples = _compute_timepoint_compactness(
                _smooth_time_axis(spatial_feats, kernel_size=5),
                labels,
                compact_distance=compact_distance,
                eps=eps,
            )
            compact_loss = compact_loss + float(umsc_weights["l5"]) * l5_loss
            valid_class_count = max(valid_class_count, l5_classes)
            valid_sample_count = max(valid_sample_count, l5_samples)
            umsc_component_losses["l5"] = l5_loss
        if umsc_weights["linf"] != 0.0:
            linf_loss, linf_classes, linf_samples = _compute_global_compactness(
                pooled_feats,
                labels,
                compact_distance=compact_distance,
                center_mode="mean",
                eps=eps,
            )
            compact_loss = compact_loss + float(umsc_weights["linf"]) * linf_loss
            valid_class_count = max(valid_class_count, linf_classes)
            valid_sample_count = max(valid_sample_count, linf_samples)
            umsc_component_losses["linf"] = linf_loss
        center_mode = "umsc"
    elif lowfreq_components is not None:
        compact_loss, valid_class_count, valid_sample_count, _ = _compute_lowfreq_compactness(
            spatial_feats,
            labels,
            lowfreq_components,
            compact_distance=compact_distance,
            eps=eps,
        )
        center_mode = "lowfreq_dct"
    elif version in {
        "v276_raw_smoothed_timepoint_compactness",
        "raw_smoothed_timepoint_compactness",
        "source_raw_smoothed_timepoint_compactness",
    }:
        compact_loss, valid_class_count, valid_sample_count = _compute_timepoint_compactness(
            _smooth_time_axis(spatial_feats, kernel_size=time_smooth_kernel_size),
            labels,
            compact_distance=compact_distance,
            eps=eps,
        )
        center_mode = "smoothed_timepoint"
    elif version in {
        "v303_time_permuted_smoothed_timepoint_compactness",
        "time_permuted_smoothed_timepoint_compactness",
        "source_time_permuted_smoothed_timepoint_compactness",
    }:
        compact_loss, valid_class_count, valid_sample_count = _compute_timepoint_compactness(
            _time_permuted_smooth_time_axis(
                spatial_feats,
                kernel_size=time_smooth_kernel_size,
                permutation_seed=time_permutation_seed,
            ),
            labels,
            compact_distance=compact_distance,
            eps=eps,
        )
        center_mode = "time_permuted_smoothed_timepoint"
    elif version in {"v276_raw_timepoint_compactness", "raw_timepoint_compactness", "source_raw_timepoint_compactness"}:
        compact_loss, valid_class_count, valid_sample_count = _compute_timepoint_compactness(
            spatial_feats,
            labels,
            compact_distance=compact_distance,
            eps=eps,
        )
        center_mode = "timepoint"
    else:
        center_mode = "trimmed" if version in {
            "v276_raw_trimmed_global_compactness",
            "raw_trimmed_global_compactness",
            "source_raw_trimmed_global_compactness",
        } else "mean"
        compact_loss, valid_class_count, valid_sample_count = _compute_global_compactness(
            pooled_feats,
            labels,
            compact_distance=compact_distance,
            center_mode=center_mode,
            eps=eps,
        )

    norm_preserve_loss = zero
    if float(norm_preserve_trade_off) != 0.0:
        norm_class_count = 0
        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            class_feats = pooled_feats[class_mask]
            if class_feats.shape[0] < 2:
                continue
            norm_class_count += 1
            norm_preserve_loss = norm_preserve_loss + _class_norm_preserve_loss(
                class_feats,
                target=norm_preserve_target,
                value=norm_preserve_value,
                eps=eps,
            )
        if norm_class_count > 0:
            norm_preserve_loss = norm_preserve_loss / (norm_class_count + eps)

    weighted_compact_loss = float(intra_trade_off) * compact_loss
    weighted_norm_loss = float(norm_preserve_trade_off) * norm_preserve_loss
    total_loss = weighted_compact_loss + weighted_norm_loss

    with torch.no_grad():
        pooled_norm = pooled_feats.norm(dim=1).mean()
        centered = pooled_feats - pooled_feats.mean(dim=0, keepdim=True)
        pooled_trace = centered.pow(2).sum(dim=1).mean()

    logs = {
        "structure_loss": float(total_loss.detach().item()),
        "compactness_loss": float(total_loss.detach().item()),
        "raw_global_compactness_base_loss": float(compact_loss.detach().item()),
        "raw_global_compactness_weighted_loss": float(weighted_compact_loss.detach().item()),
        "raw_global_valid_classes": float(valid_class_count),
        "raw_global_valid_samples": float(valid_sample_count),
        "raw_global_feature_norm": float(pooled_norm.detach().item()),
        "raw_global_cov_trace": float(pooled_trace.detach().item()),
        "raw_global_intra_trade_off": float(intra_trade_off),
        "source_structure_norm_preserve_loss": float(norm_preserve_loss.detach().item()),
        "source_structure_norm_preserve_weighted_loss": float(weighted_norm_loss.detach().item()),
        "source_structure_norm_preserve_classes": float(valid_class_count),
        "source_structure_norm_preserve_value": float(norm_preserve_value),
        "source_structure_compact_distance_mode": (
            2.0
            if str(compact_distance).lower()
            in {"normalized_mse", "l2_normalized_mse", "unit_mse"}
            else 1.0
        ),
        "source_structure_version_v275_raw_global": 1.0,
        "source_structure_raw_center_mode": (
            8.0 if center_mode == "time_permuted_smoothed_timepoint" else 7.0 if center_mode == "elastic_smoothed_timepoint" else 6.0 if center_mode == "umsc" else 5.0 if center_mode == "lowfreq_dct" else 4.0 if center_mode == "smoothed_timepoint" else 3.0 if center_mode == "timepoint" else 2.0 if center_mode == "trimmed" else 1.0
        ),
        "source_structure_raw_lowfreq_components": float(lowfreq_components or 0),
        "source_structure_time_smooth_kernel_size": float(time_smooth_kernel_size),
        "source_structure_time_permutation_seed": float(time_permutation_seed),
        "source_structure_umsc_l3_weight": float(umsc_weights["l3"] if umsc_weights else 0.0),
        "source_structure_umsc_l5_weight": float(umsc_weights["l5"] if umsc_weights else 0.0),
        "source_structure_umsc_linf_weight": float(umsc_weights["linf"] if umsc_weights else 0.0),
        "source_structure_umsc_l3_loss": float(
            umsc_component_losses.get("l3", zero).detach().item()
        ),
        "source_structure_umsc_l5_loss": float(
            umsc_component_losses.get("l5", zero).detach().item()
        ),
        "source_structure_umsc_linf_loss": float(
            umsc_component_losses.get("linf", zero).detach().item()
        ),
        "source_structure_elastic_radius": float(elastic_radius),
        "source_structure_elastic_eta": float(elastic_eta),
        "source_structure_elastic_softmin_tau": float(elastic_softmin_tau),
        "source_structure_elastic_detach_center": 1.0 if elastic_detach_center else 0.0,
        "elastic_mean_abs_offset": float(elastic_logs["mean_abs_offset"]),
        "elastic_center_weight": float(elastic_logs["center_weight"]),
        "elastic_boundary_weight": float(elastic_logs["boundary_weight"]),
        "elastic_distance_scale": float(elastic_logs["scale"]),
        "elastic_struct_loss": float(compact_loss.detach().item())
        if center_mode == "elastic_smoothed_timepoint"
        else 0.0,
    }
    return total_loss, logs
