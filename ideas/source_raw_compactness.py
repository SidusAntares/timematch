import torch
import torch.nn.functional as F


RAW_GLOBAL_COMPACTNESS_VERSIONS = {
    "v275_raw_global_compactness",
    "raw_global_compactness",
    "source_raw_global_compactness",
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


def compute_source_raw_global_compactness_loss(
    spatial_feats,
    labels,
    intra_trade_off=1.0,
    compact_distance="mse",
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

    compact_loss = zero
    norm_preserve_loss = zero
    valid_class_count = 0
    valid_sample_count = 0

    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        class_feats = pooled_feats[class_mask]
        if class_feats.shape[0] < 2:
            continue
        class_center = class_feats.mean(dim=0, keepdim=True)
        compact_loss = compact_loss + _compactness_distance(
            class_feats,
            class_center,
            mode=compact_distance,
            eps=eps,
        )
        if float(norm_preserve_trade_off) != 0.0:
            norm_preserve_loss = norm_preserve_loss + _class_norm_preserve_loss(
                class_feats,
                target=norm_preserve_target,
                value=norm_preserve_value,
                eps=eps,
            )
        valid_class_count += 1
        valid_sample_count += int(class_feats.shape[0])

    if valid_class_count > 0:
        compact_loss = compact_loss / (valid_class_count + eps)
        norm_preserve_loss = norm_preserve_loss / (valid_class_count + eps)

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
    }
    return total_loss, logs
