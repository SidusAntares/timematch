import json
import os

import torch
import torch.nn.functional as F

from ideas.v271_decomposition import split_trend_residual, temporal_differences


def load_v271_adaptive_supports(path, min_score=0.0, min_gate=0.0):
    path = str(path or "").strip()
    if not path:
        return [], {"timematch_v271_adaptive_support_file": ""}
    if not os.path.exists(path):
        print(f"v2.7.1 adaptive supports: missing file {path}; disabled")
        return [], {"timematch_v271_adaptive_support_missing": 1.0}
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    raw_supports = payload.get("adaptive_supports", payload.get("supports", []))
    supports = []
    for item in raw_supports:
        classes = item.get("classes", item.get("class_ids", item.get("pair", item.get("class_pair"))))
        if isinstance(classes, str):
            classes = [int(tok) for tok in classes.replace(":", ",").replace(";", ",").split(",") if tok != ""]
        if classes is None:
            continue
        classes = sorted({int(x) for x in classes})
        if len(classes) < 1:
            continue
        score = float(item.get("score", item.get("support_score", 0.0)))
        gate = float(item.get("gate", item.get("support_gate", 1.0)))
        if score < float(min_score) or gate < float(min_gate):
            continue
        start = item.get("start", item.get("start_doy"))
        end = item.get("end", item.get("end_doy"))
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if end < start:
            start, end = end, start
        supports.append(
            {
                "classes": classes,
                "start": start,
                "end": end,
                "score": score,
                "gate": max(0.0, min(1.0, gate)),
            }
        )

    logs = {
        "timematch_v271_adaptive_support_file": path,
        "timematch_v271_adaptive_support_count": float(len(supports)),
    }
    if supports:
        text = ", ".join(
            [f"{support['classes']}@[{support['start']},{support['end']}]" for support in supports]
        )
        print(f"v2.7.1 adaptive supports: count={len(supports)}, supports={text}")
    else:
        print(f"v2.7.1 adaptive supports: no valid supports in {path}; disabled")
    return supports, logs


def _sort_by_positions(feats, positions):
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
    return torch.gather(feats, dim=1, index=expanded), torch.gather(positions, dim=1, index=sort_indices)


def _interval_mask(positions, start, end):
    return (positions >= int(start)) & (positions <= int(end))


def _pooled_interval(curves, mask, eps=1e-6):
    weights = mask.to(dtype=curves.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(eps)
    return (curves * weights).sum(dim=1) / denom


def _classwise_pooled_cohesion(pooled, labels, valid, class_ids, eps=1e-6):
    zero = pooled.sum() * 0.0
    loss = zero
    class_count = 0
    for class_id in class_ids:
        class_mask = (labels == int(class_id)) & valid
        if int(class_mask.sum().item()) < 2:
            continue
        class_feats = pooled[class_mask]
        center = class_feats.mean(dim=0, keepdim=True)
        loss = loss + (class_feats - center).pow(2).sum(dim=1).mean()
        class_count += 1
    if class_count > 0:
        loss = loss / (class_count + eps)
    return loss, class_count


def _labels_in_classes(labels, class_ids):
    class_mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in class_ids:
        class_mask = class_mask | (labels == int(class_id))
    return class_mask


def _classwise_pooled_dynamics_consistency(pooled, labels, valid, class_ids, mode="cosine", eps=1e-6):
    zero = pooled.sum() * 0.0
    loss = zero
    class_count = 0
    mode = str(mode or "cosine").lower()
    for class_id in class_ids:
        class_mask = (labels == int(class_id)) & valid
        if int(class_mask.sum().item()) < 2:
            continue
        class_dyn = pooled[class_mask]
        center = class_dyn.mean(dim=0, keepdim=True).expand_as(class_dyn)
        if mode == "mse":
            class_loss = (class_dyn - center).pow(2).sum(dim=1).mean()
        elif mode in {"cosine", "direction"}:
            sample_norm = class_dyn.norm(dim=1)
            center_norm = center.norm(dim=1)
            nonzero = (sample_norm > eps) & (center_norm > eps)
            cosine_loss = 1.0 - F.cosine_similarity(class_dyn, center, dim=1, eps=eps)
            class_loss = cosine_loss[nonzero].mean() if bool(nonzero.any().item()) else zero
        else:
            raise ValueError(f"Unsupported dynamics mode: {mode}")
        loss = loss + class_loss
        class_count += 1
    if class_count > 0:
        loss = loss / (class_count + eps)
    return loss, class_count


def _masked_residual_energy(residual, mask, margin):
    weights = mask.to(dtype=residual.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    energy = (residual.pow(2) * weights).sum() / denom
    loss = torch.relu(energy - residual.new_tensor(float(margin))).pow(2)
    return loss, energy


def _interval_dynamics_curves(trend, positions, mask):
    if trend.shape[1] < 2:
        return trend.new_zeros((trend.shape[0], trend.shape[-1])), mask.new_zeros((trend.shape[0],), dtype=torch.bool)
    diffs = temporal_differences(trend, positions=positions)
    pair_mask = mask[:, 1:] & mask[:, :-1]
    weights = pair_mask.to(dtype=trend.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    pooled = (diffs * weights).sum(dim=1) / denom
    return pooled, pair_mask.sum(dim=1) > 0


def compute_v271_adaptive_support_loss(
    feats,
    positions,
    labels,
    supports,
    trend_kernel_size=5,
    trend_smoothing_mode="time",
    trend_bandwidth=0.0,
    trend_kernel="gaussian",
    trend_cohesion_trade_off=1.0,
    trend_dynamics_trade_off=0.05,
    residual_variance_trade_off=0.10,
    residual_energy_trade_off=0.05,
    residual_energy_margin=1.0,
    min_points=2,
    dynamics_mode="cosine",
    eps=1e-6,
):
    zero = feats.sum() * 0.0
    if not supports:
        return zero, {
            "timematch_v271_adaptive_active": 0.0,
            "timematch_v271_adaptive_support_count": 0.0,
        }

    ordered_feats, ordered_positions = _sort_by_positions(feats, positions)
    trend, residual = split_trend_residual(
        ordered_feats,
        positions=ordered_positions,
        kernel_size=trend_kernel_size,
        mode=trend_smoothing_mode,
        bandwidth=trend_bandwidth,
        kernel=trend_kernel,
    )
    total = zero
    trend_loss_total = zero
    dynamics_loss_total = zero
    residual_variance_total = zero
    residual_energy_loss_total = zero
    residual_energy_total = zero
    active_supports = 0
    active_classes = 0

    for support in supports:
        mask = _interval_mask(ordered_positions, support["start"], support["end"])
        counts = mask.sum(dim=1)
        valid = counts >= int(min_points)
        if not bool(valid.any().item()):
            continue

        class_ids = support["classes"]
        gate = float(support.get("gate", 1.0))
        class_valid = valid & _labels_in_classes(labels, class_ids)
        if not bool(class_valid.any().item()):
            continue

        trend_pooled = _pooled_interval(trend, mask, eps=eps)
        trend_loss, trend_classes = _classwise_pooled_cohesion(
            trend_pooled,
            labels,
            valid,
            class_ids,
            eps=eps,
        )
        dyn_pooled, dyn_valid = _interval_dynamics_curves(trend, ordered_positions, mask)
        dyn_loss, dyn_classes = _classwise_pooled_dynamics_consistency(
            dyn_pooled,
            labels,
            valid & dyn_valid,
            class_ids,
            mode=dynamics_mode,
            eps=eps,
        )
        dyn_loss = dyn_loss if bool(dyn_valid.any().item()) else zero
        residual_pooled = _pooled_interval(residual, mask, eps=eps)
        residual_var, residual_classes = _classwise_pooled_cohesion(
            residual_pooled,
            labels,
            valid,
            class_ids,
            eps=eps,
        )
        residual_energy_loss, residual_energy = _masked_residual_energy(
            residual[class_valid],
            mask[class_valid],
            residual_energy_margin,
        )

        support_loss = gate * (
            float(trend_cohesion_trade_off) * trend_loss
            + float(trend_dynamics_trade_off) * dyn_loss
            + float(residual_variance_trade_off) * residual_var
            + float(residual_energy_trade_off) * residual_energy_loss
        )
        total = total + support_loss
        trend_loss_total = trend_loss_total + gate * trend_loss
        dynamics_loss_total = dynamics_loss_total + gate * dyn_loss
        residual_variance_total = residual_variance_total + gate * residual_var
        residual_energy_loss_total = residual_energy_loss_total + gate * residual_energy_loss
        residual_energy_total = residual_energy_total + residual_energy
        active_supports += 1
        active_classes += max(trend_classes, dyn_classes, residual_classes)

    if active_supports > 0:
        scale = float(active_supports)
        total = total / scale
        trend_loss_total = trend_loss_total / scale
        dynamics_loss_total = dynamics_loss_total / scale
        residual_variance_total = residual_variance_total / scale
        residual_energy_loss_total = residual_energy_loss_total / scale
        residual_energy_total = residual_energy_total / scale

    logs = {
        "timematch_v271_adaptive_active": 1.0 if active_supports > 0 else 0.0,
        "timematch_v271_adaptive_support_count": float(active_supports),
        "timematch_v271_adaptive_class_count": float(active_classes),
        "timematch_v271_adaptive_trend_bandwidth": float(trend_bandwidth),
        "timematch_v271_adaptive_trend_loss": trend_loss_total,
        "timematch_v271_adaptive_dynamics_loss": dynamics_loss_total,
        "timematch_v271_adaptive_residual_variance_loss": residual_variance_total,
        "timematch_v271_adaptive_residual_energy_loss": residual_energy_loss_total,
        "timematch_v271_adaptive_residual_energy": residual_energy_total,
    }
    return total, logs
