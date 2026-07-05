#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.v301_anchor_correspondence_diagnostic import (  # noqa: E402
    bool_value,
    create_train_val_test_folds,
    load_model,
    resolve_classes,
)
from analysis.v302_anchor_order_diagnostic import build_loader  # noqa: E402
from analysis import summarize_v306_alignment_control_audit as v306  # noqa: E402
from dataset import count_pixelset_samples  # noqa: E402


TASKS = [
    "FR1_to_FR2",
    "AT1_to_DK1",
    "FR2_to_AT1",
    "DK1_to_AT1",
    "FR2_to_FR1",
    "AT1_to_FR2",
]
CONFIGS = ["plain", "raw_global", "smooth_k3", "time_permuted_smooth_k3"]
CONFIG_TO_RAW = {
    "plain": "plain",
    "raw_global": "v275_raw_w1",
    "smooth_k3": "v276_smooth_k3_w1",
    "time_permuted_smooth_k3": "v303_time_permuted_smooth_k3_w1",
}
CONFIG_ALIASES = {
    "plain": "plain",
    "raw": "raw_global",
    "raw_global": "raw_global",
    "v275_raw_w1": "raw_global",
    "smooth_k3": "smooth_k3",
    "v276_smooth_k3_w1": "smooth_k3",
    "time_permuted_smooth_k3": "time_permuted_smooth_k3",
    "v303_time_permuted_smooth_k3_w1": "time_permuted_smooth_k3",
}


def parse_args():
    parser = argparse.ArgumentParser("v3.0.5 local temporal state alignment diagnostic")
    parser.add_argument("--v303_dir", default="logs/v303_control_and_intraclass_diagnostic_20260702_120442")
    parser.add_argument("--v304_dir", default="logs/v304_mechanism_reanalysis_20260703_190144")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--source_run_tag", default="v303_control_and_intraclass_diagnostic_20260702_120442_control_runs")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--closed_set", default="True")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_pixels", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--with_extra", default="False")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_shift", type=int, default=8)
    parser.add_argument("--min_samples_per_class_time", type=int, default=5)
    parser.add_argument("--max_source_samples", type=int, default=0)
    parser.add_argument("--max_target_samples", type=int, default=0)
    parser.add_argument("--random_repeats", type=int, default=50)
    parser.add_argument("--shuffle_repeats", type=int, default=50)
    parser.add_argument("--write_cost_matrix", default="False")
    parser.add_argument("--audit_summary_mode", default="False")
    parser.add_argument("--audit_shuffle_repeats", type=int, default=100)
    parser.add_argument("--audit_block_size", type=int, default=3)
    parser.add_argument("--audit_bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3050)
    return parser.parse_args()


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


class StreamingTsvWriter:
    def __init__(self, path, fields):
        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        self.writer.writeheader()
        self.handle.flush()

    def write_many(self, rows):
        for row in rows:
            self.writer.writerow({field: fmt(row.get(field)) for field in self.fields})
        self.handle.flush()

    def close(self):
        self.handle.close()


def safe_float(value, default=None):
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(value, default=None):
    val = safe_float(value, default=None)
    if val is None:
        return default
    return int(round(val))


def mean(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def std(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return None
    m = mean(vals)
    return float(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)))


def pearson(xs, ys):
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xv = np.asarray([p[0] for p in pairs], dtype=np.float64)
    yv = np.asarray([p[1] for p in pairs], dtype=np.float64)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = float(np.linalg.norm(xv) * np.linalg.norm(yv))
    if denom <= 1e-12:
        return None
    return float((xv @ yv) / denom)


def rankdata(values):
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(xs, ys):
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    return pearson(rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs]))


def canonical_config(config):
    return CONFIG_ALIASES.get(str(config), str(config))


def source_model_name(source_dataset, task, seed, raw_config, source_run_tag, closed_set):
    tile = source_dataset.split("/")[1]
    set_tag = "closedset" if bool_value(closed_set) else "openset"
    return f"pseltae_{tile}_{set_tag}_noshift_{source_run_tag}_{task}_seed{seed}_{raw_config}_source"


def checkpoint_path(args, row):
    raw_config = CONFIG_TO_RAW.get(row["config"], row["config"])
    model_name = source_model_name(row["source"], row["task"], row["seed"], raw_config, args.source_run_tag, args.closed_set)
    return Path(args.outputs_root) / model_name / "fold_0" / "model.pt"


@torch.no_grad()
def extract_temporal_features_with_logits(model, loader, device):
    features, labels, positions, indices = [], [], [], []
    logits_list, confs, preds = [], [], []
    for sample in loader:
        pixels = sample["pixels"].to(device=device, non_blocking=True)
        mask = sample["valid_pixels"].to(device=device, non_blocking=True)
        extra = sample["extra"].to(device=device, non_blocking=True)
        pos = sample["positions"].to(device=device, non_blocking=True)
        feats = model.spatial_encoder(pixels, mask, extra)
        logits = model.decoder(model.temporal_encoder(feats, pos))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        features.append(feats.detach().cpu().float().numpy())
        labels.append(sample["label"].detach().cpu().long().numpy())
        positions.append(sample["positions"].detach().cpu().long().numpy())
        indices.append(sample["index"].detach().cpu().long().numpy())
        logits_list.append(logits.detach().cpu().float().numpy())
        confs.append(conf.detach().cpu().float().numpy())
        preds.append(pred.detach().cpu().long().numpy())
    return {
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "positions": np.concatenate(positions, axis=0),
        "indices": np.concatenate(indices, axis=0),
        "logits": np.concatenate(logits_list, axis=0),
        "pseudo_confidence": np.concatenate(confs, axis=0),
        "pseudo_labels": np.concatenate(preds, axis=0),
    }


def local_window_features(features, sample_mask, time_idx, window_size):
    half = window_size // 2
    t = features.shape[1]
    lo = max(0, time_idx - half)
    hi = min(t, time_idx + half + 1)
    block = features[sample_mask, lo:hi, :]
    if block.size == 0:
        return None
    return block.reshape(-1, features.shape[-1])


def distribution_stats(features, sample_mask, window_size, min_count):
    t = features.shape[1]
    stats = []
    for idx in range(t):
        vals = local_window_features(features, sample_mask, idx, window_size)
        if vals is None or vals.shape[0] < min_count:
            stats.append(None)
            continue
        mean_vec = vals.mean(axis=0)
        if vals.shape[0] >= 2:
            centered = vals - mean_vec
            cov = (centered.T @ centered) / max(vals.shape[0] - 1, 1)
        else:
            cov = np.zeros((features.shape[-1], features.shape[-1]), dtype=np.float64)
        stats.append({"n": int(vals.shape[0]), "mean": mean_vec.astype(np.float64), "cov": cov.astype(np.float64)})
    return stats


def cost_matrices(src_stats, tgt_stats):
    n_source = len(src_stats)
    n_target = len(tgt_stats)
    mean_l2 = np.full((n_source, n_target), np.nan, dtype=np.float64)
    coral = np.full((n_source, n_target), np.nan, dtype=np.float64)
    for i, ss in enumerate(src_stats):
        if ss is None:
            continue
        for j, tt in enumerate(tgt_stats):
            if tt is None:
                continue
            mean_l2[i, j] = float(np.linalg.norm(ss["mean"] - tt["mean"]))
            coral[i, j] = float(np.linalg.norm(ss["cov"] - tt["cov"], ord="fro") / max(ss["cov"].shape[0], 1))
    return mean_l2, coral


def valid_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def shift_indices(n_source, n_target, shift):
    return [(i, i + shift) for i in range(n_source) if 0 <= i + shift < n_target]


def distance_for_pairs(cost, pairs):
    vals = [cost[i, j] for i, j in pairs if 0 <= i < cost.shape[0] and 0 <= j < cost.shape[1] and np.isfinite(cost[i, j])]
    return valid_mean(vals), len(vals)


def best_scalar_shift(cost, max_shift):
    best_shift, best_dist, best_valid = 0, None, 0
    for shift in range(-max_shift, max_shift + 1):
        dist, valid = distance_for_pairs(cost, shift_indices(cost.shape[0], cost.shape[1], shift))
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_shift, best_dist, best_valid = shift, dist, valid
    return best_shift, best_dist, best_valid


def soft_alignment(cost, shift, radius, tau):
    total, weight_total = 0.0, 0
    entropies, deviations, max_devs = [], [], []
    n_source, n_target = cost.shape
    for i in range(n_source):
        center = i + shift
        candidates = [
            j
            for j in range(center - radius, center + radius + 1)
            if 0 <= j < n_target and np.isfinite(cost[i, j])
        ]
        if not candidates:
            continue
        vals = np.asarray([cost[i, j] for j in candidates], dtype=np.float64)
        scale = tau if tau > 0 else max(float(np.nanstd(vals)), 1e-6)
        logits = -vals / max(scale, 1e-6)
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        total += float((weights * vals).sum())
        weight_total += 1
        entropies.append(float(-(weights * np.log(weights + 1e-12)).sum()))
        dev = np.asarray([abs(j - center) for j in candidates], dtype=np.float64)
        deviations.append(float((weights * dev).sum()))
        max_devs.append(float(dev.max()))
    if weight_total == 0:
        return None, 0, None, None, None
    return total / weight_total, weight_total, mean(entropies), mean(deviations), max(max_devs) if max_devs else None


def monotonic_alignment(cost, shift, radius):
    n_source, n_target = cost.shape
    inf = float("inf")
    dp = np.full((n_source + 1, n_target + 1), inf, dtype=np.float64)
    prev = {}
    dp[0, 0] = 0.0
    for i in range(1, n_source + 1):
        for j in range(1, n_target + 1):
            ti, uj = i - 1, j - 1
            if abs(uj - (ti + shift)) > radius or not np.isfinite(cost[ti, uj]):
                continue
            choices = [(dp[i - 1, j], (i - 1, j)), (dp[i, j - 1], (i, j - 1)), (dp[i - 1, j - 1], (i - 1, j - 1))]
            best_val, best_prev = min(choices, key=lambda x: x[0])
            if math.isinf(best_val):
                continue
            dp[i, j] = best_val + cost[ti, uj]
            prev[(i, j)] = best_prev
    endpoints = [(n_source, j) for j in range(1, n_target + 1)] + [
        (i, n_target) for i in range(1, n_source + 1)
    ]
    endpoints = [p for p in endpoints if np.isfinite(dp[p])]
    if not endpoints:
        return None, 0, None, None, None, None, 0
    end = min(endpoints, key=lambda p: dp[p])
    path = []
    cur = end
    while cur != (0, 0) and cur in prev:
        if cur[0] > 0 and cur[1] > 0:
            path.append((cur[0] - 1, cur[1] - 1))
        cur = prev[cur]
    path.reverse()
    if not path:
        return None, 0, None, None, None, None, 0
    dist = float(sum(cost[i, j] for i, j in path) / len(path))
    steps, deviations = [], []
    large = 0
    last = path[0]
    for p in path[1:]:
        step = max(abs(p[0] - last[0]), abs(p[1] - last[1]))
        steps.append(step)
        if step > 1:
            large += 1
        last = p
    for i, j in path:
        deviations.append(abs(j - (i + shift)))
    return (
        dist,
        len(path),
        float(len(path) / max(n_source, n_target, 1)),
        mean(steps),
        max(steps) if steps else 0,
        mean(deviations),
        large,
    )


def random_alignment(cost, shift, radius, repeats, rng):
    values = []
    n_source, n_target = cost.shape
    for _ in range(repeats):
        chosen = []
        for i in range(n_source):
            center = i + shift
            candidates = [
                j
                for j in range(center - radius, center + radius + 1)
                if 0 <= j < n_target and np.isfinite(cost[i, j])
            ]
            if candidates:
                chosen.append(cost[i, int(rng.choice(candidates))])
        if chosen:
            values.append(float(np.mean(chosen)))
    return mean(values), std(values)


def shuffled_alignment(cost, shift, radius, tau, repeats, rng):
    soft_values, mono_values = [], []
    n_target = cost.shape[1]
    for _ in range(repeats):
        perm = rng.permutation(n_target)
        shuffled = cost[:, perm]
        soft_dist, *_ = soft_alignment(shuffled, shift, radius, tau)
        mono_dist, *_ = monotonic_alignment(shuffled, shift, radius)
        if soft_dist is not None:
            soft_values.append(soft_dist)
        if mono_dist is not None:
            mono_values.append(mono_dist)
    return mean(soft_values), std(soft_values), mean(mono_values), std(mono_values)


def local_alignment_for_cost(cost, estimated_shift, args, rng):
    n_source, n_target = cost.shape
    max_shift = min(args.max_shift, max(max(n_source, n_target) // 2, abs(n_source - n_target), 1))
    no_align, n_no = distance_for_pairs(cost, shift_indices(n_source, n_target, 0))
    global_shift, n_global = distance_for_pairs(cost, shift_indices(n_source, n_target, estimated_shift))
    best_shift, best_dist, n_best = best_scalar_shift(cost, max_shift)
    local_soft, n_soft, entropy, mean_dev, max_dev = soft_alignment(cost, estimated_shift, args.radius, args.tau)
    mono = monotonic_alignment(cost, estimated_shift, args.radius)
    rand_mean, rand_std = random_alignment(cost, estimated_shift, args.radius, args.random_repeats, rng)
    shuf_soft, shuf_soft_std, shuf_mono, shuf_mono_std = shuffled_alignment(
        cost, estimated_shift, args.radius, args.tau, args.shuffle_repeats, rng
    )
    return {
        "dist_no_align": no_align,
        "n_no_align": n_no,
        "dist_global_shift": global_shift,
        "n_global_shift": n_global,
        "best_scalar_shift": best_shift,
        "dist_best_scalar_shift": best_dist,
        "n_best_scalar_shift": n_best,
        "dist_local_soft": local_soft,
        "n_local_soft": n_soft,
        "dist_monotonic": mono[0],
        "dist_random": rand_mean,
        "dist_random_std": rand_std,
        "dist_local_soft_shuffled": shuf_soft,
        "dist_local_soft_shuffled_std": shuf_soft_std,
        "dist_monotonic_shuffled": shuf_mono,
        "dist_monotonic_shuffled_std": shuf_mono_std,
        "local_soft_improvement_vs_global": diff(global_shift, local_soft),
        "monotonic_improvement_vs_global": diff(global_shift, mono[0]),
        "local_soft_improvement_vs_random": diff(rand_mean, local_soft),
        "monotonic_improvement_vs_random": diff(rand_mean, mono[0]),
        "local_soft_real_vs_shuffled_gap": diff(shuf_soft, local_soft),
        "monotonic_real_vs_shuffled_gap": diff(shuf_mono, mono[0]),
        "alignment_entropy": entropy,
        "mean_abs_local_deviation": mean_dev,
        "max_abs_local_deviation": max_dev,
        "path_length": mono[1],
        "path_length_ratio": mono[2],
        "mean_step_size": mono[3],
        "max_step_size": mono[4],
        "num_large_jumps": mono[6],
    }


def diff(a, b):
    if a is None or b is None:
        return None
    return a - b


def class_name(classes, class_id):
    if class_id is None or int(class_id) < 0 or int(class_id) >= len(classes):
        return ""
    return classes[int(class_id)]


def label_mode_masks(source_obj, target_obj, label_mode, class_id=None):
    src_labels = source_obj["labels"]
    tgt_labels = target_obj["labels"]
    if label_mode == "all":
        return np.ones_like(src_labels, dtype=bool), np.ones_like(tgt_labels, dtype=bool)
    src_mask = src_labels == class_id
    if label_mode == "oracle":
        tgt_mask = tgt_labels == class_id
    elif label_mode == "pseudo":
        tgt_mask = target_obj["pseudo_labels"] == class_id
    elif label_mode == "pseudo_conf09":
        tgt_mask = (target_obj["pseudo_labels"] == class_id) & (target_obj["pseudo_confidence"] >= 0.9)
    else:
        raise ValueError(label_mode)
    return src_mask, tgt_mask


def collect_feature_pair(args, row, classes, checkpoint):
    random.seed(int(row["seed"]))
    np.random.seed(int(row["seed"]))
    counts = {
        row["source"]: count_pixelset_samples(
            args.data_root,
            row["source"],
            classes,
            closed_set=bool_value(args.closed_set),
        ),
        row["target"]: count_pixelset_samples(
            args.data_root,
            row["target"],
            classes,
            closed_set=bool_value(args.closed_set),
        ),
    }
    folds = create_train_val_test_folds([row["source"], row["target"]], 1, counts, 0.1, 0.2)
    splits = folds[0]
    source_loader = build_loader(args, row["source"], classes, splits[row["source"]]["train"], args.max_source_samples)
    target_loader = build_loader(args, row["target"], classes, splits[row["target"]]["train"], args.max_target_samples)
    model_args = argparse.Namespace(
        input_dim=args.input_dim,
        with_extra=args.with_extra,
        checkpoint_path=str(checkpoint),
        source_model="",
        device=args.device,
    )
    model, _ = load_model(model_args, classes)
    source_obj = extract_temporal_features_with_logits(model, source_loader, args.device)
    target_obj = extract_temporal_features_with_logits(model, target_loader, args.device)
    return source_obj, target_obj


def parse_control_rows(args):
    path = Path(args.v303_dir) / "control_da_results.tsv"
    rows = read_tsv(path)
    out = []
    for row in rows:
        task = row.get("task", "")
        config = canonical_config(row.get("config", ""))
        seed = str(row.get("seed", ""))
        if task not in TASKS or config not in CONFIGS or seed not in {"1", "2", "3"}:
            continue
        out.append({
            "task": task,
            "source": row.get("source", ""),
            "target": row.get("target", ""),
            "seed": int(seed),
            "config": config,
            "source_on_target_f1": safe_float(row.get("source_on_target_f1")),
            "da_f1": safe_float(row.get("da_f1")),
            "da_gain": safe_float(row.get("da_gain")),
            "estimated_shift": safe_int(row.get("estimated_shift"), 0),
            "shift_missing": 0 if str(row.get("estimated_shift", "")).strip() else 1,
        })
    return out[: args.limit_rows] if args.limit_rows else out


def safe_div(a, b):
    if a is None or b is None or abs(b) <= 1e-12:
        return None
    return a / b


def emit_v306_audit(audit, out_base, metric, cost, estimated_shift, args, rng):
    if audit is None:
        return
    geom = v306.cost_geometry(cost, estimated_shift, args.radius)
    geometry_row = {
        "task": out_base["task"],
        "source": out_base["source"],
        "target": out_base["target"],
        "seed": out_base["seed"],
        "config": out_base["config"],
        "label_mode": out_base["label_mode"],
        "class_id": out_base["class_id"],
        "distance_metric": metric,
        **geom,
        "valid_class": out_base.get("valid_class", 1),
        "skip_reason": out_base.get("skip_reason", ""),
    }
    real_soft, behavior = v306.soft_alignment(cost, estimated_shift, args.radius, args.tau)
    real_mono, _, _, _ = v306.monotonic_alignment(cost, estimated_shift, args.radius)
    behavior_row = {
        "task": out_base["task"],
        "seed": out_base["seed"],
        "config": out_base["config"],
        "label_mode": out_base["label_mode"],
        "class_id": out_base["class_id"],
        "distance_metric": metric,
        **behavior,
        "interpretation_flag": v306.interpret_flag(behavior, None),
    }
    dist_global = geom.get("global_shift_band_cost")
    finite = cost[np.isfinite(cost)]
    if finite.size:
        z_cost = (cost - float(np.mean(finite))) / max(float(np.std(finite)), 1e-12)
        z_soft, _ = v306.soft_alignment(z_cost, estimated_shift, args.radius, args.tau)
        z_mono, _, _, _ = v306.monotonic_alignment(z_cost, estimated_shift, args.radius)
        z_perm = z_cost[:, rng.permutation(z_cost.shape[1])]
        z_soft_ctrl, _ = v306.soft_alignment(z_perm, estimated_shift, args.radius, args.tau)
        z_mono_ctrl, _, _, _ = v306.monotonic_alignment(z_perm, estimated_shift, args.radius)
    else:
        z_soft = z_mono = z_soft_ctrl = z_mono_ctrl = None
    shuffle_rows = []
    full_shuffle_local_gaps = []
    full_shuffle_monotonic_gaps = []
    for control_type in ["full_shuffle", "circular_shift", "block_shuffle", "within_band_random", "time_reversed"]:
        repeats = 1 if control_type == "time_reversed" else args.audit_shuffle_repeats
        for repeat_id in range(repeats):
            if control_type == "within_band_random":
                ctrl_soft, ctrl_mean_dev, ctrl_large = v306.random_within_band(cost, estimated_shift, args.radius, rng)
                ctrl_mono = None
                ctrl_path = None
            else:
                ctrl_cost = v306.transformed_cost(cost, control_type, rng, args.audit_block_size)
                ctrl_soft, _ = v306.soft_alignment(ctrl_cost, estimated_shift, args.radius, args.tau)
                ctrl_mono, ctrl_path, ctrl_mean_dev, ctrl_large = v306.monotonic_alignment(ctrl_cost, estimated_shift, args.radius)
            local_gap = v306.diff(ctrl_soft, real_soft)
            mono_gap = v306.diff(ctrl_mono, real_mono)
            if control_type == "full_shuffle":
                full_shuffle_local_gaps.append(local_gap)
                full_shuffle_monotonic_gaps.append(mono_gap)
            shuffle_rows.append({
                "task": out_base["task"],
                "seed": out_base["seed"],
                "config": out_base["config"],
                "label_mode": out_base["label_mode"],
                "class_id": out_base["class_id"],
                "distance_metric": metric,
                "control_type": control_type,
                "repeat_id": repeat_id,
                "real_local_soft_dist": real_soft,
                "control_local_soft_dist": ctrl_soft,
                "real_minus_control_local_soft_gap": local_gap,
                "real_monotonic_dist": real_mono,
                "control_monotonic_dist": ctrl_mono,
                "real_minus_control_monotonic_gap": mono_gap,
                "control_path_length_ratio": ctrl_path,
                "control_mean_abs_deviation": ctrl_mean_dev,
                "control_num_large_jumps": ctrl_large,
            })
    raw_local_shuffle_gap = mean(full_shuffle_local_gaps)
    raw_mono_shuffle_gap = mean(full_shuffle_monotonic_gaps)
    normalized_row = {
        "task": out_base["task"],
        "seed": out_base["seed"],
        "config": out_base["config"],
        "label_mode": out_base["label_mode"],
        "class_id": out_base["class_id"],
        "distance_metric": metric,
        "raw_local_soft_improvement": v306.diff(dist_global, real_soft),
        "relative_local_soft_improvement": safe_div(v306.diff(dist_global, real_soft), dist_global),
        "raw_monotonic_improvement": v306.diff(dist_global, real_mono),
        "relative_monotonic_improvement": safe_div(v306.diff(dist_global, real_mono), dist_global),
        "raw_local_real_vs_shuffled_gap": raw_local_shuffle_gap,
        "relative_local_real_vs_shuffled_gap": safe_div(raw_local_shuffle_gap, real_soft),
        "z_normalized_local_real_vs_shuffled_gap": None if z_soft_ctrl is None or z_soft is None else z_soft_ctrl - z_soft,
        "raw_monotonic_real_vs_shuffled_gap": raw_mono_shuffle_gap,
        "relative_monotonic_real_vs_shuffled_gap": safe_div(raw_mono_shuffle_gap, real_mono),
        "z_normalized_monotonic_real_vs_shuffled_gap": None if z_mono_ctrl is None or z_mono is None else z_mono_ctrl - z_mono,
        "dist_global_shift": dist_global,
        "dist_local_soft": real_soft,
        "dist_monotonic": real_mono,
    }
    audit["geometry"].append(geometry_row)
    audit["behavior"].append(behavior_row)
    audit["normalized"].append(normalized_row)
    for row in shuffle_rows:
        stats = audit["shuffle_stats"][row["control_type"]]
        stats["local"].append(row.get("real_minus_control_local_soft_gap"))
        stats["monotonic"].append(row.get("real_minus_control_monotonic_gap"))
    audit["geometry_writer"].write_many([geometry_row])
    audit["behavior_writer"].write_many([behavior_row])
    audit["normalized_writer"].write_many([normalized_row])
    audit["shuffle_writer"].write_many(shuffle_rows)


def analyze_row(args, row, output_dir, cost_writer=None, audit=None):
    checkpoint = checkpoint_path(args, row)
    base = {
        "task": row["task"],
        "source": row["source"],
        "target": row["target"],
        "seed": row["seed"],
        "config": row["config"],
        "feature_stage": "source_stage_encoder",
        "estimated_global_shift": row["estimated_shift"],
        "shift_missing": row["shift_missing"],
    }
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    classes = resolve_classes(args.data_root, row["source"], bool_value(args.closed_set))
    source_obj, target_obj = collect_feature_pair(args, row, classes, checkpoint)
    feature_rows = [{
        **base,
        "checkpoint_path": str(checkpoint),
        "n_source_samples": int(source_obj["features"].shape[0]),
        "n_target_samples": int(target_obj["features"].shape[0]),
        "feature_shape": "x".join(str(x) for x in source_obj["features"].shape),
    }]
    rows = []
    class_specs = [("all", -1)]
    class_ids = sorted(set(int(x) for x in source_obj["labels"].tolist()))
    for cid in class_ids:
        class_specs.append(("oracle", cid))
        class_specs.append(("pseudo", cid))
        class_specs.append(("pseudo_conf09", cid))
    rng = np.random.default_rng(args.seed + int(row["seed"]) * 1009)
    for label_mode, class_id in class_specs:
        src_mask, tgt_mask = label_mode_masks(source_obj, target_obj, label_mode, class_id if class_id >= 0 else None)
        out_base = {
            **base,
            "label_mode": label_mode,
            "class_id": class_id,
            "class_name_if_available": class_name(classes, class_id),
            "n_source_samples": int(src_mask.sum()),
            "n_target_samples": int(tgt_mask.sum()),
            "valid_class": 1,
            "skip_reason": "",
        }
        if int(src_mask.sum()) < args.min_samples_per_class_time or int(tgt_mask.sum()) < args.min_samples_per_class_time:
            rows.append({**out_base, "valid_class": 0, "skip_reason": "too_few_samples"})
            continue
        src_stats = distribution_stats(source_obj["features"], src_mask, args.window_size, args.min_samples_per_class_time)
        tgt_stats = distribution_stats(target_obj["features"], tgt_mask, args.window_size, args.min_samples_per_class_time)
        mean_cost, coral_cost = cost_matrices(src_stats, tgt_stats)
        mean_diag = local_alignment_for_cost(mean_cost, int(row["estimated_shift"]), args, rng)
        coral_diag = local_alignment_for_cost(coral_cost, int(row["estimated_shift"]), args, rng)
        emit_v306_audit(audit, out_base, "mean_l2", mean_cost, int(row["estimated_shift"]), args, rng)
        emit_v306_audit(audit, out_base, "coral", coral_cost, int(row["estimated_shift"]), args, rng)
        n_valid = sum(1 for idx in range(mean_cost.shape[0]) if np.isfinite(mean_cost[idx]).any())
        out = {
            **out_base,
            "n_valid_time_points": n_valid,
            **{f"{k}_mean_l2": v for k, v in mean_diag.items()},
            **{f"{k}_coral": v for k, v in coral_diag.items() if k.startswith("dist_") or k.endswith("_gap") or k.endswith("_vs_global") or k.endswith("_vs_random")},
            "best_scalar_shift": mean_diag.get("best_scalar_shift"),
            "alignment_entropy": mean_diag.get("alignment_entropy"),
            "mean_abs_local_deviation": mean_diag.get("mean_abs_local_deviation"),
            "max_abs_local_deviation": mean_diag.get("max_abs_local_deviation"),
            "path_length": mean_diag.get("path_length"),
            "path_length_ratio": mean_diag.get("path_length_ratio"),
            "mean_step_size": mean_diag.get("mean_step_size"),
            "max_step_size": mean_diag.get("max_step_size"),
            "num_large_jumps": mean_diag.get("num_large_jumps"),
        }
        rows.append(out)
        if cost_writer is not None:
            for i in range(mean_cost.shape[0]):
                for j in range(mean_cost.shape[1]):
                    if np.isfinite(mean_cost[i, j]) or np.isfinite(coral_cost[i, j]):
                        cost_writer.writerow({
                            **out_base,
                            "time_source": i,
                            "time_target": j,
                            "n_source": src_stats[i]["n"] if src_stats[i] is not None else "",
                            "n_target": tgt_stats[j]["n"] if tgt_stats[j] is not None else "",
                            "mean_l2": fmt(float(mean_cost[i, j])) if np.isfinite(mean_cost[i, j]) else "",
                            "coral": fmt(float(coral_cost[i, j])) if np.isfinite(coral_cost[i, j]) else "",
                        })
    return rows, feature_rows


def weighted_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("valid_class", "0")) != "1":
            continue
        key = (row["task"], row["source"], row["target"], row["seed"], row["config"], row["feature_stage"], row["label_mode"])
        grouped[key].append(row)
    out = []
    for key, vals in sorted(grouped.items()):
        weights = np.asarray([max(safe_float(v.get("n_target_samples"), 0), 1.0) for v in vals], dtype=np.float64)
        weights = weights / weights.sum()
        def wmean(field):
            arr = np.asarray([safe_float(v.get(field), float("nan")) for v in vals], dtype=np.float64)
            mask = np.isfinite(arr)
            if not mask.any():
                return None
            ww = weights[mask] / weights[mask].sum()
            return float((arr[mask] * ww).sum())
        task, source, target, seed, config, feature_stage, label_mode = key
        out.append({
            "task": task,
            "source": source,
            "target": target,
            "seed": seed,
            "config": config,
            "feature_stage": feature_stage,
            "label_mode": label_mode,
            "n_valid_classes": len(vals),
            "weighted_local_soft_improvement_vs_global": wmean("local_soft_improvement_vs_global_mean_l2"),
            "weighted_monotonic_improvement_vs_global": wmean("monotonic_improvement_vs_global_mean_l2"),
            "weighted_local_soft_improvement_vs_random": wmean("local_soft_improvement_vs_random_mean_l2"),
            "weighted_monotonic_improvement_vs_random": wmean("monotonic_improvement_vs_random_mean_l2"),
            "weighted_local_soft_real_vs_shuffled_gap": wmean("local_soft_real_vs_shuffled_gap_mean_l2"),
            "weighted_monotonic_real_vs_shuffled_gap": wmean("monotonic_real_vs_shuffled_gap_mean_l2"),
            "weighted_alignment_entropy": wmean("alignment_entropy"),
            "weighted_mean_abs_local_deviation": wmean("mean_abs_local_deviation"),
            "weighted_path_length_ratio": wmean("path_length_ratio"),
            "weighted_num_large_jumps": wmean("num_large_jumps"),
            "weighted_dist_global_shift": wmean("dist_global_shift_mean_l2"),
            "weighted_dist_local_soft": wmean("dist_local_soft_mean_l2"),
            "weighted_dist_monotonic": wmean("dist_monotonic_mean_l2"),
            "weighted_dist_random": wmean("dist_random_mean_l2"),
            "weighted_dist_shuffled": wmean("dist_local_soft_shuffled_mean_l2"),
            "positive_local_soft_classes": sum(1 for v in vals if safe_float(v.get("local_soft_improvement_vs_global_mean_l2"), -1) > 0),
            "positive_monotonic_classes": sum(1 for v in vals if safe_float(v.get("monotonic_improvement_vs_global_mean_l2"), -1) > 0),
            "positive_real_vs_shuffled_classes": sum(1 for v in vals if safe_float(v.get("local_soft_real_vs_shuffled_gap_mean_l2"), -1) > 0),
        })
    return out


def build_da_relation(summary_rows, da_rows):
    da_map = {(r["task"], str(r["seed"]), r["config"]): r for r in da_rows}
    out = []
    for row in summary_rows:
        da = da_map.get((row["task"], str(row["seed"]), row["config"]), {})
        out.append({
            **row,
            "da_f1": da.get("da_f1"),
            "source_on_target_f1": da.get("source_on_target_f1"),
            "da_gain": da.get("da_gain"),
        })
    return out


def correlation_summary(rows):
    predictors = [
        "weighted_local_soft_improvement_vs_global",
        "weighted_monotonic_improvement_vs_global",
        "weighted_local_soft_real_vs_shuffled_gap",
        "weighted_monotonic_real_vs_shuffled_gap",
        "weighted_mean_abs_local_deviation",
        "weighted_path_length_ratio",
        "weighted_num_large_jumps",
    ]
    targets = ["da_f1", "source_on_target_f1", "da_gain"]
    out = []
    for label_mode in sorted(set(r["label_mode"] for r in rows)):
        subset = [r for r in rows if r["label_mode"] == label_mode]
        for predictor in predictors:
            xs = [r.get(predictor) for r in subset]
            for target in targets:
                ys = [r.get(target) for r in subset]
                out.append({
                    "label_mode": label_mode,
                    "predictor": predictor,
                    "target": target,
                    "n": sum(1 for x, y in zip(xs, ys) if safe_float(x) is not None and safe_float(y) is not None),
                    "pearson_corr": pearson(xs, ys),
                    "spearman_corr": spearman(xs, ys),
                })
    return out


def write_summary_md(path, args, diag_rows, task_rows, failed_rows):
    def avg(field, mode):
        return mean([r.get(field) for r in task_rows if r["label_mode"] == mode])
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v3.0.5 局部时间状态软对齐诊断总结\n\n")
        handle.write("## 1. 实验目的\n\n")
        handle.write("本轮不训练新模型，只用已有 checkpoint 离线比较 global shift、局部 soft alignment、受限单调 alignment 与随机/打乱时间对照。\n\n")
        handle.write("## 2. 设置\n\n")
        handle.write(f"- window_size={args.window_size}\n")
        handle.write(f"- radius={args.radius}\n")
        handle.write(f"- tau={args.tau}\n")
        handle.write(f"- random_repeats={args.random_repeats}\n")
        handle.write(f"- shuffle_repeats={args.shuffle_repeats}\n")
        handle.write(f"- 有效 class 记录：{sum(1 for r in diag_rows if str(r.get('valid_class')) == '1')}\n")
        handle.write(f"- 失败记录：{len(failed_rows)}\n\n")
        handle.write("## 3. 汇总指标\n\n")
        handle.write("| label_mode | local soft vs global | monotonic vs global | real vs shuffled | path ratio |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for mode in sorted(set(r["label_mode"] for r in task_rows)):
            handle.write(
                f"| {mode} | {fmt(avg('weighted_local_soft_improvement_vs_global', mode))} | "
                f"{fmt(avg('weighted_monotonic_improvement_vs_global', mode))} | "
                f"{fmt(avg('weighted_local_soft_real_vs_shuffled_gap', mode))} | "
                f"{fmt(avg('weighted_path_length_ratio', mode))} |\n"
            )
        handle.write("\n## 4. 说明\n\n")
        handle.write("oracle label 只用于离线上界诊断，不能作为无监督训练信号。相关性只作辅助，不作因果结论。\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("logs") / f"v305_local_temporal_state_alignment_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    control_rows = parse_control_rows(args)
    all_diag, feature_rows, failed_rows = [], [], []
    cost_handle = None
    cost_writer = None
    cost_fields = [
        "task", "source", "target", "seed", "config", "feature_stage", "label_mode", "class_id",
        "time_source", "time_target", "n_source", "n_target", "mean_l2", "coral",
    ]
    if bool_value(args.write_cost_matrix):
        cost_handle = (output_dir / "local_cost_matrix.tsv").open("w", encoding="utf-8", newline="")
        cost_writer = csv.DictWriter(cost_handle, fieldnames=cost_fields, delimiter="\t", extrasaction="ignore")
        cost_writer.writeheader()
    else:
        write_tsv(output_dir / "local_cost_matrix.tsv", [], cost_fields)

    diag_fields = [
        "task", "source", "target", "seed", "config", "feature_stage", "label_mode", "class_id", "class_name_if_available",
        "n_source_samples", "n_target_samples", "n_valid_time_points",
        "dist_no_align_mean_l2", "dist_global_shift_mean_l2", "dist_best_scalar_shift_mean_l2", "dist_local_soft_mean_l2",
        "dist_monotonic_mean_l2", "dist_random_mean_l2", "dist_random_std_mean_l2", "dist_local_soft_shuffled_mean_l2",
        "dist_monotonic_shuffled_mean_l2", "dist_no_align_coral", "dist_global_shift_coral", "dist_best_scalar_shift_coral",
        "dist_local_soft_coral", "dist_monotonic_coral", "dist_random_coral", "dist_local_soft_shuffled_coral",
        "dist_monotonic_shuffled_coral", "best_scalar_shift", "estimated_global_shift", "shift_missing",
        "local_soft_improvement_vs_global_mean_l2", "monotonic_improvement_vs_global_mean_l2",
        "local_soft_improvement_vs_random_mean_l2", "monotonic_improvement_vs_random_mean_l2",
        "local_soft_real_vs_shuffled_gap_mean_l2", "monotonic_real_vs_shuffled_gap_mean_l2",
        "alignment_entropy", "mean_abs_local_deviation", "max_abs_local_deviation",
        "path_length", "path_length_ratio", "mean_step_size", "max_step_size", "num_large_jumps",
        "valid_class", "skip_reason",
    ]
    feature_fields = [
        "task", "source", "target", "seed", "config", "feature_stage", "estimated_global_shift",
        "shift_missing", "checkpoint_path", "n_source_samples", "n_target_samples", "feature_shape",
    ]
    failed_fields = ["task", "source", "target", "seed", "config", "error", "checkpoint_path"]
    diag_stream = StreamingTsvWriter(output_dir / "local_alignment_diagnostic.tsv", diag_fields)
    feature_stream = StreamingTsvWriter(output_dir / "feature_extraction_index.tsv", feature_fields)
    failed_stream = StreamingTsvWriter(output_dir / "failed_runs.tsv", failed_fields)
    audit = None
    if bool_value(args.audit_summary_mode):
        audit = {
            "geometry": [],
            "shuffle_stats": defaultdict(lambda: {"local": [], "monotonic": []}),
            "behavior": [],
            "normalized": [],
            "geometry_writer": StreamingTsvWriter(output_dir / "v306_cost_matrix_geometry.tsv", v306.GEOMETRY_FIELDS),
            "shuffle_writer": StreamingTsvWriter(output_dir / "v306_shuffle_control_audit.tsv", v306.SHUFFLE_FIELDS),
            "behavior_writer": StreamingTsvWriter(output_dir / "v306_local_soft_behavior.tsv", v306.BEHAVIOR_FIELDS),
            "normalized_writer": StreamingTsvWriter(output_dir / "v306_normalized_alignment_summary.tsv", v306.NORMALIZED_FIELDS),
            "status_writer": StreamingTsvWriter(
                output_dir / "v306_audit_status.tsv",
                ["task", "seed", "config", "status", "geometry_rows", "behavior_rows", "normalized_rows", "shuffle_rows", "error"],
            ),
        }

    for idx, row in enumerate(control_rows, start=1):
        print(f"V305_START|{idx}/{len(control_rows)}|task={row['task']}|seed={row['seed']}|config={row['config']}", flush=True)
        audit_counts_before = None
        if audit is not None:
            audit_counts_before = (
                len(audit["geometry"]),
                len(audit["behavior"]),
                len(audit["normalized"]),
                sum(len(v["local"]) for v in audit["shuffle_stats"].values()),
            )
        try:
            diag, feats = analyze_row(args, row, output_dir, cost_writer, audit)
            all_diag.extend(diag)
            feature_rows.extend(feats)
            diag_stream.write_many(diag)
            feature_stream.write_many(feats)
            if audit is not None:
                before = audit_counts_before
                after = (
                    len(audit["geometry"]),
                    len(audit["behavior"]),
                    len(audit["normalized"]),
                    sum(len(v["local"]) for v in audit["shuffle_stats"].values()),
                )
                audit["status_writer"].write_many([{
                    "task": row["task"],
                    "seed": row["seed"],
                    "config": row["config"],
                    "status": "DONE",
                    "geometry_rows": after[0] - before[0],
                    "behavior_rows": after[1] - before[1],
                    "normalized_rows": after[2] - before[2],
                    "shuffle_rows": after[3] - before[3],
                    "error": "",
                }])
            print(f"V305_DONE|task={row['task']}|seed={row['seed']}|config={row['config']}|rows={len(diag)}", flush=True)
        except Exception as exc:  # keep the batch running
            failed = {**row, "error": repr(exc), "checkpoint_path": str(checkpoint_path(args, row))}
            failed_rows.append(failed)
            failed_stream.write_many([failed])
            if audit is not None:
                before = audit_counts_before
                after = (
                    len(audit["geometry"]),
                    len(audit["behavior"]),
                    len(audit["normalized"]),
                    sum(len(v["local"]) for v in audit["shuffle_stats"].values()),
                )
                audit["status_writer"].write_many([{
                    "task": row["task"],
                    "seed": row["seed"],
                    "config": row["config"],
                    "status": "FAIL",
                    "geometry_rows": after[0] - before[0],
                    "behavior_rows": after[1] - before[1],
                    "normalized_rows": after[2] - before[2],
                    "shuffle_rows": after[3] - before[3],
                    "error": repr(exc),
                }])
            print(f"V305_FAIL|task={row['task']}|seed={row['seed']}|config={row['config']}|error={repr(exc)}", flush=True)
    diag_stream.close()
    feature_stream.close()
    failed_stream.close()
    if cost_handle is not None:
        cost_handle.close()
    if audit is not None:
        for key in ["geometry_writer", "shuffle_writer", "behavior_writer", "normalized_writer"]:
            audit[key].close()
        audit["status_writer"].close()

    task_rows = weighted_summary(all_diag)
    da_relation = build_da_relation(task_rows, control_rows)
    corr_rows = correlation_summary(da_relation)
    if audit is not None:
        audit_args = argparse.Namespace(
            seed=args.seed,
            bootstrap_repeats=args.audit_bootstrap_repeats,
            v305_dir=str(output_dir),
            v304_dir=args.v304_dir,
        )
        monotonic_rows = v306.build_monotonic_stability(all_diag, audit_args, random.Random(args.seed + 306))
        audit_corr = v306.build_da_correlation(
            audit["normalized"],
            audit["geometry"],
            audit["behavior"],
            da_relation,
        )
        write_tsv(output_dir / "v306_monotonic_stability.tsv", monotonic_rows, v306.MONOTONIC_FIELDS)
        write_tsv(output_dir / "v306_da_relation_correlation.tsv", audit_corr, v306.CORR_FIELDS)
        shuffle_summary_rows = [
            {
                "control_type": control,
                "real_minus_control_local_soft_gap": mean(values["local"]),
                "real_minus_control_monotonic_gap": mean(values["monotonic"]),
            }
            for control, values in sorted(audit["shuffle_stats"].items())
        ]
        v306.write_summary(
            output_dir,
            audit_args,
            audit["geometry"],
            shuffle_summary_rows,
            audit["behavior"],
            monotonic_rows,
            audit["normalized"],
            audit_corr,
            [],
            failed_rows,
            len(audit["geometry"]),
            sum(1 for r in audit["geometry"] if str(r.get("valid_class")) == "1"),
        )
    else:
        write_tsv(output_dir / "v306_cost_matrix_geometry.tsv", [], v306.GEOMETRY_FIELDS)
        write_tsv(output_dir / "v306_shuffle_control_audit.tsv", [], v306.SHUFFLE_FIELDS)
        write_tsv(output_dir / "v306_local_soft_behavior.tsv", [], v306.BEHAVIOR_FIELDS)
        write_tsv(output_dir / "v306_monotonic_stability.tsv", [], v306.MONOTONIC_FIELDS)
        write_tsv(output_dir / "v306_normalized_alignment_summary.tsv", [], v306.NORMALIZED_FIELDS)
        write_tsv(output_dir / "v306_da_relation_correlation.tsv", [], v306.CORR_FIELDS)

    task_fields = [
        "task", "source", "target", "seed", "config", "feature_stage", "label_mode", "n_valid_classes",
        "weighted_local_soft_improvement_vs_global", "weighted_monotonic_improvement_vs_global",
        "weighted_local_soft_improvement_vs_random", "weighted_monotonic_improvement_vs_random",
        "weighted_local_soft_real_vs_shuffled_gap", "weighted_monotonic_real_vs_shuffled_gap",
        "weighted_alignment_entropy", "weighted_mean_abs_local_deviation", "weighted_path_length_ratio",
        "weighted_num_large_jumps", "weighted_dist_global_shift", "weighted_dist_local_soft",
        "weighted_dist_monotonic", "weighted_dist_random", "weighted_dist_shuffled",
        "positive_local_soft_classes", "positive_monotonic_classes", "positive_real_vs_shuffled_classes",
    ]
    relation_fields = task_fields + ["da_f1", "source_on_target_f1", "da_gain"]
    write_tsv(output_dir / "local_alignment_task_summary.tsv", task_rows, task_fields)
    write_tsv(output_dir / "local_alignment_da_relation.tsv", da_relation, relation_fields)
    write_tsv(output_dir / "local_alignment_correlation_summary.tsv", corr_rows, ["label_mode", "predictor", "target", "n", "pearson_corr", "spearman_corr"])
    write_tsv(output_dir / "missing_records.tsv", [], ["task", "seed", "config", "missing"])
    write_summary_md(output_dir / "v305_local_temporal_state_alignment_diagnostic_summary.md", args, all_diag, task_rows, failed_rows)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"CONTROL_ROWS={len(control_rows)}")
    print(f"DIAGNOSTIC_ROWS={len(all_diag)}")
    print(f"TASK_SUMMARY_ROWS={len(task_rows)}")
    print(f"FAILED_ROWS={len(failed_rows)}")


if __name__ == "__main__":
    main()
