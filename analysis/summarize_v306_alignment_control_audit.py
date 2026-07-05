#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


TASKS = ["FR1_to_FR2", "AT1_to_DK1", "FR2_to_AT1", "DK1_to_AT1", "FR2_to_FR1", "AT1_to_FR2"]
CONFIGS = ["plain", "raw_global", "smooth_k3", "time_permuted_smooth_k3"]
LABEL_MODES = ["all", "oracle", "pseudo", "pseudo_conf09"]
METRICS = ["mean_l2", "coral"]


def parse_args():
    parser = argparse.ArgumentParser("v3.0.6 alignment control audit")
    parser.add_argument("--v305_dir", default="logs/v305_local_temporal_state_alignment_diagnostic_20260704_185542")
    parser.add_argument("--v304_dir", default="logs/v304_mechanism_reanalysis_20260703_190144")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--shuffle_repeats", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=3)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3060)
    return parser.parse_args()


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


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
    val = safe_float(value)
    if val is None:
        return default
    return int(round(val))


def mean(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def std(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
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
    xv -= xv.mean()
    yv -= yv.mean()
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


def same_direction(xs, ys):
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None and x != 0 and y != 0]
    if not pairs:
        return 0, None
    count = sum(1 for x, y in pairs if (x > 0 and y > 0) or (x < 0 and y < 0))
    return count, float(count / len(pairs))


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def iter_tsv(path):
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def default_output_dir():
    return Path("logs") / f"v306_alignment_control_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def key_from_cost_row(row):
    return (
        row.get("task", ""),
        str(row.get("seed", "")),
        row.get("config", ""),
        row.get("label_mode", ""),
        str(row.get("class_id", "")),
    )


def key_from_diag_row(row):
    return (
        row.get("task", ""),
        str(row.get("seed", "")),
        row.get("config", ""),
        row.get("label_mode", ""),
        str(row.get("class_id", "")),
    )


def build_diag_map(diag_rows):
    return {key_from_diag_row(row): row for row in diag_rows}


def build_cost_groups(cost_path):
    groups = defaultdict(list)
    total_rows = 0
    for row in iter_tsv(cost_path):
        total_rows += 1
        if row.get("task") not in TASKS:
            continue
        if row.get("config") not in CONFIGS:
            continue
        if row.get("label_mode") not in LABEL_MODES:
            continue
        groups[key_from_cost_row(row)].append(row)
    return groups, total_rows


def build_cost_matrix(rows, metric):
    entries = []
    max_i = max_j = -1
    for row in rows:
        i = safe_int(row.get("time_source"))
        j = safe_int(row.get("time_target"))
        val = safe_float(row.get(metric))
        if i is None or j is None or val is None:
            continue
        entries.append((i, j, val))
        max_i = max(max_i, i)
        max_j = max(max_j, j)
    if max_i < 0 or max_j < 0 or not entries:
        return None
    cost = np.full((max_i + 1, max_j + 1), np.nan, dtype=np.float64)
    for i, j, val in entries:
        cost[i, j] = val
    return cost


def valid_pairs(cost, shift, radius=None):
    n_source, n_target = cost.shape
    pairs = []
    for i in range(n_source):
        center = i + shift
        if radius is None:
            candidates = [center]
        else:
            candidates = range(center - radius, center + radius + 1)
        for j in candidates:
            if 0 <= j < n_target and np.isfinite(cost[i, j]):
                pairs.append((i, j))
    return pairs


def pair_mean(cost, pairs):
    vals = [cost[i, j] for i, j in pairs if np.isfinite(cost[i, j])]
    return mean(vals)


def row_min(cost, shift=None, radius=None):
    vals = []
    n_source, n_target = cost.shape
    for i in range(n_source):
        if shift is None or radius is None:
            js = range(n_target)
        else:
            center = i + shift
            js = range(center - radius, center + radius + 1)
        row_vals = [cost[i, j] for j in js if 0 <= j < n_target and np.isfinite(cost[i, j])]
        if row_vals:
            vals.append(min(row_vals))
    return mean(vals)


def cost_geometry(cost, shift, radius):
    n_source, n_target = cost.shape
    global_pairs = valid_pairs(cost, shift, radius=0)
    band_pairs = valid_pairs(cost, shift, radius=radius)
    band_set = set(band_pairs)
    finite_pairs = [(i, j) for i in range(n_source) for j in range(n_target) if np.isfinite(cost[i, j])]
    off_band_pairs = [p for p in finite_pairs if p not in band_set]
    global_cost = pair_mean(cost, global_pairs)
    band_mean = pair_mean(cost, band_pairs)
    off_band_mean = pair_mean(cost, off_band_pairs)
    band_row = row_min(cost, shift=shift, radius=radius)
    full_row = row_min(cost)
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return {}
    filled = np.where(np.isfinite(cost), cost, np.nanmean(finite))
    try:
        singular = np.linalg.svd(filled, compute_uv=False)
        energy = float(np.sum(singular ** 2))
        low_rank_ratio = float((singular[0] ** 2) / energy) if energy > 1e-12 else None
        probs = (singular ** 2) / energy if energy > 1e-12 else np.asarray([])
        effective_rank = float(math.exp(-float(np.sum(probs * np.log(probs + 1e-12))))) if probs.size else None
    except np.linalg.LinAlgError:
        low_rank_ratio = None
        effective_rank = None
    return {
        "n_source_time": n_source,
        "n_target_time": n_target,
        "n_valid_entries": int(finite.size),
        "global_shift_band_cost": global_cost,
        "row_min_cost": full_row,
        "band_row_min_cost": band_row,
        "full_row_min_cost": full_row,
        "row_min_advantage": diff(global_cost, band_row),
        "full_search_extra_advantage": diff(band_row, full_row),
        "cost_temporal_contrast": diff(off_band_mean, band_mean),
        "diagonal_band_advantage": diff(off_band_mean, band_mean),
        "cost_matrix_mean": float(np.nanmean(cost)),
        "cost_matrix_std": float(np.nanstd(cost)),
        "row_mean_std": float(np.nanstd(np.nanmean(cost, axis=1))),
        "col_mean_std": float(np.nanstd(np.nanmean(cost, axis=0))),
        "low_rank_ratio": low_rank_ratio,
        "effective_rank": effective_rank,
        "temporal_autocorr_source_axis": adjacent_corr(cost, axis=0),
        "temporal_autocorr_target_axis": adjacent_corr(cost, axis=1),
    }


def adjacent_corr(cost, axis):
    vals = []
    n = cost.shape[axis]
    for idx in range(n - 1):
        a = cost[idx, :] if axis == 0 else cost[:, idx]
        b = cost[idx + 1, :] if axis == 0 else cost[:, idx + 1]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            continue
        aa = a[mask] - np.mean(a[mask])
        bb = b[mask] - np.mean(b[mask])
        denom = np.linalg.norm(aa) * np.linalg.norm(bb)
        if denom > 1e-12:
            vals.append(float((aa @ bb) / denom))
    return mean(vals)


def diff(a, b):
    if a is None or b is None:
        return None
    return a - b


def soft_alignment(cost, shift, radius, tau):
    n_source, n_target = cost.shape
    total = []
    entropies, effs, max_weights = [], [], []
    selected_offsets, abs_offsets, argmin_offsets, soft_offsets = [], [], [], []
    edge_hits = 0
    best_vs_global, best_vs_second = [], []
    for i in range(n_source):
        center = i + shift
        candidates = [j for j in range(center - radius, center + radius + 1) if 0 <= j < n_target and np.isfinite(cost[i, j])]
        if not candidates:
            continue
        vals = np.asarray([cost[i, j] for j in candidates], dtype=np.float64)
        scale = tau if tau > 0 else max(float(np.nanstd(vals)), 1e-6)
        logits = -vals / max(scale, 1e-6)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        offsets = np.asarray([j - center for j in candidates], dtype=np.float64)
        total.append(float((weights * vals).sum()))
        entropy = float(-(weights * np.log(weights + 1e-12)).sum())
        entropies.append(entropy)
        effs.append(float(math.exp(entropy)))
        max_weights.append(float(weights.max()))
        sel = float((weights * offsets).sum())
        selected_offsets.append(sel)
        abs_offsets.append(abs(sel))
        soft_offsets.append(int(round(sel)))
        argmin_idx = int(np.argmin(vals))
        argmin_offsets.append(int(offsets[argmin_idx]))
        if abs(int(offsets[argmin_idx])) >= radius:
            edge_hits += 1
        global_cost = cost[i, center] if 0 <= center < n_target and np.isfinite(cost[i, center]) else None
        if global_cost is not None:
            best_vs_global.append(float(global_cost - vals[argmin_idx]))
        if len(vals) >= 2:
            sorted_vals = np.sort(vals)
            best_vs_second.append(float(sorted_vals[1] - sorted_vals[0]))
    n = len(total)
    if n == 0:
        return None, {}
    return mean(total), {
        "alignment_entropy_mean": mean(entropies),
        "effective_candidates_mean": mean(effs),
        "max_weight_mean": mean(max_weights),
        "selected_offset_mean": mean(selected_offsets),
        "selected_offset_abs_mean": mean(abs_offsets),
        "selected_offset_std": std(selected_offsets),
        "edge_selection_rate": float(edge_hits / n),
        "cost_gap_best_vs_global": mean(best_vs_global),
        "cost_gap_best_vs_second_best": mean(best_vs_second),
        "argmin_offset_histogram_json": json.dumps(histogram(argmin_offsets), ensure_ascii=False, sort_keys=True),
        "soft_expected_offset_histogram_json": json.dumps(histogram(soft_offsets), ensure_ascii=False, sort_keys=True),
    }


def histogram(values):
    counts = defaultdict(int)
    for val in values:
        counts[str(val)] += 1
    return dict(counts)


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
    endpoints = [(n_source, j) for j in range(1, n_target + 1)] + [(i, n_target) for i in range(1, n_source + 1)]
    endpoints = [p for p in endpoints if np.isfinite(dp[p])]
    if not endpoints:
        return None, None, None, None
    end = min(endpoints, key=lambda p: dp[p])
    path, cur = [], end
    while cur != (0, 0) and cur in prev:
        if cur[0] > 0 and cur[1] > 0:
            path.append((cur[0] - 1, cur[1] - 1))
        cur = prev[cur]
    path.reverse()
    if not path:
        return None, None, None, None
    dist = float(sum(cost[i, j] for i, j in path) / len(path))
    steps, devs, large = [], [], 0
    last = path[0]
    for p in path[1:]:
        step = max(abs(p[0] - last[0]), abs(p[1] - last[1]))
        steps.append(step)
        if step > 1:
            large += 1
        last = p
    for i, j in path:
        devs.append(abs(j - (i + shift)))
    return dist, float(len(path) / max(n_source, n_target, 1)), mean(devs), large


def transformed_cost(cost, control_type, rng, block_size):
    n_target = cost.shape[1]
    if control_type == "full_shuffle":
        return cost[:, rng.permutation(n_target)]
    if control_type == "circular_shift":
        offset = int(rng.integers(0, max(n_target, 1)))
        return np.roll(cost, shift=offset, axis=1)
    if control_type == "block_shuffle":
        blocks = [np.arange(i, min(i + block_size, n_target)) for i in range(0, n_target, block_size)]
        order = list(range(len(blocks)))
        rng.shuffle(order)
        perm = np.concatenate([blocks[i] for i in order]) if blocks else np.arange(n_target)
        return cost[:, perm]
    if control_type == "time_reversed":
        return cost[:, ::-1]
    return cost


def random_within_band(cost, shift, radius, rng):
    chosen = []
    n_source, n_target = cost.shape
    offsets, large = [], 0
    last_j = None
    for i in range(n_source):
        center = i + shift
        candidates = [j for j in range(center - radius, center + radius + 1) if 0 <= j < n_target and np.isfinite(cost[i, j])]
        if not candidates:
            continue
        j = int(rng.choice(candidates))
        chosen.append(cost[i, j])
        offsets.append(abs(j - center))
        if last_j is not None and abs(j - last_j) > 1:
            large += 1
        last_j = j
    return mean(chosen), float(mean(offsets) or 0.0), large


def interpret_flag(behavior, real_vs_shuffle):
    flags = []
    if (behavior.get("effective_candidates_mean") or 0) >= 2.5 and (behavior.get("cost_gap_best_vs_second_best") or 0) <= 1e-4:
        flags.append("diffuse_soft_matching")
    if (behavior.get("edge_selection_rate") or 0) >= 0.5:
        flags.append("boundary_seeking")
    if (behavior.get("selected_offset_abs_mean") or 0) <= 0.25:
        flags.append("close_to_global_shift")
    if (behavior.get("selected_offset_abs_mean") or 0) > 0.25 and real_vs_shuffle is not None and real_vs_shuffle <= 0:
        flags.append("non_temporal_low_cost_search")
    return ",".join(flags) if flags else "none"


def bootstrap_ci(values, rng, repeats):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None
    means = []
    for _ in range(repeats):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scope_filter(rows, scope):
    if scope == "all_configs":
        return rows
    if scope == "excluding_plain":
        return [r for r in rows if r.get("config") != "plain"]
    if scope == "raw_smooth_timeperm_only":
        return [r for r in rows if r.get("config") in {"raw_global", "smooth_k3", "time_permuted_smooth_k3"}]
    return rows


def load_da_map(v304_dir):
    rows = read_tsv(Path(v304_dir) / "mechanism_master_table.tsv")
    out = {}
    for row in rows:
        key = (row.get("task", ""), str(row.get("seed", "")), row.get("config", ""))
        out[key] = row
    return out


def write_empty_outputs(output_dir):
    write_tsv(output_dir / "v306_cost_matrix_geometry.tsv", [], GEOMETRY_FIELDS)
    write_tsv(output_dir / "v306_shuffle_control_audit.tsv", [], SHUFFLE_FIELDS)
    write_tsv(output_dir / "v306_local_soft_behavior.tsv", [], BEHAVIOR_FIELDS)
    write_tsv(output_dir / "v306_monotonic_stability.tsv", [], MONOTONIC_FIELDS)
    write_tsv(output_dir / "v306_normalized_alignment_summary.tsv", [], NORMALIZED_FIELDS)
    write_tsv(output_dir / "v306_da_relation_correlation.tsv", [], CORR_FIELDS)


GEOMETRY_FIELDS = [
    "task", "source", "target", "seed", "config", "label_mode", "class_id", "distance_metric",
    "n_source_time", "n_target_time", "n_valid_entries", "global_shift_band_cost", "row_min_cost",
    "band_row_min_cost", "full_row_min_cost", "row_min_advantage", "full_search_extra_advantage",
    "cost_temporal_contrast", "diagonal_band_advantage", "cost_matrix_mean", "cost_matrix_std",
    "row_mean_std", "col_mean_std", "low_rank_ratio", "effective_rank", "temporal_autocorr_source_axis",
    "temporal_autocorr_target_axis", "valid_class", "skip_reason",
]
SHUFFLE_FIELDS = [
    "task", "seed", "config", "label_mode", "class_id", "distance_metric", "control_type", "repeat_id",
    "real_local_soft_dist", "control_local_soft_dist", "real_minus_control_local_soft_gap",
    "real_monotonic_dist", "control_monotonic_dist", "real_minus_control_monotonic_gap",
    "control_path_length_ratio", "control_mean_abs_deviation", "control_num_large_jumps",
]
BEHAVIOR_FIELDS = [
    "task", "seed", "config", "label_mode", "class_id", "distance_metric", "alignment_entropy_mean",
    "effective_candidates_mean", "max_weight_mean", "selected_offset_mean", "selected_offset_abs_mean",
    "selected_offset_std", "edge_selection_rate", "cost_gap_best_vs_global", "cost_gap_best_vs_second_best",
    "argmin_offset_histogram_json", "soft_expected_offset_histogram_json", "interpretation_flag",
]
MONOTONIC_FIELDS = [
    "task", "config", "label_mode", "distance_metric", "mean_monotonic_real_vs_shuffled_gap",
    "std_monotonic_real_vs_shuffled_gap", "positive_seed_count", "n_seed", "mean_gap_excluding_plain",
    "positive_task_count_excluding_plain", "bootstrap_mean", "bootstrap_ci_low", "bootstrap_ci_high",
    "leave_one_task_min", "leave_one_task_max", "stability_flag",
]
NORMALIZED_FIELDS = [
    "task", "seed", "config", "label_mode", "class_id", "distance_metric", "raw_local_soft_improvement",
    "relative_local_soft_improvement", "raw_monotonic_improvement", "relative_monotonic_improvement",
    "raw_local_real_vs_shuffled_gap", "relative_local_real_vs_shuffled_gap",
    "z_normalized_local_real_vs_shuffled_gap", "raw_monotonic_real_vs_shuffled_gap",
    "relative_monotonic_real_vs_shuffled_gap", "z_normalized_monotonic_real_vs_shuffled_gap",
    "dist_global_shift", "dist_local_soft", "dist_monotonic",
]
CORR_FIELDS = ["predictor", "target_metric", "label_mode", "config_scope", "n", "pearson", "spearman", "same_direction_count", "same_direction_rate"]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    v305_dir = Path(args.v305_dir)
    cost_path = v305_dir / "local_cost_matrix.tsv"
    diag_path = v305_dir / "local_alignment_diagnostic.tsv"
    task_summary_path = v305_dir / "local_alignment_task_summary.tsv"
    da_path = v305_dir / "local_alignment_da_relation.tsv"
    corr_path = v305_dir / "local_alignment_correlation_summary.tsv"
    input_files = [cost_path, diag_path, task_summary_path, da_path, corr_path]
    missing = [{"path": str(path), "reason": "missing_input_file"} for path in input_files if not path.exists()]
    if missing:
        write_empty_outputs(output_dir)
        write_tsv(output_dir / "missing_records.tsv", missing, ["path", "reason"])
        write_tsv(output_dir / "failed_runs.tsv", [{"error": "missing_required_input"}], ["error"])
        write_summary(output_dir, args, [], [], [], [], [], [], missing, [{"error": "missing_required_input"}], 0, 0)
        raise SystemExit(2)

    diag_rows = read_tsv(diag_path)
    diag_map = build_diag_map(diag_rows)
    cost_groups, raw_cost_row_count = build_cost_groups(cost_path)
    if raw_cost_row_count == 0:
        write_empty_outputs(output_dir)
        missing_rows = [{"path": str(cost_path), "reason": "local_cost_matrix_has_header_only_or_no_data"}]
        failed_rows = [{"error": "empty_local_cost_matrix", "path": str(cost_path)}]
        write_tsv(output_dir / "missing_records.tsv", missing_rows, ["path", "reason"])
        write_tsv(output_dir / "failed_runs.tsv", failed_rows, ["error", "path"])
        write_summary(output_dir, args, [], [], [], [], [], [], missing_rows, failed_rows, 0, 0)
        print(f"OUTPUT_DIR={output_dir}")
        print("COST_MATRIX_ROWS=0")
        print("FAILED=empty_local_cost_matrix")
        raise SystemExit(2)

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    geometry_rows, shuffle_rows, behavior_rows, normalized_rows, failed_rows = [], [], [], [], []
    valid_class_count = 0
    for idx, (key, rows) in enumerate(sorted(cost_groups.items()), start=1):
        task, seed, config, label_mode, class_id = key
        diag = diag_map.get(key, {})
        source = rows[0].get("source", "")
        target = rows[0].get("target", "")
        shift = safe_int(diag.get("estimated_global_shift"), 0)
        if str(diag.get("valid_class", "1")) == "1":
            valid_class_count += 1
        for metric in METRICS:
            cost = build_cost_matrix(rows, metric)
            if cost is None:
                failed_rows.append({"task": task, "seed": seed, "config": config, "label_mode": label_mode, "class_id": class_id, "distance_metric": metric, "error": "missing_metric_cost_matrix"})
                continue
            geom = cost_geometry(cost, shift, args.radius)
            geometry_rows.append({
                "task": task, "source": source, "target": target, "seed": seed, "config": config, "label_mode": label_mode,
                "class_id": class_id, "distance_metric": metric, **geom, "valid_class": 1, "skip_reason": "",
            })
            real_soft, behavior = soft_alignment(cost, shift, args.radius, args.tau)
            real_mono, path_ratio, mean_abs_dev, num_large = monotonic_alignment(cost, shift, args.radius)
            real_shuffle_gap = None
            diag_key = f"local_soft_real_vs_shuffled_gap_{metric}"
            if diag_key in diag:
                real_shuffle_gap = safe_float(diag.get(diag_key))
            behavior_rows.append({
                "task": task, "seed": seed, "config": config, "label_mode": label_mode, "class_id": class_id,
                "distance_metric": metric, **behavior, "interpretation_flag": interpret_flag(behavior, real_shuffle_gap),
            })
            dist_global = geom.get("global_shift_band_cost")
            dist_local = real_soft
            dist_mono = real_mono
            local_gap_ref = None
            mono_gap_ref = None
            # z-normalized real-vs-shuffled uses a deterministic full-shuffle control.
            finite = cost[np.isfinite(cost)]
            z_cost = (cost - float(np.mean(finite))) / max(float(np.std(finite)), 1e-12) if finite.size else cost
            z_soft, _ = soft_alignment(z_cost, shift, args.radius, args.tau)
            z_mono, _, _, _ = monotonic_alignment(z_cost, shift, args.radius)
            z_perm = z_cost[:, rng.permutation(z_cost.shape[1])]
            z_soft_ctrl, _ = soft_alignment(z_perm, shift, args.radius, args.tau)
            z_mono_ctrl, _, _, _ = monotonic_alignment(z_perm, shift, args.radius)
            local_gap_ref = None if z_soft_ctrl is None or z_soft is None else z_soft_ctrl - z_soft
            mono_gap_ref = None if z_mono_ctrl is None or z_mono is None else z_mono_ctrl - z_mono
            normalized_rows.append({
                "task": task, "seed": seed, "config": config, "label_mode": label_mode, "class_id": class_id,
                "distance_metric": metric,
                "raw_local_soft_improvement": diff(dist_global, dist_local),
                "relative_local_soft_improvement": safe_div(diff(dist_global, dist_local), dist_global),
                "raw_monotonic_improvement": diff(dist_global, dist_mono),
                "relative_monotonic_improvement": safe_div(diff(dist_global, dist_mono), dist_global),
                "raw_local_real_vs_shuffled_gap": safe_float(diag.get(f"local_soft_real_vs_shuffled_gap_{metric}")),
                "relative_local_real_vs_shuffled_gap": safe_div(safe_float(diag.get(f"local_soft_real_vs_shuffled_gap_{metric}")), dist_local),
                "z_normalized_local_real_vs_shuffled_gap": local_gap_ref,
                "raw_monotonic_real_vs_shuffled_gap": safe_float(diag.get(f"monotonic_real_vs_shuffled_gap_{metric}")),
                "relative_monotonic_real_vs_shuffled_gap": safe_div(safe_float(diag.get(f"monotonic_real_vs_shuffled_gap_{metric}")), dist_mono),
                "z_normalized_monotonic_real_vs_shuffled_gap": mono_gap_ref,
                "dist_global_shift": dist_global,
                "dist_local_soft": dist_local,
                "dist_monotonic": dist_mono,
            })
            for control_type in ["full_shuffle", "circular_shift", "block_shuffle", "within_band_random", "time_reversed"]:
                repeats = 1 if control_type == "time_reversed" else args.shuffle_repeats
                for repeat_id in range(repeats):
                    if control_type == "within_band_random":
                        ctrl_soft, ctrl_mean_dev, ctrl_large = random_within_band(cost, shift, args.radius, rng)
                        ctrl_mono = None
                        ctrl_path = None
                    else:
                        ctrl_cost = transformed_cost(cost, control_type, rng, args.block_size)
                        ctrl_soft, _ = soft_alignment(ctrl_cost, shift, args.radius, args.tau)
                        ctrl_mono, ctrl_path, ctrl_mean_dev, ctrl_large = monotonic_alignment(ctrl_cost, shift, args.radius)
                    shuffle_rows.append({
                        "task": task, "seed": seed, "config": config, "label_mode": label_mode, "class_id": class_id,
                        "distance_metric": metric, "control_type": control_type, "repeat_id": repeat_id,
                        "real_local_soft_dist": real_soft,
                        "control_local_soft_dist": ctrl_soft,
                        "real_minus_control_local_soft_gap": diff(ctrl_soft, real_soft),
                        "real_monotonic_dist": real_mono,
                        "control_monotonic_dist": ctrl_mono,
                        "real_minus_control_monotonic_gap": diff(ctrl_mono, real_mono),
                        "control_path_length_ratio": ctrl_path,
                        "control_mean_abs_deviation": ctrl_mean_dev,
                        "control_num_large_jumps": ctrl_large,
                    })
        print(f"V306_DONE|{idx}/{len(cost_groups)}|task={task}|seed={seed}|config={config}|label={label_mode}|class={class_id}", flush=True)

    monotonic_rows = build_monotonic_stability(diag_rows, args, py_rng)
    corr_rows = build_da_correlation(normalized_rows, geometry_rows, behavior_rows, read_tsv(da_path))
    write_tsv(output_dir / "v306_cost_matrix_geometry.tsv", geometry_rows, GEOMETRY_FIELDS)
    write_tsv(output_dir / "v306_shuffle_control_audit.tsv", shuffle_rows, SHUFFLE_FIELDS)
    write_tsv(output_dir / "v306_local_soft_behavior.tsv", behavior_rows, BEHAVIOR_FIELDS)
    write_tsv(output_dir / "v306_monotonic_stability.tsv", monotonic_rows, MONOTONIC_FIELDS)
    write_tsv(output_dir / "v306_normalized_alignment_summary.tsv", normalized_rows, NORMALIZED_FIELDS)
    write_tsv(output_dir / "v306_da_relation_correlation.tsv", corr_rows, CORR_FIELDS)
    write_tsv(output_dir / "missing_records.tsv", [], ["path", "reason"])
    write_tsv(output_dir / "failed_runs.tsv", failed_rows, ["task", "seed", "config", "label_mode", "class_id", "distance_metric", "error"])
    write_summary(output_dir, args, geometry_rows, shuffle_rows, behavior_rows, monotonic_rows, normalized_rows, corr_rows, [], failed_rows, raw_cost_row_count, valid_class_count)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"COST_MATRIX_ROWS={raw_cost_row_count}")
    print(f"VALID_COST_GROUPS={len(cost_groups)}")
    print(f"VALID_CLASS_COUNT={valid_class_count}")
    print(f"FAILED_ROWS={len(failed_rows)}")


def safe_div(a, b):
    if a is None or b is None or abs(b) <= 1e-12:
        return None
    return a / b


def build_monotonic_stability(diag_rows, args, py_rng):
    base = []
    for row in diag_rows:
        if row.get("task") not in TASKS or row.get("config") not in CONFIGS or row.get("label_mode") not in LABEL_MODES:
            continue
        for metric in METRICS:
            val = safe_float(row.get(f"monotonic_real_vs_shuffled_gap_{metric}"))
            if val is None:
                continue
            base.append({**row, "distance_metric": metric, "gap": val})
    out = []
    keys = set((r["task"], r["config"], r["label_mode"], r["distance_metric"]) for r in base)
    for task, config, label_mode, metric in sorted(keys):
        subset = [r for r in base if r["task"] == task and r["config"] == config and r["label_mode"] == label_mode and r["distance_metric"] == metric]
        vals = [r["gap"] for r in subset]
        all_same_mode_metric = [r for r in base if r["label_mode"] == label_mode and r["distance_metric"] == metric]
        excluding_plain = [r["gap"] for r in all_same_mode_metric if r["config"] != "plain"]
        task_means_excluding = []
        for t in sorted(set(r["task"] for r in all_same_mode_metric)):
            tv = [r["gap"] for r in all_same_mode_metric if r["task"] == t and r["config"] != "plain"]
            if tv:
                task_means_excluding.append(mean(tv))
        boot_mean, ci_low, ci_high = bootstrap_ci(vals, np.random.default_rng(args.seed + 17), args.bootstrap_repeats)
        leave_vals = []
        for t in sorted(set(r["task"] for r in all_same_mode_metric)):
            lv = [r["gap"] for r in all_same_mode_metric if r["task"] != t]
            if lv:
                leave_vals.append(mean(lv))
        mean_ex = mean(excluding_plain)
        flag = "stable_temporal_signal"
        if mean_ex is None or abs(mean_ex) <= 1e-4 or (ci_low is not None and ci_high is not None and ci_low <= 0 <= ci_high):
            flag = "plain_dominated_or_unstable"
        elif sum(1 for v in task_means_excluding if v is not None and v > 0) < max(1, math.ceil(len(task_means_excluding) / 2)):
            flag = "task_specific"
        out.append({
            "task": task, "config": config, "label_mode": label_mode, "distance_metric": metric,
            "mean_monotonic_real_vs_shuffled_gap": mean(vals),
            "std_monotonic_real_vs_shuffled_gap": std(vals),
            "positive_seed_count": sum(1 for v in vals if v > 0),
            "n_seed": len(vals),
            "mean_gap_excluding_plain": mean_ex,
            "positive_task_count_excluding_plain": sum(1 for v in task_means_excluding if v is not None and v > 0),
            "bootstrap_mean": boot_mean,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "leave_one_task_min": min(leave_vals) if leave_vals else None,
            "leave_one_task_max": max(leave_vals) if leave_vals else None,
            "stability_flag": flag,
        })
    return out


def build_da_correlation(norm_rows, geometry_rows, behavior_rows, da_rows):
    da_map = {(r.get("task", ""), str(r.get("seed", "")), r.get("config", "")): r for r in da_rows}
    geom_map = {(r["task"], str(r["seed"]), r["config"], r["label_mode"], str(r["class_id"]), r["distance_metric"]): r for r in geometry_rows}
    beh_map = {(r["task"], str(r["seed"]), r["config"], r["label_mode"], str(r["class_id"]), r["distance_metric"]): r for r in behavior_rows}
    merged = []
    for row in norm_rows:
        key = (row["task"], str(row["seed"]), row["config"], row["label_mode"], str(row["class_id"]), row["distance_metric"])
        da = da_map.get((row["task"], str(row["seed"]), row["config"]), {})
        merged.append({**row, **geom_map.get(key, {}), **beh_map.get(key, {}), **da})
    predictors = [
        "relative_local_soft_improvement", "relative_monotonic_improvement",
        "relative_local_real_vs_shuffled_gap", "relative_monotonic_real_vs_shuffled_gap",
        "z_normalized_local_real_vs_shuffled_gap", "z_normalized_monotonic_real_vs_shuffled_gap",
        "low_rank_ratio", "cost_temporal_contrast", "diagonal_band_advantage",
        "alignment_entropy_mean", "effective_candidates_mean", "path_length_ratio",
    ]
    targets = ["da_f1", "source_on_target_f1", "da_gain"]
    rows = []
    for label_mode in LABEL_MODES:
        by_mode = [r for r in merged if r.get("label_mode") == label_mode]
        for scope in ["all_configs", "excluding_plain", "raw_smooth_timeperm_only"]:
            scoped = scope_filter(by_mode, scope)
            for predictor in predictors:
                xs = [r.get(predictor) for r in scoped]
                for target in targets:
                    ys = [r.get(target) for r in scoped]
                    count, rate = same_direction(xs, ys)
                    rows.append({
                        "predictor": predictor, "target_metric": target, "label_mode": label_mode,
                        "config_scope": scope,
                        "n": sum(1 for x, y in zip(xs, ys) if safe_float(x) is not None and safe_float(y) is not None),
                        "pearson": pearson(xs, ys), "spearman": spearman(xs, ys),
                        "same_direction_count": count, "same_direction_rate": rate,
                    })
    return rows


def summarize_control(shuffle_rows, control_type, field):
    vals = [r.get(field) for r in shuffle_rows if r.get("control_type") == control_type]
    return mean(vals)


def write_summary(output_dir, args, geometry_rows, shuffle_rows, behavior_rows, monotonic_rows, normalized_rows, corr_rows, missing_rows, failed_rows, cost_row_count, valid_class_count):
    path = output_dir / "v306_alignment_control_audit_summary.md"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v3.0.6 Alignment Control Audit 总结\n\n")
        handle.write("## 1. 实验目的\n\n")
        handle.write("本轮只解释 v3.0.5 的异常现象：shuffled target time 反而更好，以及 monotonic 的弱正信号是否稳定。不训练新模型，不新增 loss。\n\n")
        handle.write("## 2. 输入与记录数\n\n")
        handle.write(f"- v305_dir: `{args.v305_dir}`\n")
        handle.write(f"- v304_dir: `{args.v304_dir}`\n")
        handle.write(f"- cost matrix 原始行数: {cost_row_count}\n")
        handle.write(f"- 有效 cost geometry 行数: {len(geometry_rows)}\n")
        handle.write(f"- 有效 class 组数: {valid_class_count}\n")
        handle.write(f"- 缺失记录数: {len(missing_rows)}\n")
        handle.write(f"- 失败记录数: {len(failed_rows)}\n\n")
        if cost_row_count == 0:
            handle.write("## 3. 结果\n\n")
            handle.write("`local_cost_matrix.tsv` 没有数据行，本轮不能完成 cost-matrix 依赖的 v3.0.6 审计。请重新生成包含矩阵内容的 v3.0.5 输出，或在服务器上以 `WRITE_COST_MATRIX=True` 重跑 v3.0.5 分析。\n")
            return
        handle.write("## 3. Cost matrix 几何结构\n\n")
        handle.write(f"- 平均 low_rank_ratio: {fmt(mean([r.get('low_rank_ratio') for r in geometry_rows]))}\n")
        handle.write(f"- 平均 cost_temporal_contrast: {fmt(mean([r.get('cost_temporal_contrast') for r in geometry_rows]))}\n")
        handle.write(f"- 平均 row_min_advantage: {fmt(mean([r.get('row_min_advantage') for r in geometry_rows]))}\n\n")
        handle.write("## 4. Shuffle control 审核\n\n")
        for control in ["full_shuffle", "circular_shift", "block_shuffle", "within_band_random", "time_reversed"]:
            handle.write(f"- {control}: local gap={fmt(summarize_control(shuffle_rows, control, 'real_minus_control_local_soft_gap'))}, monotonic gap={fmt(summarize_control(shuffle_rows, control, 'real_minus_control_monotonic_gap'))}\n")
        handle.write("\n## 5. Local soft 行为\n\n")
        handle.write(f"- 平均 effective_candidates: {fmt(mean([r.get('effective_candidates_mean') for r in behavior_rows]))}\n")
        handle.write(f"- 平均 edge_selection_rate: {fmt(mean([r.get('edge_selection_rate') for r in behavior_rows]))}\n")
        handle.write(f"- 平均 best-second gap: {fmt(mean([r.get('cost_gap_best_vs_second_best') for r in behavior_rows]))}\n\n")
        handle.write("## 6. Monotonic 稳定性\n\n")
        non_plain = [r for r in monotonic_rows if r.get("config") != "plain"]
        handle.write(f"- excluding plain 后平均 gap: {fmt(mean([r.get('mean_monotonic_real_vs_shuffled_gap') for r in non_plain]))}\n")
        handle.write(f"- stable_temporal_signal 行数: {sum(1 for r in monotonic_rows if r.get('stability_flag') == 'stable_temporal_signal')}\n\n")
        handle.write("## 7. 归一化后结论\n\n")
        handle.write(f"- 平均 relative local gap: {fmt(mean([r.get('relative_local_real_vs_shuffled_gap') for r in normalized_rows]))}\n")
        handle.write(f"- 平均 z-normalized local gap: {fmt(mean([r.get('z_normalized_local_real_vs_shuffled_gap') for r in normalized_rows]))}\n")
        handle.write(f"- 平均 z-normalized monotonic gap: {fmt(mean([r.get('z_normalized_monotonic_real_vs_shuffled_gap') for r in normalized_rows]))}\n\n")
        handle.write("## 8. 与 DA 的关系\n\n")
        handle.write("相关性只作辅助，不作因果解释。详见 `v306_da_relation_correlation.tsv`。\n\n")
        handle.write("## 9. 对 v3 的影响\n\n")
        handle.write("本节只判断当前诊断是否支持继续时间对齐路线，不宣称任何新方法有效。\n\n")
        handle.write("## 10. 最小结论\n\n")
        z_local = mean([r.get("z_normalized_local_real_vs_shuffled_gap") for r in normalized_rows])
        mono_non_plain = mean([r.get("mean_monotonic_real_vs_shuffled_gap") for r in non_plain])
        if z_local is not None and z_local > 0 and mono_non_plain is not None and mono_non_plain > 0:
            handle.write("当前结果提供一定真实时间结构信号，但仍需结合 shuffle controls 和 local soft 行为判断是否足够进入 v3.1。\n")
        else:
            handle.write("当前结果不足以支持直接进入局部时间对齐训练路线。\n")


if __name__ == "__main__":
    main()
