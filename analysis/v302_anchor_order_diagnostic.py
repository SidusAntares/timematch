#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.v301_anchor_correspondence_diagnostic import (
    bool_value,
    create_train_val_test_folds,
    extract_temporal_features,
    flatten_timepoints,
    load_model,
    maybe_limit_indices,
    resolve_classes,
    sample_rows,
    soft_assign,
    write_json,
    zscore_apply,
    zscore_fit,
)
from dataset import PixelSetData, count_pixelset_samples
from torchvision import transforms
from torch.utils import data
from transforms import Normalize, RandomSamplePixels, ToTensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="v3.0.2 offline diagnostic for same-label temporal anchor order."
    )
    parser.add_argument("--mode", choices=["run", "aggregate"], default="run")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--source", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint_config", default="smooth_k3")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_root", default="")
    parser.add_argument("--da_rows", default="")
    parser.add_argument("--closed_set", default="True")
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_pixels", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--with_extra", default="False")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--max_source_samples", type=int, default=0)
    parser.add_argument("--max_target_samples", type=int, default=0)
    parser.add_argument("--max_codebook_timepoints", type=int, default=120000)
    parser.add_argument("--min_source_samples_per_class", type=int, default=20)
    parser.add_argument("--min_target_samples_per_class", type=int, default=20)
    parser.add_argument("--shuffle_repeats", type=int, default=50)
    parser.add_argument("--rank_permutation_repeats", type=int, default=50)
    parser.add_argument("--dtw_window_ratio", type=float, default=0.2)
    parser.add_argument("--write_sample_sequences", default="False")
    return parser.parse_args()


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    return float(text)


def mean(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def build_loader(args, dataset_name, classes, split_indices, max_samples=0):
    split_indices = maybe_limit_indices(split_indices, max_samples, args.seed)
    transform = transforms.Compose([
        RandomSamplePixels(args.num_pixels),
        Normalize(),
        ToTensor(),
    ])
    dataset = PixelSetData(
        args.data_root,
        dataset_name,
        classes,
        transform,
        indices=split_indices,
        closed_set=bool_value(args.closed_set),
    )
    return data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


def entropy_js(p, q, eps=1e-8):
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * (np.log(p) - np.log(m))).sum() + 0.5 * (q * (np.log(q) - np.log(m))).sum())


def l2_distance(p, q):
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float((a @ b) / denom)


def spearman_corr(a, b):
    return pearson_corr(rankdata(a), rankdata(b))


def kendall_tau_b(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float("nan")
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(len(a) - 1):
        for j in range(i + 1, len(a)):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if abs(da) <= 1e-12 and abs(db) <= 1e-12:
                continue
            if abs(da) <= 1e-12:
                ties_a += 1
                continue
            if abs(db) <= 1e-12:
                ties_b += 1
                continue
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    if denom <= 1e-12:
        return float("nan")
    return float((concordant - discordant) / denom)


def interpolate_to_common_grid(source_times, source_values, target_times, target_values):
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    if len(source_values) < 3 or len(target_values) < 3:
        return None, None
    n_grid = max(3, min(len(source_values), len(target_values)))
    grid = np.linspace(0.0, 1.0, n_grid)
    src_x = np.linspace(0.0, 1.0, len(source_values))
    tgt_x = np.linspace(0.0, 1.0, len(target_values))
    return np.interp(grid, src_x, source_values), np.interp(grid, tgt_x, target_values)


def transition_matrix(sequences, k_value):
    counts = np.zeros((k_value, k_value), dtype=np.float64)
    for seq in sequences:
        seq = list(seq)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[int(a), int(b)] += 1.0
    counts += 1e-8
    return counts / counts.sum(axis=1, keepdims=True)


def constrained_dtw(a, b, window_ratio=0.2):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan"), 0, float("nan"), 0, 0
    window = max(abs(n - m), int(math.ceil(max(n, m) * window_ratio)))
    inf = float("inf")
    dp = np.full((n + 1, m + 1), inf)
    prev = {}
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            choices = [
                (dp[i - 1, j], (i - 1, j)),
                (dp[i, j - 1], (i, j - 1)),
                (dp[i - 1, j - 1], (i - 1, j - 1)),
            ]
            best_val, best_prev = min(choices, key=lambda x: x[0])
            dp[i, j] = abs(a[i - 1] - b[j - 1]) + best_val
            prev[(i, j)] = best_prev
    if math.isinf(dp[n, m]):
        return float("nan"), 0, float("nan"), 0, 0
    path = []
    cur = (n, m)
    while cur != (0, 0) and cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    steps = []
    large_jumps = 0
    last = (0, 0)
    for p in path:
        step = max(abs(p[0] - last[0]), abs(p[1] - last[1]))
        steps.append(step)
        if step > 1:
            large_jumps += 1
        last = p
    path_len = len(path)
    return (
        float(dp[n, m] / max(path_len, 1)),
        path_len,
        float(path_len / max(n, m)),
        float(np.mean(steps)) if steps else float("nan"),
        large_jumps,
    )


def class_name(classes, class_id):
    if class_id is None or int(class_id) < 0 or int(class_id) >= len(classes):
        return ""
    return classes[int(class_id)]


def make_sequence_rows(args, domain, data_obj, q, hard, ranks):
    n, t = data_obj["positions"].shape
    q3 = q.reshape(n, t, q.shape[-1])
    hard2 = hard.reshape(n, t)
    rows = []
    rank_rows = []
    for i in range(n):
        label = int(data_obj["labels"][i])
        sample_id = int(data_obj["indices"][i])
        for j in range(t):
            dist = q3[i, j]
            anchor_id = int(hard2[i, j])
            base = {
                "task": args.task,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "checkpoint_config": args.checkpoint_config,
                "domain": domain,
                "split": "train",
                "sample_id": sample_id,
                "label_if_available": label,
                "time_index": j,
                "time_value": int(data_obj["positions"][i, j]),
                "hard_anchor_id": anchor_id,
            }
            rows.append({
                **base,
                "max_anchor_prob": float(dist.max()),
                "anchor_distribution_json": json.dumps([float(x) for x in dist]),
            })
            rank_rows.append({
                **{k: base[k] for k in [
                    "task", "source", "target", "seed", "checkpoint_config",
                    "domain", "sample_id", "label_if_available", "time_index",
                    "time_value", "hard_anchor_id",
                ]},
                "anchor_time_rank": int(ranks[anchor_id]),
            })
    return rows, rank_rows


def source_anchor_ranks(args, source_positions, hard_source):
    flat_pos = source_positions.reshape(-1)
    counts = np.bincount(hard_source, minlength=args.K).astype(np.float64)
    mean_times = []
    rows = []
    for anchor_id in range(args.K):
        mask = hard_source == anchor_id
        mean_t = float(flat_pos[mask].mean()) if mask.any() else float("inf")
        mean_times.append(mean_t)
    order = np.argsort(mean_times)
    ranks = np.empty(args.K, dtype=np.int64)
    for rank, anchor_id in enumerate(order, start=1):
        ranks[anchor_id] = rank
    for anchor_id in range(args.K):
        rows.append({
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            "anchor_id": anchor_id,
            "mean_source_time": None if math.isinf(mean_times[anchor_id]) else mean_times[anchor_id],
            "anchor_time_rank": int(ranks[anchor_id]),
            "source_anchor_count": int(counts[anchor_id]),
            "source_anchor_fraction": float(counts[anchor_id] / max(counts.sum(), 1.0)),
        })
    return ranks, rows


def class_trajectory_rows(args, classes, domain, data_obj, hard, ranks):
    n, t = data_obj["positions"].shape
    hard2 = hard.reshape(n, t)
    rows = []
    by_class = {}
    for class_id in sorted(set(int(x) for x in data_obj["labels"].tolist())):
        mask = data_obj["labels"] == class_id
        positions = data_obj["positions"][mask]
        anchors = hard2[mask]
        rank_values = ranks[anchors]
        by_time = defaultdict(list)
        anchor_by_time = defaultdict(list)
        for i in range(rank_values.shape[0]):
            for j in range(rank_values.shape[1]):
                tm = int(positions[i, j])
                by_time[tm].append(float(rank_values[i, j]))
                anchor_by_time[tm].append(int(anchors[i, j]))
        traj = []
        for tm in sorted(by_time):
            vals = np.asarray(by_time[tm], dtype=np.float64)
            anchor_counts = np.bincount(anchor_by_time[tm], minlength=args.K).astype(np.float64)
            dist = anchor_counts / max(anchor_counts.sum(), 1.0)
            row = {
                "task": args.task,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "checkpoint_config": args.checkpoint_config,
                "class_id": class_id,
                "class_name_if_available": class_name(classes, class_id),
                "domain": domain,
                "time_index": len(traj),
                "time_value": tm,
                "n_samples": int(mask.sum()),
                "mean_anchor_rank": float(vals.mean()),
                "median_anchor_rank": float(np.median(vals)),
                "rank_std": float(vals.std()),
                "anchor_distribution_json": json.dumps([float(x) for x in dist]),
            }
            traj.append(row)
            rows.append(row)
        by_class[class_id] = {
            "n_samples": int(mask.sum()),
            "positions": positions,
            "anchors": anchors,
            "rank_values": rank_values,
            "trajectory_rows": traj,
        }
    return rows, by_class


def trajectory_arrays(rows):
    rows = sorted(rows, key=lambda r: (safe_float(r["time_value"]), safe_float(r["time_index"])))
    return (
        np.asarray([safe_float(r["time_value"]) for r in rows], dtype=np.float64),
        np.asarray([safe_float(r["mean_anchor_rank"]) for r in rows], dtype=np.float64),
    )


def shuffle_values(values, rng):
    values = np.asarray(values, dtype=np.float64).copy()
    rng.shuffle(values)
    return values


def valid_pair(args, source_info, target_info):
    if source_info is None or target_info is None:
        return False, "missing_class"
    if source_info["n_samples"] < args.min_source_samples_per_class:
        return False, "too_few_source_samples"
    if target_info["n_samples"] < args.min_target_samples_per_class:
        return False, "too_few_target_samples"
    if len(source_info["trajectory_rows"]) < 3 or len(target_info["trajectory_rows"]) < 3:
        return False, "too_few_time_bins"
    return True, ""


def run_class_diagnostics(args, classes, source_by_class, target_by_class, ranks):
    order_rows = []
    perm_rows = []
    transition_rows = []
    dtw_rows = []
    summary_rows = []
    rng = np.random.default_rng(args.seed + 2027)
    all_classes = sorted(set(source_by_class) | set(target_by_class))
    for class_id in all_classes:
        src = source_by_class.get(class_id)
        tgt = target_by_class.get(class_id)
        valid, reason = valid_pair(args, src, tgt)
        base = {
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            "class_id": class_id,
            "class_name_if_available": class_name(classes, class_id),
            "n_source_samples": src["n_samples"] if src else 0,
            "n_target_samples": tgt["n_samples"] if tgt else 0,
            "n_time_bins": min(len(src["trajectory_rows"]) if src else 0, len(tgt["trajectory_rows"]) if tgt else 0),
        }
        if not valid:
            summary_rows.append({**base, "valid_class": 0, "skip_reason": reason})
            continue
        src_times, src_values = trajectory_arrays(src["trajectory_rows"])
        tgt_times, tgt_values = trajectory_arrays(tgt["trajectory_rows"])
        src_interp, tgt_interp = interpolate_to_common_grid(src_times, src_values, tgt_times, tgt_values)
        order_real = kendall_tau_b(src_interp, tgt_interp)
        shuffled_corrs = [
            kendall_tau_b(src_interp, shuffle_values(tgt_interp, rng))
            for _ in range(args.shuffle_repeats)
        ]
        order_shuf_mean = mean(shuffled_corrs)
        order_shuf_std = float(np.nanstd(shuffled_corrs)) if shuffled_corrs else float("nan")
        order_gap = order_real - order_shuf_mean if order_shuf_mean is not None else None
        order_rows.append({
            **base,
            "order_corr_real": order_real,
            "order_corr_shuffled_mean": order_shuf_mean,
            "order_corr_shuffled_std": order_shuf_std,
            "order_gap": order_gap,
            "order_corr_metric": "kendall_tau_b_on_source_anchor_time_rank",
            "shuffle_repeats": args.shuffle_repeats,
        })

        permuted_corrs = []
        for _ in range(args.rank_permutation_repeats):
            perm = np.arange(1, args.K + 1)
            rng.shuffle(perm)
            rank_map = {anchor_id: perm[anchor_id] for anchor_id in range(args.K)}
            src_perm = permuted_mean_trajectory(src["anchors"], src["positions"], rank_map)
            tgt_perm = permuted_mean_trajectory(tgt["anchors"], tgt["positions"], rank_map)
            src_p_times, src_p_vals = src_perm
            tgt_p_times, tgt_p_vals = tgt_perm
            src_p, tgt_p = interpolate_to_common_grid(src_p_times, src_p_vals, tgt_p_times, tgt_p_vals)
            if src_p is not None:
                permuted_corrs.append(kendall_tau_b(src_p, tgt_p))
        perm_mean = mean(permuted_corrs)
        perm_std = float(np.nanstd(permuted_corrs)) if permuted_corrs else float("nan")
        perm_gap = order_real - perm_mean if perm_mean is not None else None
        perm_rows.append({
            **base,
            "order_corr_real": order_real,
            "order_corr_rank_permuted_mean": perm_mean,
            "order_corr_rank_permuted_std": perm_std,
            "rank_permutation_gap": perm_gap,
            "rank_permutation_repeats": args.rank_permutation_repeats,
        })

        src_trans = transition_matrix(src["anchors"], args.K)
        tgt_trans = transition_matrix(tgt["anchors"], args.K)
        trans_js = entropy_js(src_trans, tgt_trans)
        trans_l2 = l2_distance(src_trans, tgt_trans)
        trans_js_shuf = []
        trans_l2_shuf = []
        for _ in range(args.shuffle_repeats):
            shuffled = np.asarray(tgt["anchors"]).copy()
            for row in shuffled:
                rng.shuffle(row)
            shuf_trans = transition_matrix(shuffled, args.K)
            trans_js_shuf.append(entropy_js(src_trans, shuf_trans))
            trans_l2_shuf.append(l2_distance(src_trans, shuf_trans))
        trans_js_shuf_mean = mean(trans_js_shuf)
        trans_l2_shuf_mean = mean(trans_l2_shuf)
        trans_gap = trans_js_shuf_mean - trans_js if trans_js_shuf_mean is not None else None
        trans_l2_gap = trans_l2_shuf_mean - trans_l2 if trans_l2_shuf_mean is not None else None
        transition_rows.append({
            **base,
            "transition_js_real": trans_js,
            "transition_js_shuffled_mean": trans_js_shuf_mean,
            "transition_js_shuffled_std": float(np.nanstd(trans_js_shuf)) if trans_js_shuf else float("nan"),
            "transition_gap": trans_gap,
            "transition_l2_real": trans_l2,
            "transition_l2_shuffled_mean": trans_l2_shuf_mean,
            "transition_l2_shuffled_std": float(np.nanstd(trans_l2_shuf)) if trans_l2_shuf else float("nan"),
            "transition_l2_gap": trans_l2_gap,
            "source_transition_matrix_json": json.dumps(src_trans.round(6).tolist()),
            "target_transition_matrix_json": json.dumps(tgt_trans.round(6).tolist()),
            "shuffle_repeats": args.shuffle_repeats,
        })

        dtw_real, path_len, path_ratio, mean_step, large_jumps = constrained_dtw(
            src_interp, tgt_interp, args.dtw_window_ratio
        )
        dtw_shuf = []
        for _ in range(args.shuffle_repeats):
            shuffled = shuffle_values(tgt_interp, rng)
            value, *_ = constrained_dtw(src_interp, shuffled, args.dtw_window_ratio)
            dtw_shuf.append(value)
        dtw_shuf_mean = mean(dtw_shuf)
        dtw_gap = dtw_shuf_mean - dtw_real if dtw_shuf_mean is not None else None
        dtw_rows.append({
            **base,
            "dtw_real": dtw_real,
            "dtw_shuffled_mean": dtw_shuf_mean,
            "dtw_shuffled_std": float(np.nanstd(dtw_shuf)) if dtw_shuf else float("nan"),
            "dtw_gap": dtw_gap,
            "dtw_window_ratio": args.dtw_window_ratio,
            "dtw_path_length": path_len,
            "dtw_path_length_ratio": path_ratio,
            "dtw_mean_step_size": mean_step,
            "dtw_max_step_size": 1,
            "dtw_num_large_jumps": large_jumps,
            "shuffle_repeats": args.shuffle_repeats,
        })

        summary_rows.append({
            **base,
            "order_corr_real": order_real,
            "order_corr_shuffled_mean": order_shuf_mean,
            "order_gap": order_gap,
            "rank_permutation_gap": perm_gap,
            "transition_js_real": trans_js,
            "transition_js_shuffled_mean": trans_js_shuf_mean,
            "transition_gap": trans_gap,
            "dtw_real": dtw_real,
            "dtw_shuffled_mean": dtw_shuf_mean,
            "dtw_gap": dtw_gap,
            "dtw_path_length_ratio": path_ratio,
            "valid_class": 1,
            "skip_reason": "",
        })
    return order_rows, perm_rows, transition_rows, dtw_rows, summary_rows


def permuted_mean_trajectory(anchors, positions, rank_map):
    by_time = defaultdict(list)
    for i in range(anchors.shape[0]):
        for j in range(anchors.shape[1]):
            by_time[int(positions[i, j])].append(float(rank_map[int(anchors[i, j])]))
    times = sorted(by_time)
    values = [float(np.mean(by_time[t])) for t in times]
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def weighted_task_summary(class_rows):
    valid_rows = [r for r in class_rows if str(r.get("valid_class")) in {"1", "1.0"}]
    if not valid_rows:
        return {}
    weights = np.asarray([
        min(safe_float(r.get("n_source_samples")) or 0, safe_float(r.get("n_target_samples")) or 0)
        for r in valid_rows
    ], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(valid_rows), dtype=np.float64)
    def wmean(field):
        vals = np.asarray([safe_float(r.get(field)) if safe_float(r.get(field)) is not None else np.nan for r in valid_rows])
        mask = ~np.isnan(vals)
        if not mask.any():
            return None
        return float(np.average(vals[mask], weights=weights[mask]))
    return {
        "n_valid_classes": len(valid_rows),
        "weighted_order_corr_real": wmean("order_corr_real"),
        "weighted_order_corr_shuffled": wmean("order_corr_shuffled_mean"),
        "weighted_order_gap": wmean("order_gap"),
        "weighted_rank_permutation_gap": wmean("rank_permutation_gap"),
        "weighted_transition_js_real": wmean("transition_js_real"),
        "weighted_transition_js_shuffled": wmean("transition_js_shuffled_mean"),
        "weighted_transition_gap": wmean("transition_gap"),
        "weighted_dtw_real": wmean("dtw_real"),
        "weighted_dtw_shuffled": wmean("dtw_shuffled_mean"),
        "weighted_dtw_gap": wmean("dtw_gap"),
        "weighted_dtw_path_length_ratio": wmean("dtw_path_length_ratio"),
        "positive_order_gap_classes": sum((safe_float(r.get("order_gap")) or 0) > 0 for r in valid_rows),
        "positive_transition_gap_classes": sum((safe_float(r.get("transition_gap")) or 0) > 0 for r in valid_rows),
        "positive_dtw_gap_classes": sum((safe_float(r.get("dtw_gap")) or 0) > 0 for r in valid_rows),
        "total_valid_classes": len(valid_rows),
    }


def run_one(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_rows = []
    try:
        classes = resolve_classes(args.data_root, args.source, bool_value(args.closed_set))
        indices = {
            args.source: count_pixelset_samples(args.data_root, args.source, classes, closed_set=bool_value(args.closed_set)),
            args.target: count_pixelset_samples(args.data_root, args.target, classes, closed_set=bool_value(args.closed_set)),
        }
        splits = create_train_val_test_folds(
            [args.source, args.target],
            args.num_folds,
            indices,
            args.val_ratio,
            args.test_ratio,
        )[0]
        source_loader = build_loader(args, args.source, classes, splits[args.source]["train"], args.max_source_samples)
        target_loader = build_loader(args, args.target, classes, splits[args.target]["train"], args.max_target_samples)
        model, checkpoint = load_model(args, classes)
        source_data = extract_temporal_features(model, source_loader, args.device)
        target_data = extract_temporal_features(model, target_loader, args.device)
        source_flat_raw = flatten_timepoints(source_data["features"], source_data["positions"], source_data["labels"])
        mean_vec, std_vec = zscore_fit(source_flat_raw["features"])
        source_features = zscore_apply(source_data["features"], mean_vec, std_vec)
        target_features = zscore_apply(target_data["features"], mean_vec, std_vec)
        source_flat = flatten_timepoints(source_features, source_data["positions"], source_data["labels"])
        target_flat = flatten_timepoints(target_features, target_data["positions"], target_data["labels"])
        codebook_idx = sample_rows(source_flat["features"].shape[0], args.max_codebook_timepoints, args.seed)
        kmeans = KMeans(
            n_clusters=args.K,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
            random_state=args.seed,
        )
        kmeans.fit(source_flat["features"][codebook_idx])
        centroids = kmeans.cluster_centers_.astype(np.float64)
        source_d2 = ((source_flat["features"][codebook_idx] - centroids[kmeans.labels_]) ** 2).sum(axis=1)
        tau = max(float(np.median(source_d2)), 1e-6)
        q_source = soft_assign(source_flat["features"], centroids, tau)
        q_target = soft_assign(target_flat["features"], centroids, tau)
        hard_source = q_source.argmax(axis=1)
        hard_target = q_target.argmax(axis=1)
        ranks, rank_rows = source_anchor_ranks(args, source_data["positions"], hard_source)
        source_seq_rows, source_rank_seq_rows = make_sequence_rows(args, "source", source_data, q_source, hard_source, ranks)
        target_seq_rows, target_rank_seq_rows = make_sequence_rows(args, "target", target_data, q_target, hard_target, ranks)
        source_traj_rows, source_by_class = class_trajectory_rows(args, classes, "source", source_data, hard_source, ranks)
        target_traj_rows, target_by_class = class_trajectory_rows(args, classes, "target", target_data, hard_target, ranks)
        order_rows, perm_rows, trans_rows, dtw_rows, class_summary_rows = run_class_diagnostics(
            args, classes, source_by_class, target_by_class, ranks
        )
        task_summary = {
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            **weighted_task_summary(class_summary_rows),
        }
        write_tsv(output_dir / "anchor_time_rank.tsv", rank_rows, ANCHOR_RANK_FIELDS)
        if bool_value(args.write_sample_sequences):
            write_tsv(output_dir / "sample_anchor_sequence.tsv", source_seq_rows + target_seq_rows, SAMPLE_SEQUENCE_FIELDS)
            write_tsv(output_dir / "sample_anchor_rank_sequence.tsv", source_rank_seq_rows + target_rank_seq_rows, SAMPLE_RANK_FIELDS)
        write_tsv(output_dir / "class_anchor_rank_trajectory.tsv", source_traj_rows + target_traj_rows, CLASS_TRAJ_FIELDS)
        write_tsv(output_dir / "class_order_diagnostic.tsv", order_rows, CLASS_ORDER_FIELDS)
        write_tsv(output_dir / "anchor_rank_permutation_control.tsv", perm_rows, RANK_PERM_FIELDS)
        write_tsv(output_dir / "class_transition_diagnostic.tsv", trans_rows, CLASS_TRANSITION_FIELDS)
        write_tsv(output_dir / "class_dtw_diagnostic.tsv", dtw_rows, CLASS_DTW_FIELDS)
        write_tsv(output_dir / "class_order_transition_summary.tsv", class_summary_rows, CLASS_SUMMARY_FIELDS)
        write_tsv(output_dir / "task_order_transition_summary.tsv", [task_summary], TASK_SUMMARY_FIELDS)
        write_json(output_dir / "feature_extraction_meta.json", {
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            "feature_stage": "model.spatial_encoder raw temporal feature",
            "checkpoint_path": str(checkpoint),
            "K": args.K,
            "tau": tau,
            "order_corr_metric": "kendall_tau_b_on_source_anchor_time_rank",
            "dtw_window_ratio": args.dtw_window_ratio,
            "target_label_use": "offline diagnostic only",
        })
        write_tsv(output_dir / "failed_runs.tsv", [], FAILED_FIELDS)
    except Exception as exc:
        failed_rows.append({
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            "status": "failed",
            "error": repr(exc),
        })
        write_tsv(output_dir / "failed_runs.tsv", failed_rows, FAILED_FIELDS)
        raise


SAMPLE_SEQUENCE_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "domain", "split",
    "sample_id", "label_if_available", "time_index", "time_value", "hard_anchor_id",
    "max_anchor_prob", "anchor_distribution_json",
]
ANCHOR_RANK_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "anchor_id",
    "mean_source_time", "anchor_time_rank", "source_anchor_count", "source_anchor_fraction",
]
SAMPLE_RANK_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "domain", "sample_id",
    "label_if_available", "time_index", "time_value", "hard_anchor_id", "anchor_time_rank",
]
CLASS_TRAJ_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "class_id",
    "class_name_if_available", "domain", "time_index", "time_value", "n_samples",
    "mean_anchor_rank", "median_anchor_rank", "rank_std", "anchor_distribution_json",
]
CLASS_ORDER_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "class_id",
    "class_name_if_available", "n_source_samples", "n_target_samples", "n_time_bins",
    "order_corr_real", "order_corr_shuffled_mean", "order_corr_shuffled_std",
    "order_gap", "order_corr_metric", "shuffle_repeats",
]
RANK_PERM_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "class_id",
    "class_name_if_available", "order_corr_real", "order_corr_rank_permuted_mean",
    "order_corr_rank_permuted_std", "rank_permutation_gap", "rank_permutation_repeats",
]
CLASS_TRANSITION_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "class_id",
    "class_name_if_available", "n_source_samples", "n_target_samples",
    "transition_js_real", "transition_js_shuffled_mean", "transition_js_shuffled_std",
    "transition_gap", "transition_l2_real", "transition_l2_shuffled_mean",
    "transition_l2_shuffled_std", "transition_l2_gap", "source_transition_matrix_json",
    "target_transition_matrix_json", "shuffle_repeats",
]
CLASS_DTW_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "class_id",
    "class_name_if_available", "dtw_real", "dtw_shuffled_mean", "dtw_shuffled_std",
    "dtw_gap", "dtw_window_ratio", "dtw_path_length", "dtw_path_length_ratio",
    "dtw_mean_step_size", "dtw_max_step_size", "dtw_num_large_jumps", "shuffle_repeats",
]
CLASS_SUMMARY_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "class_id",
    "class_name_if_available", "n_source_samples", "n_target_samples", "n_time_bins",
    "order_corr_real", "order_corr_shuffled_mean", "order_gap", "rank_permutation_gap",
    "transition_js_real", "transition_js_shuffled_mean", "transition_gap", "dtw_real",
    "dtw_shuffled_mean", "dtw_gap", "dtw_path_length_ratio", "valid_class", "skip_reason",
]
TASK_SUMMARY_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "n_valid_classes",
    "weighted_order_corr_real", "weighted_order_corr_shuffled", "weighted_order_gap",
    "weighted_rank_permutation_gap", "weighted_transition_js_real",
    "weighted_transition_js_shuffled", "weighted_transition_gap", "weighted_dtw_real",
    "weighted_dtw_shuffled", "weighted_dtw_gap", "weighted_dtw_path_length_ratio",
    "positive_order_gap_classes", "positive_transition_gap_classes",
    "positive_dtw_gap_classes", "total_valid_classes",
]
FAILED_FIELDS = ["task", "source", "target", "seed", "checkpoint_config", "status", "error"]


def collect_files(input_root, filename):
    return sorted(Path(input_root).glob(f"*/*/{filename}")) + sorted(Path(input_root).glob(f"*/{filename}"))


def aggregate_files(input_root, output_dir, filename, fields):
    rows = []
    for path in collect_files(input_root, filename):
        rows.extend(read_tsv(path))
    write_tsv(Path(output_dir) / filename, rows, fields)
    return rows


def direction(value):
    value = safe_float(value)
    if value is None:
        return "missing"
    if value > 1e-9:
        return "up"
    if value < -1e-9:
        return "down"
    return "flat"


def canonical_config(config):
    if config in {"v276_smooth_k3_w1", "smooth_k3"}:
        return "smooth_k3"
    if config == "plain":
        return "plain"
    return config


def load_da_rows(path):
    rows = []
    if not path:
        return rows
    for row in read_tsv(path):
        cfg = canonical_config(row.get("config", ""))
        if cfg not in {"plain", "smooth_k3"}:
            continue
        if row.get("status", "ok") not in {"", "ok"}:
            continue
        rows.append({**row, "checkpoint_config": cfg})
    return rows


def make_smooth_plain_relation(task_rows, da_rows, output_dir):
    by_task_cfg = defaultdict(list)
    for row in task_rows:
        by_task_cfg[(row["task"], canonical_config(row["checkpoint_config"]))].append(row)
    da_by_task_cfg = defaultdict(list)
    for row in da_rows:
        by_task_cfg_key = (row["task"], canonical_config(row["checkpoint_config"]))
        da_by_task_cfg[by_task_cfg_key].append(row)
    tasks = sorted(set(task for task, _ in by_task_cfg) | set(task for task, _ in da_by_task_cfg))
    rows = []
    for task in tasks:
        plain = by_task_cfg.get((task, "plain"), [])
        smooth = by_task_cfg.get((task, "smooth_k3"), [])
        plain_da = da_by_task_cfg.get((task, "plain"), [])
        smooth_da = da_by_task_cfg.get((task, "smooth_k3"), [])
        def delta(field, left=smooth, right=plain):
            left_value = mean([r.get(field) for r in left])
            right_value = mean([r.get(field) for r in right])
            if left_value is None or right_value is None:
                return None
            return left_value - right_value
        smooth_da_f1 = mean([r.get("da_f1") for r in smooth_da])
        plain_da_f1 = mean([r.get("da_f1") for r in plain_da])
        smooth_sot = mean([r.get("source_on_target_f1") for r in smooth_da])
        plain_sot = mean([r.get("source_on_target_f1") for r in plain_da])
        delta_da = None if smooth_da_f1 is None or plain_da_f1 is None else smooth_da_f1 - plain_da_f1
        delta_sot = None if smooth_sot is None or plain_sot is None else smooth_sot - plain_sot
        d_order = delta("weighted_order_gap")
        d_trans = delta("weighted_transition_gap")
        d_dtw = delta("weighted_dtw_gap")
        row = {
            "task": task,
            "source": plain[0].get("source") if plain else (smooth[0].get("source") if smooth else ""),
            "target": plain[0].get("target") if plain else (smooth[0].get("target") if smooth else ""),
            "delta_da_f1": delta_da,
            "delta_source_on_target": delta_sot,
            "delta_weighted_order_gap": d_order,
            "delta_weighted_transition_gap": d_trans,
            "delta_weighted_dtw_gap": d_dtw,
            "delta_positive_order_gap_classes": delta("positive_order_gap_classes"),
            "delta_positive_transition_gap_classes": delta("positive_transition_gap_classes"),
            "delta_positive_dtw_gap_classes": delta("positive_dtw_gap_classes"),
            "direction_da": direction(delta_da),
            "direction_order": direction(d_order),
            "direction_transition": direction(d_trans),
            "direction_dtw": direction(d_dtw),
            "same_direction_da_order": int(direction(delta_da) == direction(d_order)),
            "same_direction_da_transition": int(direction(delta_da) == direction(d_trans)),
            "same_direction_da_dtw": int(direction(delta_da) == direction(d_dtw)),
        }
        dirs = {row["direction_da"], row["direction_order"], row["direction_transition"]}
        if row["direction_da"] == "up" and row["direction_order"] == "up" and row["direction_transition"] == "up":
            label = "da_up_order_up_transition_up"
        elif row["direction_da"] == "up" and not (row["direction_order"] == "up" or row["direction_transition"] == "up"):
            label = "da_up_order_not_up"
        elif row["direction_da"] == "down" and (row["direction_order"] == "up" or row["direction_transition"] == "up"):
            label = "da_down_order_up"
        elif row["direction_da"] == "down" and row["direction_order"] == "down" and row["direction_transition"] == "down":
            label = "da_down_order_down"
        else:
            label = "mixed"
        row["interpretation_label"] = label
        rows.append(row)
    fields = [
        "task", "source", "target", "delta_da_f1", "delta_source_on_target",
        "delta_weighted_order_gap", "delta_weighted_transition_gap", "delta_weighted_dtw_gap",
        "delta_positive_order_gap_classes", "delta_positive_transition_gap_classes",
        "delta_positive_dtw_gap_classes", "direction_da", "direction_order",
        "direction_transition", "direction_dtw", "same_direction_da_order",
        "same_direction_da_transition", "same_direction_da_dtw", "interpretation_label",
    ]
    write_tsv(Path(output_dir) / "smooth_plain_order_relation_summary.tsv", rows, fields)
    return rows


def write_summary_legacy_unused(output_dir, task_rows, relation_rows, failed_rows):
    by_cfg = defaultdict(list)
    for row in task_rows:
        by_cfg[canonical_config(row["checkpoint_config"])].append(row)
    lines = []
    lines.append("# v3.0.2 时间锚点顺序诊断总结\n\n")
    lines.append("## 1. 实验设置\n\n")
    lines.append("- version: `v302_anchor_order_diagnostic`\n")
    lines.append("- feature stage: `model.spatial_encoder` raw temporal feature\n")
    lines.append("- K: 6\n")
    lines.append("- class filtering: source/target each >= 20 samples\n")
    lines.append(f"- failed runs: {len(failed_rows)}\n\n")
    lines.append("## 2. 本轮问题\n\n")
    lines.append("本轮只验证同标签 source-target 时间锚点序列是否保留真实时间顺序信息；不训练新方法，不新增 loss。\n\n")
    lines.append("## 3. 任务级顺序与转移诊断\n\n")
    lines.append("| config | n | order gap | rank permutation gap | transition gap | DTW gap | path length ratio |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for cfg, rows in sorted(by_cfg.items()):
        lines.append(
            f"| {cfg} | {len(rows)} | {fmt(mean([r.get('weighted_order_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_rank_permutation_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_transition_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_dtw_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_dtw_path_length_ratio') for r in rows]))} |\n"
        )
    lines.append("\n## 4. smooth vs plain 与 DA 的关系\n\n")
    lines.append("| task | ΔDA | Δsource-on-target | Δorder gap | Δtransition gap | ΔDTW gap | label |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    for row in relation_rows:
        lines.append(
            f"| {row['task'].replace('_to_', '->')} | {fmt(row.get('delta_da_f1'))} | "
            f"{fmt(row.get('delta_source_on_target'))} | {fmt(row.get('delta_weighted_order_gap'))} | "
            f"{fmt(row.get('delta_weighted_transition_gap'))} | {fmt(row.get('delta_weighted_dtw_gap'))} | "
            f"{row.get('interpretation_label')} |\n"
        )
    lines.append("\n## 5. 最小结论\n\n")
    lines.append("这里只报告数据判断，不提出新方法，不声称理论已成立。DTW 仅作为辅助指标。\n")
    (Path(output_dir) / "v302_anchor_order_diagnostic_summary.md").write_text("".join(lines), encoding="utf-8")


def write_summary(output_dir, task_rows, relation_rows, failed_rows):
    by_cfg = defaultdict(list)
    for row in task_rows:
        by_cfg[canonical_config(row["checkpoint_config"])].append(row)
    lines = []
    lines.append("# v3.0.2 时间锚点顺序诊断总结\n\n")
    lines.append("## 1. 实验设置\n\n")
    lines.append("- version: `v302_anchor_order_diagnostic`\n")
    lines.append("- feature stage: `model.spatial_encoder` raw temporal feature\n")
    lines.append("- K: 6\n")
    lines.append("- class filtering: source/target each >= 20 samples\n")
    lines.append(f"- failed runs: {len(failed_rows)}\n\n")
    lines.append("## 2. 本轮问题\n\n")
    lines.append("本轮只验证同标签 source-target 时间锚点序列是否保留真实时间顺序信息；不训练新方法，不新增 loss。\n\n")
    lines.append("## 3. 任务级顺序与转移诊断\n\n")
    lines.append("| config | n | order gap | rank permutation gap | transition gap | DTW gap | path length ratio |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for cfg, rows in sorted(by_cfg.items()):
        lines.append(
            f"| {cfg} | {len(rows)} | {fmt(mean([r.get('weighted_order_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_rank_permutation_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_transition_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_dtw_gap') for r in rows]))} | "
            f"{fmt(mean([r.get('weighted_dtw_path_length_ratio') for r in rows]))} |\n"
        )
    lines.append("\n## 4. smooth vs plain 与 DA 的关系\n\n")
    lines.append("| task | ΔDA | Δsource-on-target | Δorder gap | Δtransition gap | ΔDTW gap | label |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    for row in relation_rows:
        lines.append(
            f"| {row['task'].replace('_to_', '->')} | {fmt(row.get('delta_da_f1'))} | "
            f"{fmt(row.get('delta_source_on_target'))} | {fmt(row.get('delta_weighted_order_gap'))} | "
            f"{fmt(row.get('delta_weighted_transition_gap'))} | {fmt(row.get('delta_weighted_dtw_gap'))} | "
            f"{row.get('interpretation_label')} |\n"
        )
    lines.append("\n## 5. 最小结论\n\n")
    lines.append("这里只报告数据判断，不提出新方法，不声称理论已成立。DTW 仅作为辅助指标。\n")
    (Path(output_dir) / "v302_anchor_order_diagnostic_summary.md").write_text("".join(lines), encoding="utf-8")


def aggregate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = args.input_root or str(output_dir / "runs")
    aggregate_specs = [
        ("anchor_time_rank.tsv", ANCHOR_RANK_FIELDS),
        ("class_anchor_rank_trajectory.tsv", CLASS_TRAJ_FIELDS),
        ("class_order_diagnostic.tsv", CLASS_ORDER_FIELDS),
        ("anchor_rank_permutation_control.tsv", RANK_PERM_FIELDS),
        ("class_transition_diagnostic.tsv", CLASS_TRANSITION_FIELDS),
        ("class_dtw_diagnostic.tsv", CLASS_DTW_FIELDS),
        ("class_order_transition_summary.tsv", CLASS_SUMMARY_FIELDS),
        ("task_order_transition_summary.tsv", TASK_SUMMARY_FIELDS),
    ]
    if bool_value(args.write_sample_sequences):
        aggregate_specs = [
            ("sample_anchor_sequence.tsv", SAMPLE_SEQUENCE_FIELDS),
            ("sample_anchor_rank_sequence.tsv", SAMPLE_RANK_FIELDS),
        ] + aggregate_specs
    for filename, fields in aggregate_specs:
        aggregate_files(input_root, output_dir, filename, fields)
    failed_rows = aggregate_files(input_root, output_dir, "failed_runs.tsv", FAILED_FIELDS)
    failed_rows = [r for r in failed_rows if r.get("status")]
    write_tsv(output_dir / "failed_runs.tsv", failed_rows, FAILED_FIELDS)
    task_rows = read_tsv(output_dir / "task_order_transition_summary.tsv")
    relation_rows = make_smooth_plain_relation(task_rows, load_da_rows(args.da_rows), output_dir)
    write_summary(output_dir, task_rows, relation_rows, failed_rows)


def main():
    args = parse_args()
    if args.mode == "aggregate":
        aggregate(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
