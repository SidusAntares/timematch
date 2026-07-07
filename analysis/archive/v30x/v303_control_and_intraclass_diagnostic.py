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
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.v302_anchor_order_diagnostic import (
    CLASS_DTW_FIELDS,
    CLASS_ORDER_FIELDS,
    CLASS_SUMMARY_FIELDS,
    CLASS_TRANSITION_FIELDS,
    RANK_PERM_FIELDS,
    TASK_SUMMARY_FIELDS,
    bool_value,
    build_loader,
    class_name,
    class_trajectory_rows,
    constrained_dtw,
    entropy_js,
    extract_temporal_features,
    flatten_timepoints,
    load_model,
    mean,
    read_tsv,
    resolve_classes,
    run_class_diagnostics,
    safe_float,
    sample_rows,
    soft_assign,
    source_anchor_ranks,
    weighted_task_summary,
    write_tsv,
    zscore_apply,
    zscore_fit,
)
from analysis.v301_anchor_correspondence_diagnostic import create_train_val_test_folds
from dataset import count_pixelset_samples


TASKS = {
    "FR1_to_FR2": ("france/30TXT/2017", "france/31TCJ/2017"),
    "FR1_to_DK1": ("france/30TXT/2017", "denmark/32VNH/2017"),
    "FR1_to_AT1": ("france/30TXT/2017", "austria/33UVP/2017"),
    "FR2_to_FR1": ("france/31TCJ/2017", "france/30TXT/2017"),
    "FR2_to_DK1": ("france/31TCJ/2017", "denmark/32VNH/2017"),
    "FR2_to_AT1": ("france/31TCJ/2017", "austria/33UVP/2017"),
    "DK1_to_FR1": ("denmark/32VNH/2017", "france/30TXT/2017"),
    "DK1_to_FR2": ("denmark/32VNH/2017", "france/31TCJ/2017"),
    "DK1_to_AT1": ("denmark/32VNH/2017", "austria/33UVP/2017"),
    "AT1_to_FR1": ("austria/33UVP/2017", "france/30TXT/2017"),
    "AT1_to_FR2": ("austria/33UVP/2017", "france/31TCJ/2017"),
    "AT1_to_DK1": ("austria/33UVP/2017", "denmark/32VNH/2017"),
}

CONFIG_ALIASES = {
    "plain": "plain",
    "v276_smooth_k3_w1": "smooth_k3",
    "v275_raw_w1": "raw_global",
    "v303_time_permuted_smooth_k3_w1": "time_permuted_smooth_k3",
}


def parse_args():
    parser = argparse.ArgumentParser("v3.0.3 control and intraclass diagnostic")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--source_run_tag", default="")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--closed_set", default="True")
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
    parser.add_argument("--min_samples_per_class", type=int, default=20)
    parser.add_argument("--max_pairwise_samples", type=int, default=200)
    parser.add_argument("--max_dtw_pairs_per_class", type=int, default=500)
    parser.add_argument("--pairwise_repeats", type=int, default=5)
    parser.add_argument("--shuffle_repeats", type=int, default=50)
    parser.add_argument("--rank_permutation_repeats", type=int, default=50)
    parser.add_argument("--dtw_window_ratio", type=float, default=0.2)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_rows(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def canonical_config(config):
    return CONFIG_ALIASES.get(str(config), str(config))


def task_spec(task):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return TASKS[task]


def set_tag(closed_set):
    return "closedset" if bool_value(closed_set) else "openset"


def source_model_name(source_dataset, task, seed, config, run_tag, closed_set):
    source_tile = source_dataset.split("/")[1]
    return f"pseltae_{source_tile}_{set_tag(closed_set)}_noshift_{run_tag}_{task}_seed{seed}_{config}_source"


def read_control_rows(log_dir):
    rows_path = Path(log_dir) / "raw_strength_rows.tsv"
    rows = read_tsv(rows_path)
    out = []
    for row in rows:
        config = row.get("config", "")
        if config not in CONFIG_ALIASES:
            continue
        task = row.get("task", "")
        if task not in TASKS:
            continue
        source, target = task_spec(task)
        source_on_target = safe_float(row.get("source_on_target_f1"))
        da_f1 = safe_float(row.get("da_f1"))
        out.append({
            "task": task,
            "source": source,
            "target": target,
            "seed": int(float(row.get("seed", 0))),
            "config": canonical_config(config),
            "raw_config": config,
            "source_self_f1": safe_float(row.get("source_self_f1")),
            "source_on_target_f1": source_on_target,
            "da_f1": da_f1,
            "da_gain": None if da_f1 is None or source_on_target is None else da_f1 - source_on_target,
            "estimated_shift": row.get("initial_shift", ""),
            "final_shift_if_available": row.get("last_shift", ""),
            "pseudo_coverage": row.get("last_coverage", ""),
            "pseudo_confidence": row.get("last_mean_conf", ""),
            "status": row.get("status", ""),
            "log": row.get("log", ""),
        })
    return out


def checkpoint_path(args, row):
    source_model = source_model_name(
        row["source"],
        row["task"],
        row["seed"],
        row["raw_config"],
        args.source_run_tag,
        args.closed_set,
    )
    return Path(args.outputs_root) / source_model / "fold_0" / "model.pt"


def permutation_seed(task, seed):
    offsets = {
        "FR1_to_FR2": 101,
        "AT1_to_DK1": 211,
        "FR2_to_AT1": 307,
        "DK1_to_AT1": 401,
        "FR2_to_FR1": 503,
        "AT1_to_FR2": 601,
    }
    return int(seed) + offsets.get(task, 0)


def build_time_permutation_rows(rows, time_steps):
    import torch
    out = []
    seen = set()
    for row in rows:
        if row["raw_config"] != "v303_time_permuted_smooth_k3_w1":
            continue
        key = (row["task"], row["seed"])
        if key in seen:
            continue
        seen.add(key)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(permutation_seed(row["task"], row["seed"]) + 1009 * int(time_steps))
        perm = torch.randperm(int(time_steps), generator=generator).cpu().numpy().astype(int).tolist()
        inv = [0] * int(time_steps)
        for rank, idx in enumerate(perm):
            inv[int(idx)] = rank
        out.append({
            "task": row["task"],
            "seed": row["seed"],
            "T": int(time_steps),
            "permutation_seed": permutation_seed(row["task"], row["seed"]),
            "permutation_json": json.dumps(perm),
            "inverse_permutation_json": json.dumps(inv),
        })
    return out


def transition_matrix_for_sequence(seq, k_value):
    counts = np.zeros((k_value, k_value), dtype=np.float64) + 1e-8
    for a, b in zip(seq[:-1], seq[1:]):
        counts[int(a), int(b)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def entropy(values):
    counts = np.bincount(np.asarray(values, dtype=np.int64).reshape(-1), minlength=1).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def kendall_tau_b(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return None
    concordant = discordant = ties_a = ties_b = 0
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
        return None
    return float((concordant - discordant) / denom)


def intraclass_for_domain(args, classes, base, domain, data_obj, hard_flat, ranks):
    n, t = data_obj["positions"].shape
    hard = hard_flat.reshape(n, t)
    rank_values = ranks[hard]
    rows = []
    rng = np.random.default_rng(args.seed + int(base["seed"]) * 97 + (0 if domain == "source" else 1))
    for class_id in sorted(set(int(x) for x in data_obj["labels"].tolist())):
        mask = data_obj["labels"] == class_id
        anchors = hard[mask]
        ranks_cls = rank_values[mask]
        n_samples = int(mask.sum())
        row = {
            **base,
            "domain": domain,
            "class_id": class_id,
            "class_name_if_available": class_name(classes, class_id),
            "n_samples": n_samples,
        }
        if n_samples < args.min_samples_per_class:
            rows.append({**row, "valid_class": 0, "skip_reason": "too_few_samples"})
            continue
        entropy_value = entropy(anchors)
        sample_count = min(n_samples, int(args.max_pairwise_samples))
        pairwise_corrs = []
        dtw_values = []
        dtw_path_ratios = []
        dtw_large_jumps = []
        for _ in range(int(args.pairwise_repeats)):
            sample_idx = rng.choice(n_samples, size=sample_count, replace=False)
            pair_budget = min(int(args.max_dtw_pairs_per_class), sample_count * (sample_count - 1) // 2)
            pair_seen = 0
            for a_pos in range(sample_count - 1):
                for b_pos in range(a_pos + 1, sample_count):
                    a = ranks_cls[sample_idx[a_pos]]
                    b = ranks_cls[sample_idx[b_pos]]
                    corr = kendall_tau_b(a, b)
                    if corr is not None:
                        pairwise_corrs.append(corr)
                    if pair_seen < pair_budget:
                        dtw, path_len, path_ratio, _, large_jumps = constrained_dtw(a, b, args.dtw_window_ratio)
                        if not math.isnan(dtw):
                            dtw_values.append(dtw)
                            dtw_path_ratios.append(path_ratio)
                            dtw_large_jumps.append(large_jumps)
                        pair_seen += 1
                if pair_seen >= pair_budget:
                    break
        matrices = np.stack([transition_matrix_for_sequence(seq, args.K) for seq in anchors], axis=0)
        mean_matrix = matrices.mean(axis=0)
        js_vals = [entropy_js(m, mean_matrix) for m in matrices]
        l2_vals = [float(np.linalg.norm(m.reshape(-1) - mean_matrix.reshape(-1))) for m in matrices]
        rows.append({
            **row,
            "valid_class": 1,
            "skip_reason": "",
            "intra_class_anchor_entropy": entropy_value,
            "mean_pairwise_order_corr": mean(pairwise_corrs),
            "std_pairwise_order_corr": float(np.nanstd(pairwise_corrs)) if pairwise_corrs else None,
            "mean_transition_js_to_class_mean": mean(js_vals),
            "mean_transition_l2_to_class_mean": mean(l2_vals),
            "mean_constrained_dtw": mean(dtw_values),
            "mean_dtw_path_ratio": mean(dtw_path_ratios),
            "mean_dtw_large_jumps": mean(dtw_large_jumps),
        })
    return rows


def diagnose_checkpoint(args, row):
    ckpt = checkpoint_path(args, row)
    if not ckpt.exists():
        return None, None, [{"task": row["task"], "seed": row["seed"], "config": row["config"], "status": "missing_checkpoint", "error": str(ckpt)}], None
    random.seed(int(row["seed"]))
    np.random.seed(int(row["seed"]))
    import torch
    torch.manual_seed(int(row["seed"]))
    classes = resolve_classes(args.data_root, row["source"], bool_value(args.closed_set))
    indices = {
        row["source"]: count_pixelset_samples(args.data_root, row["source"], classes, closed_set=bool_value(args.closed_set)),
        row["target"]: count_pixelset_samples(args.data_root, row["target"], classes, closed_set=bool_value(args.closed_set)),
    }
    splits = create_train_val_test_folds([row["source"], row["target"]], 1, indices, 0.1, 0.2)[0]
    source_loader = build_loader(args, row["source"], classes, splits[row["source"]]["train"], 0)
    target_loader = build_loader(args, row["target"], classes, splits[row["target"]]["train"], 0)
    diag_args = argparse.Namespace(**vars(args))
    diag_args.source = row["source"]
    diag_args.target = row["target"]
    diag_args.task = row["task"]
    diag_args.seed = int(row["seed"])
    diag_args.checkpoint_config = row["config"]
    diag_args.checkpoint_path = str(ckpt)
    diag_args.output_dir = str(Path(args.output_dir) / "_tmp")
    model, _ = load_model(diag_args, classes)
    source_data = extract_temporal_features(model, source_loader, args.device)
    target_data = extract_temporal_features(model, target_loader, args.device)
    source_flat_raw = flatten_timepoints(source_data["features"], source_data["positions"], source_data["labels"])
    mean_vec, std_vec = zscore_fit(source_flat_raw["features"])
    source_features = zscore_apply(source_data["features"], mean_vec, std_vec)
    target_features = zscore_apply(target_data["features"], mean_vec, std_vec)
    source_flat = flatten_timepoints(source_features, source_data["positions"], source_data["labels"])
    target_flat = flatten_timepoints(target_features, target_data["positions"], target_data["labels"])
    codebook_idx = sample_rows(source_flat["features"].shape[0], args.max_codebook_timepoints, int(row["seed"]))
    kmeans = KMeans(n_clusters=args.K, n_init=10, max_iter=300, random_state=int(row["seed"]))
    kmeans.fit(source_flat["features"][codebook_idx])
    centroids = kmeans.cluster_centers_.astype(np.float64)
    source_d2 = ((source_flat["features"][codebook_idx] - centroids[kmeans.labels_]) ** 2).sum(axis=1)
    tau = max(float(np.median(source_d2)), 1e-6)
    q_source = soft_assign(source_flat["features"], centroids, tau)
    q_target = soft_assign(target_flat["features"], centroids, tau)
    hard_source = q_source.argmax(axis=1)
    hard_target = q_target.argmax(axis=1)
    ranks, _ = source_anchor_ranks(diag_args, source_data["positions"], hard_source)
    _, source_by_class = class_trajectory_rows(diag_args, classes, "source", source_data, hard_source, ranks)
    _, target_by_class = class_trajectory_rows(diag_args, classes, "target", target_data, hard_target, ranks)
    order_rows, perm_rows, trans_rows, dtw_rows, class_summary_rows = run_class_diagnostics(
        diag_args, classes, source_by_class, target_by_class, ranks
    )
    task_order = {
        "task": row["task"],
        "source": row["source"],
        "target": row["target"],
        "seed": row["seed"],
        "config": row["config"],
        "time_steps": int(source_features.shape[1]),
        **weighted_task_summary(class_summary_rows),
    }
    base = {
        "task": row["task"],
        "source": row["source"],
        "target": row["target"],
        "seed": row["seed"],
        "config": row["config"],
    }
    intraclass_rows = []
    intraclass_rows.extend(intraclass_for_domain(args, classes, base, "source", source_data, hard_source, ranks))
    intraclass_rows.extend(intraclass_for_domain(args, classes, base, "target", target_data, hard_target, ranks))
    return task_order, {
        "class_order": order_rows,
        "rank_perm": perm_rows,
        "transition": trans_rows,
        "dtw": dtw_rows,
        "class_summary": class_summary_rows,
    }, [], intraclass_rows


def weighted_summary(rows, group_fields, value_fields, weight_field="n_samples"):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in group_fields)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        valid = [r for r in group if str(r.get("valid_class")) in {"1", "1.0"}]
        item = {field: value for field, value in zip(group_fields, key)}
        item["n_valid_classes"] = len(valid)
        for field in value_fields:
            vals = []
            weights = []
            for row in valid:
                value = safe_float(row.get(field))
                weight = safe_float(row.get(weight_field)) or 1.0
                if value is not None:
                    vals.append(value)
                    weights.append(weight)
            item[field.replace("mean_", "weighted_").replace("intra_class_", "weighted_")] = (
                float(np.average(vals, weights=weights)) if vals else None
            )
        out.append(item)
    return out


def delta(a, b):
    if a is None or b is None:
        return None
    return a - b


def build_control_summaries(control_rows, order_rows):
    by_key = {(r["task"], int(r["seed"]), r["config"]): r for r in order_rows}
    merged = []
    for row in control_rows:
        if row.get("status") != "ok":
            continue
        order = by_key.get((row["task"], int(row["seed"]), row["config"]), {})
        merged.append({**row, **order})
    config_summary = []
    for config in sorted({r["config"] for r in merged}):
        group = [r for r in merged if r["config"] == config]
        config_summary.append({
            "config": config,
            "n_runs": len(group),
            "mean_source_on_target": mean([safe_float(r.get("source_on_target_f1")) for r in group]),
            "mean_da_f1": mean([safe_float(r.get("da_f1")) for r in group]),
            "mean_da_gain": mean([safe_float(r.get("da_gain")) for r in group]),
            "mean_order_gap": mean([safe_float(r.get("weighted_order_gap")) for r in group]),
            "mean_rank_perm_gap": mean([safe_float(r.get("weighted_rank_permutation_gap")) for r in group]),
            "mean_transition_gap": mean([safe_float(r.get("weighted_transition_gap")) for r in group]),
            "mean_dtw_gap": mean([safe_float(r.get("weighted_dtw_gap")) for r in group]),
            "mean_dtw_path_ratio": mean([safe_float(r.get("weighted_dtw_path_length_ratio")) for r in group]),
        })
    task_summary = []
    for task in sorted({r["task"] for r in merged}):
        base = {r["config"]: r for r in aggregate_by_task_config([r for r in merged if r["task"] == task])}
        plain = base.get("plain", {})
        smooth = base.get("smooth_k3", {})
        raw = base.get("raw_global", {})
        perm = base.get("time_permuted_smooth_k3", {})
        def d(row, field):
            return delta(safe_float(row.get(field)), safe_float(plain.get(field)))
        item = {
            "task": task,
            "delta_da_smooth": d(smooth, "da_f1"),
            "delta_order_smooth": d(smooth, "weighted_order_gap"),
            "delta_transition_smooth": d(smooth, "weighted_transition_gap"),
            "delta_da_raw_global": d(raw, "da_f1"),
            "delta_order_raw_global": d(raw, "weighted_order_gap"),
            "delta_transition_raw_global": d(raw, "weighted_transition_gap"),
            "delta_da_time_permuted": d(perm, "da_f1"),
            "delta_order_time_permuted": d(perm, "weighted_order_gap"),
            "delta_transition_time_permuted": d(perm, "weighted_transition_gap"),
        }
        item["smooth_order_da_same_direction"] = same_sign(item["delta_da_smooth"], item["delta_order_smooth"])
        item["raw_order_da_same_direction"] = same_sign(item["delta_da_raw_global"], item["delta_order_raw_global"])
        item["time_permuted_order_da_same_direction"] = same_sign(item["delta_da_time_permuted"], item["delta_order_time_permuted"])
        item["interpretation"] = interpret_control(item)
        task_summary.append(item)
    return merged, config_summary, task_summary


def aggregate_by_task_config(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["config"]].append(row)
    out = []
    fields = [
        "source_on_target_f1", "da_f1", "da_gain", "weighted_order_gap",
        "weighted_rank_permutation_gap", "weighted_transition_gap", "weighted_dtw_gap",
        "weighted_dtw_path_length_ratio",
    ]
    for config, group in groups.items():
        item = {"task": group[0]["task"], "config": config}
        for field in fields:
            item[field] = mean([safe_float(r.get(field)) for r in group])
        out.append(item)
    return out


def same_sign(a, b):
    if a is None or b is None:
        return ""
    if abs(a) <= 1e-12 or abs(b) <= 1e-12:
        return "flat"
    return "1" if (a > 0) == (b > 0) else "0"


def interpret_control(row):
    smooth_good = (row.get("delta_da_smooth") or 0) > 0 and (row.get("delta_order_smooth") or 0) > 0
    raw_good = (row.get("delta_da_raw_global") or 0) > 0 and (row.get("delta_order_raw_global") or 0) > 0
    perm_good = (row.get("delta_da_time_permuted") or 0) > 0 and (row.get("delta_order_time_permuted") or 0) > 0
    if smooth_good and not raw_good and not perm_good:
        return "temporal_order_explanation_strengthened"
    if perm_good:
        return "possible_non_temporal_smooth_confound"
    if raw_good:
        return "possible_feature_quality_confound_raw_global"
    if (row.get("delta_da_smooth") or 0) > 0 and ((row.get("delta_da_raw_global") or 0) > 0 or (row.get("delta_da_time_permuted") or 0) > 0):
        return "temporal_order_explanation_weakened"
    return "mixed"


def build_intraclass_relation(control_rows, intra_summary):
    da_by_task_config = {
        (r["task"], r["config"]): r for r in aggregate_by_task_config([r for r in control_rows if r.get("status") == "ok"])
    }
    by = {(r["task"], r["config"], r["domain"]): r for r in intra_summary}
    rows = []
    for task in sorted({r["task"] for r in intra_summary}):
        plain_s = by.get((task, "plain", "source"), {})
        smooth_s = by.get((task, "smooth_k3", "source"), {})
        plain_t = by.get((task, "plain", "target"), {})
        smooth_t = by.get((task, "smooth_k3", "target"), {})
        plain_da = da_by_task_config.get((task, "plain"), {})
        smooth_da = da_by_task_config.get((task, "smooth_k3"), {})
        row = {
            "task": task,
            "delta_da_smooth": delta(safe_float(smooth_da.get("da_f1")), safe_float(plain_da.get("da_f1"))),
            "delta_source_anchor_entropy": delta(safe_float(smooth_s.get("weighted_anchor_entropy")), safe_float(plain_s.get("weighted_anchor_entropy"))),
            "delta_source_pairwise_order_corr": delta(safe_float(smooth_s.get("weighted_pairwise_order_corr")), safe_float(plain_s.get("weighted_pairwise_order_corr"))),
            "delta_source_transition_js": delta(safe_float(smooth_s.get("weighted_transition_js_to_class_mean")), safe_float(plain_s.get("weighted_transition_js_to_class_mean"))),
            "delta_source_dtw": delta(safe_float(smooth_s.get("weighted_constrained_dtw")), safe_float(plain_s.get("weighted_constrained_dtw"))),
            "delta_target_anchor_entropy": delta(safe_float(smooth_t.get("weighted_anchor_entropy")), safe_float(plain_t.get("weighted_anchor_entropy"))),
            "delta_target_pairwise_order_corr": delta(safe_float(smooth_t.get("weighted_pairwise_order_corr")), safe_float(plain_t.get("weighted_pairwise_order_corr"))),
            "delta_target_transition_js": delta(safe_float(smooth_t.get("weighted_transition_js_to_class_mean")), safe_float(plain_t.get("weighted_transition_js_to_class_mean"))),
            "delta_target_dtw": delta(safe_float(smooth_t.get("weighted_constrained_dtw")), safe_float(plain_t.get("weighted_constrained_dtw"))),
        }
        row["source_intraclass_improved"] = intraclass_improved(row, "source")
        row["target_intraclass_improved"] = intraclass_improved(row, "target")
        row["intraclass_da_same_direction"] = same_sign(row["delta_da_smooth"], intraclass_score(row))
        row["interpretation"] = interpret_intraclass(row)
        rows.append(row)
    return rows


def intraclass_score(row):
    vals = [
        -(row.get("delta_source_anchor_entropy") or 0),
        row.get("delta_source_pairwise_order_corr") or 0,
        -(row.get("delta_source_transition_js") or 0),
        -(row.get("delta_source_dtw") or 0),
        -(row.get("delta_target_anchor_entropy") or 0),
        row.get("delta_target_pairwise_order_corr") or 0,
        -(row.get("delta_target_transition_js") or 0),
        -(row.get("delta_target_dtw") or 0),
    ]
    return sum(vals)


def intraclass_improved(row, domain):
    checks = [
        (row.get(f"delta_{domain}_anchor_entropy") or 0) < 0,
        (row.get(f"delta_{domain}_pairwise_order_corr") or 0) > 0,
        (row.get(f"delta_{domain}_transition_js") or 0) < 0,
        (row.get(f"delta_{domain}_dtw") or 0) < 0,
    ]
    return int(sum(checks) >= 3)


def interpret_intraclass(row):
    da_up = (row.get("delta_da_smooth") or 0) > 0
    src = row.get("source_intraclass_improved") == 1
    tgt = row.get("target_intraclass_improved") == 1
    if da_up and src and tgt:
        return "intraclass_consistency_supports_da_gain"
    if (not da_up) and (not src or not tgt):
        return "intraclass_consistency_aligned_with_da_drop"
    if src and not tgt:
        return "source_structure_improves_but_target_unstable"
    if src or tgt:
        return "intraclass_consistency_not_sufficient"
    return "target_intraclass_unstable_alignment_risk"


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    return pearson(rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs]))


def sign_test_p(success, n):
    if n <= 0:
        return None
    from math import comb
    tail = sum(comb(n, k) for k in range(success, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def build_stats(control_vs_rows, repeats, seed):
    xs = [safe_float(r.get("delta_da_smooth")) for r in control_vs_rows]
    ys = [safe_float(r.get("delta_order_smooth")) for r in control_vs_rows]
    pairs = [(x, y, r["task"]) for x, y, r in zip(xs, ys, control_vs_rows) if x is not None and y is not None]
    same = sum((x > 0) == (y > 0) for x, y, _ in pairs if abs(x) > 1e-12 and abs(y) > 1e-12)
    n = len(pairs)
    stat_rows = [
        {"metric": "sign_same_direction_count", "value": same, "p_value_if_available": sign_test_p(same, n), "ci_low": "", "ci_high": "", "notes": f"n={n}"},
        {"metric": "pearson_delta_da_order", "value": pearson([p[0] for p in pairs], [p[1] for p in pairs]), "p_value_if_available": "", "ci_low": "", "ci_high": "", "notes": ""},
        {"metric": "spearman_delta_da_order", "value": spearman([p[0] for p in pairs], [p[1] for p in pairs]), "p_value_if_available": "", "ci_low": "", "ci_high": "", "notes": ""},
    ]
    loo = []
    for left_out in [p[2] for p in pairs]:
        subset = [p for p in pairs if p[2] != left_out]
        same_sub = sum((x > 0) == (y > 0) for x, y, _ in subset if abs(x) > 1e-12 and abs(y) > 1e-12)
        loo.append({
            "left_out_task": left_out,
            "n_remaining": len(subset),
            "same_direction_count": same_sub,
            "same_direction_rate": same_sub / max(len(subset), 1),
            "pearson_corr_delta_da_order": pearson([p[0] for p in subset], [p[1] for p in subset]),
            "spearman_corr_delta_da_order": spearman([p[0] for p in subset], [p[1] for p in subset]),
        })
    rng = np.random.default_rng(seed)
    rates = []
    pearsons = []
    spearmans = []
    if pairs:
        for _ in range(int(repeats)):
            idx = rng.integers(0, len(pairs), size=len(pairs))
            sample = [pairs[i] for i in idx]
            rates.append(sum((x > 0) == (y > 0) for x, y, _ in sample) / len(sample))
            pearsons.append(pearson([p[0] for p in sample], [p[1] for p in sample]))
            spearmans.append(spearman([p[0] for p in sample], [p[1] for p in sample]))
    def ci(vals):
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        if not vals:
            return None, None, None
        return mean(vals), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    rate_m, rate_l, rate_h = ci(rates)
    pear_m, pear_l, pear_h = ci(pearsons)
    spear_m, spear_l, spear_h = ci(spearmans)
    boot = [{
        "bootstrap_repeats": repeats,
        "mean_same_direction_rate": rate_m,
        "ci95_same_direction_low": rate_l,
        "ci95_same_direction_high": rate_h,
        "mean_pearson": pear_m,
        "ci95_pearson_low": pear_l,
        "ci95_pearson_high": pear_h,
        "mean_spearman": spear_m,
        "ci95_spearman_low": spear_l,
        "ci95_spearman_high": spear_h,
    }]
    return stat_rows, loo, boot


def write_summary(path, control_config, control_vs, intra_relation, stats_rows, failed_rows):
    lines = ["# v3.0.3 控制与域内一致性诊断总结\n\n"]
    lines.append("## 1. 实验目的\n\n")
    lines.append("本轮用于检查 smooth/order_gap/DA 同向是否可能来自一般特征紧致性共因；不是新方法实验。\n\n")
    lines.append("## 2. 非时序对照配置级结果\n\n")
    lines.append("| config | n | DA F1 | DA gain | order gap | transition gap | DTW gap |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in control_config:
        lines.append(
            f"| {row['config']} | {row['n_runs']} | {fmt(row.get('mean_da_f1'))} | {fmt(row.get('mean_da_gain'))} | "
            f"{fmt(row.get('mean_order_gap'))} | {fmt(row.get('mean_transition_gap'))} | {fmt(row.get('mean_dtw_gap'))} |\n"
        )
    lines.append("\n## 3. smooth / raw / time-permuted 相对 plain\n\n")
    lines.append("| task | ΔDA smooth | Δorder smooth | ΔDA raw | Δorder raw | ΔDA perm | Δorder perm | interpretation |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for row in control_vs:
        lines.append(
            f"| {row['task'].replace('_to_', '->')} | {fmt(row.get('delta_da_smooth'))} | {fmt(row.get('delta_order_smooth'))} | "
            f"{fmt(row.get('delta_da_raw_global'))} | {fmt(row.get('delta_order_raw_global'))} | "
            f"{fmt(row.get('delta_da_time_permuted'))} | {fmt(row.get('delta_order_time_permuted'))} | {row.get('interpretation')} |\n"
        )
    lines.append("\n## 4. 域内同类一致性关系\n\n")
    lines.append("| task | ΔDA smooth | source improved | target improved | interpretation |\n")
    lines.append("|---|---:|---:|---:|---|\n")
    for row in intra_relation:
        lines.append(
            f"| {row['task'].replace('_to_', '->')} | {fmt(row.get('delta_da_smooth'))} | "
            f"{row.get('source_intraclass_improved')} | {row.get('target_intraclass_improved')} | {row.get('interpretation')} |\n"
        )
    lines.append("\n## 5. 统计检验\n\n")
    lines.append("| metric | value | p-value | notes |\n")
    lines.append("|---|---:|---:|---|\n")
    for row in stats_rows:
        lines.append(f"| {row['metric']} | {fmt(row.get('value'))} | {fmt(row.get('p_value_if_available'))} | {row.get('notes','')} |\n")
    lines.append("\n## 6. 风险点\n\n")
    lines.append("- target label 只用于离线诊断，不能部署。\n")
    lines.append("- DTW 只作辅助，不能作为主证据。\n")
    lines.append("- 若 raw/time-permuted 对照也同步提升 DA 和 order_gap，时间顺序解释需要降级。\n")
    lines.append(f"- failed runs: {len(failed_rows)}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


CONTROL_DA_FIELDS = [
    "task", "source", "target", "seed", "config", "source_self_f1",
    "source_on_target_f1", "da_f1", "da_gain", "estimated_shift",
    "final_shift_if_available", "pseudo_coverage", "pseudo_confidence", "status", "log",
]
CONTROL_ORDER_FIELDS = [
    "task", "source", "target", "seed", "config", "weighted_order_gap",
    "weighted_rank_permutation_gap", "weighted_transition_gap", "weighted_dtw_gap",
    "weighted_dtw_path_length_ratio", "positive_order_gap_classes",
    "positive_transition_gap_classes", "n_valid_classes",
]
CONTROL_CONFIG_FIELDS = [
    "config", "n_runs", "mean_source_on_target", "mean_da_f1", "mean_da_gain",
    "mean_order_gap", "mean_rank_perm_gap", "mean_transition_gap",
    "mean_dtw_gap", "mean_dtw_path_ratio",
]
CONTROL_VS_FIELDS = [
    "task", "delta_da_smooth", "delta_order_smooth", "delta_transition_smooth",
    "delta_da_raw_global", "delta_order_raw_global", "delta_transition_raw_global",
    "delta_da_time_permuted", "delta_order_time_permuted", "delta_transition_time_permuted",
    "smooth_order_da_same_direction", "raw_order_da_same_direction",
    "time_permuted_order_da_same_direction", "interpretation",
]
INTRA_FIELDS = [
    "task", "source", "target", "seed", "config", "domain", "class_id",
    "class_name_if_available", "n_samples", "valid_class", "skip_reason",
    "intra_class_anchor_entropy", "mean_pairwise_order_corr", "std_pairwise_order_corr",
    "mean_transition_js_to_class_mean", "mean_transition_l2_to_class_mean",
    "mean_constrained_dtw", "mean_dtw_path_ratio", "mean_dtw_large_jumps",
]
INTRA_SUMMARY_FIELDS = [
    "task", "source", "target", "seed", "config", "domain", "n_valid_classes",
    "weighted_anchor_entropy", "weighted_pairwise_order_corr",
    "weighted_transition_js_to_class_mean", "weighted_transition_l2_to_class_mean",
    "weighted_constrained_dtw", "weighted_dtw_path_ratio", "weighted_dtw_large_jumps",
]
INTRA_REL_FIELDS = [
    "task", "delta_da_smooth", "delta_source_anchor_entropy",
    "delta_source_pairwise_order_corr", "delta_source_transition_js", "delta_source_dtw",
    "delta_target_anchor_entropy", "delta_target_pairwise_order_corr",
    "delta_target_transition_js", "delta_target_dtw", "source_intraclass_improved",
    "target_intraclass_improved", "intraclass_da_same_direction", "interpretation",
]
STAT_FIELDS = ["metric", "value", "p_value_if_available", "ci_low", "ci_high", "notes"]
LOO_FIELDS = ["left_out_task", "n_remaining", "same_direction_count", "same_direction_rate", "pearson_corr_delta_da_order", "spearman_corr_delta_da_order"]
BOOT_FIELDS = ["bootstrap_repeats", "mean_same_direction_rate", "ci95_same_direction_low", "ci95_same_direction_high", "mean_pearson", "ci95_pearson_low", "ci95_pearson_high", "mean_spearman", "ci95_spearman_low", "ci95_spearman_high"]
FAILED_FIELDS = ["task", "seed", "config", "status", "error"]
PERM_FIELDS = ["task", "seed", "T", "permutation_seed", "permutation_json", "inverse_permutation_json"]


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir or args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.source_run_tag:
        args.source_run_tag = log_dir.name
    control_rows = read_control_rows(log_dir)
    write_rows(output_dir / "control_da_results.tsv", control_rows, CONTROL_DA_FIELDS)
    order_rows = []
    failed = []
    intraclass_rows = []
    time_steps = None
    for row in control_rows:
        if row.get("status") != "ok":
            failed.append({"task": row["task"], "seed": row["seed"], "config": row["config"], "status": row.get("status"), "error": "non_ok_training"})
            continue
        print(f"DIAG|task={row['task']}|seed={row['seed']}|config={row['config']}", flush=True)
        try:
            task_order, _, fail_rows, intra = diagnose_checkpoint(args, row)
        except Exception as exc:
            failed.append({
                "task": row["task"],
                "seed": row["seed"],
                "config": row["config"],
                "status": "diagnostic_exception",
                "error": repr(exc),
            })
            continue
        if fail_rows:
            failed.extend(fail_rows)
            continue
        order_rows.append(task_order)
        intraclass_rows.extend(intra)
        if time_steps is None and task_order.get("time_steps"):
            time_steps = int(task_order["time_steps"])
    write_rows(output_dir / "control_order_results.tsv", order_rows, CONTROL_ORDER_FIELDS)
    write_rows(output_dir / "intraclass_consistency_results.tsv", intraclass_rows, INTRA_FIELDS)
    intra_summary = weighted_summary(
        intraclass_rows,
        ["task", "source", "target", "seed", "config", "domain"],
        [
            "intra_class_anchor_entropy",
            "mean_pairwise_order_corr",
            "mean_transition_js_to_class_mean",
            "mean_transition_l2_to_class_mean",
            "mean_constrained_dtw",
            "mean_dtw_path_ratio",
            "mean_dtw_large_jumps",
        ],
    )
    write_rows(output_dir / "intraclass_consistency_summary.tsv", intra_summary, INTRA_SUMMARY_FIELDS)
    _, control_config, control_vs = build_control_summaries(control_rows, order_rows)
    write_rows(output_dir / "control_config_summary.tsv", control_config, CONTROL_CONFIG_FIELDS)
    write_rows(output_dir / "control_vs_smooth_summary.tsv", control_vs, CONTROL_VS_FIELDS)
    intra_relation = build_intraclass_relation(control_rows, intra_summary)
    write_rows(output_dir / "smooth_plain_intraclass_relation.tsv", intra_relation, INTRA_REL_FIELDS)
    stats_rows, loo_rows, boot_rows = build_stats(control_vs, args.bootstrap_repeats, args.seed)
    write_rows(output_dir / "order_da_statistical_test.tsv", stats_rows, STAT_FIELDS)
    write_rows(output_dir / "leave_one_task_out_stats.tsv", loo_rows, LOO_FIELDS)
    write_rows(output_dir / "bootstrap_order_da_stats.tsv", boot_rows, BOOT_FIELDS)
    write_rows(output_dir / "failed_runs.tsv", failed, FAILED_FIELDS)
    if time_steps is None:
        time_steps = 128
    write_rows(output_dir / "time_permutation_by_task_seed.tsv", build_time_permutation_rows(control_rows, time_steps), PERM_FIELDS)
    write_summary(
        output_dir / "v303_control_and_intraclass_diagnostic_summary.md",
        control_config,
        control_vs,
        intra_relation,
        stats_rows,
        failed,
    )
    print(f"Wrote v3.0.3 diagnostic outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
