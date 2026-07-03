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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torch.utils import data
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import PixelSetData, count_pixelset_samples
from models.stclassifier import PseLTae
from transforms import Normalize, RandomSamplePixels, ToTensor
from utils import label_utils


TAG_TO_DATASET = {
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
    "DK1": "denmark/32VNH/2017",
    "AT1": "austria/33UVP/2017",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="v3.0.1 offline diagnostic for source-defined temporal anchor correspondence."
    )
    parser.add_argument("--mode", choices=["run", "aggregate"], default="run")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--source", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint_config", default="smooth_k3")
    parser.add_argument("--source_model", default="")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_root", default="")
    parser.add_argument("--v300_log_dir", default="")
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
    parser.add_argument("--max_silhouette_timepoints", type=int, default=5000)
    parser.add_argument("--max_stability_points", type=int, default=12000)
    parser.add_argument("--max_temporal_shift", type=int, default=60)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--matrix_fit_method", choices=["least_squares", "projected_gd"], default="least_squares")
    parser.add_argument("--shuffle_control_repeats", type=int, default=1)
    return parser.parse_args()


def bool_value(text):
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


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


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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
    return float(sum(vals) / len(vals))


def parse_task(task):
    source_tag, target_tag = task.split("_to_")
    return source_tag, target_tag


def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset_name in datasets:
            if isinstance(num_indices, dict):
                indices = list(range(num_indices[dataset_name]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val
            random.shuffle(indices)
            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) == n - n_test
            splits[dataset_name] = {"train": train_indices, "val": val_indices, "test": test_indices}
        folds.append(splits)
    return folds


def resolve_classes(data_root, source, closed_set):
    source_classes = label_utils.get_classes(
        source.split("/")[0],
        combine_spring_and_winter=False,
    )
    if closed_set:
        source_classes = [cls for cls in source_classes if cls != "unknown"]
    source_data = PixelSetData(data_root, source, source_classes, closed_set=closed_set)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    return [source_classes[i] for i in labels[counts >= 200]]


def maybe_limit_indices(indices, max_samples, seed):
    indices = list(indices)
    if max_samples and len(indices) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = sorted(indices[:max_samples])
    return set(indices)


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


def load_model(args, classes):
    model = PseLTae(
        input_dim=args.input_dim,
        num_classes=len(classes),
        with_extra=bool_value(args.with_extra),
    )
    checkpoint = None
    if args.checkpoint_path:
        checkpoint = Path(args.checkpoint_path)
    elif args.source_model:
        checkpoint = Path("outputs") / args.source_model / "fold_0" / "model.pt"
        if not checkpoint.exists():
            checkpoint = Path(args.source_model)
    if checkpoint is None:
        raise FileNotFoundError("Either --checkpoint_path or --source_model is required.")
    if not checkpoint.is_absolute():
        checkpoint = ROOT_DIR / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)["state_dict"]
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def extract_temporal_features(model, loader, device):
    features = []
    labels = []
    positions = []
    sample_indices = []
    for sample in loader:
        pixels = sample["pixels"].to(device=device, non_blocking=True)
        mask = sample["valid_pixels"].to(device=device, non_blocking=True)
        extra = sample["extra"].to(device=device, non_blocking=True)
        feats = model.spatial_encoder(pixels, mask, extra).detach().cpu().float().numpy()
        features.append(feats)
        labels.append(sample["label"].detach().cpu().long().numpy())
        positions.append(sample["positions"].detach().cpu().long().numpy())
        sample_indices.append(sample["index"].detach().cpu().long().numpy())
    return {
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "positions": np.concatenate(positions, axis=0),
        "indices": np.concatenate(sample_indices, axis=0),
    }


def flatten_timepoints(features, positions=None, labels=None):
    n, t, d = features.shape
    flat = features.reshape(n * t, d)
    out = {"features": flat}
    if positions is not None:
        out["positions"] = positions.reshape(n * t)
    if labels is not None:
        out["labels"] = np.repeat(labels, t)
    return out


def sample_rows(count, max_count, seed):
    if max_count <= 0 or count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def zscore_fit(flat_features):
    mean_vec = flat_features.mean(axis=0)
    std_vec = flat_features.std(axis=0)
    std_vec = np.maximum(std_vec, 1e-6)
    return mean_vec, std_vec


def zscore_apply(features, mean_vec, std_vec):
    return (features - mean_vec.reshape(1, 1, -1)) / std_vec.reshape(1, 1, -1)


def soft_assign(flat_features, centroids, tau, chunk_size=65536):
    out = []
    centroids = centroids.astype(np.float64)
    for start in range(0, flat_features.shape[0], chunk_size):
        x = flat_features[start:start + chunk_size].astype(np.float64)
        d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        logits = -d2 / max(float(tau), 1e-8)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        out.append(probs.astype(np.float32))
    return np.concatenate(out, axis=0)


def entropy_rows(probs, eps=1e-12):
    return -(probs * np.log(probs + eps)).sum(axis=1)


def distribution_effective_count(dist, eps=1e-12):
    ent = float(-(dist * np.log(dist + eps)).sum())
    return float(math.exp(ent))


def normalized_entropy(dist, eps=1e-12):
    dist = np.asarray(dist, dtype=np.float64)
    if dist.size <= 1 or dist.sum() <= 0:
        return 0.0
    dist = dist / dist.sum()
    return float((-(dist * np.log(dist + eps)).sum()) / math.log(dist.size))


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)
    kl_pm = float((p * (np.log(p + eps) - np.log(m + eps))).sum())
    kl_qm = float((q * (np.log(q + eps) - np.log(m + eps))).sum())
    return 0.5 * (kl_pm + kl_qm)


def l2_distance(p, q):
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))


def aggregate_by_time(q, positions, labels=None, class_id=None):
    q = q.reshape(positions.shape[0], positions.shape[1], q.shape[-1])
    positions = positions.astype(np.int64)
    if labels is not None and class_id is not None:
        mask_samples = labels == class_id
        q = q[mask_samples]
        positions = positions[mask_samples]
    sums = defaultdict(lambda: np.zeros(q.shape[-1], dtype=np.float64))
    counts = defaultdict(int)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            t = int(positions[i, j])
            sums[t] += q[i, j]
            counts[t] += 1
    return {t: sums[t] / max(counts[t], 1) for t in sorted(sums)}, dict(counts)


def rows_for_time_distribution(task, source, target, seed, config, k_value, domain, dist_by_time, counts):
    rows = []
    for time_bin, dist in sorted(dist_by_time.items()):
        rows.append({
            "task": task,
            "source": source,
            "target": target,
            "seed": seed,
            "checkpoint_config": config,
            "K": k_value,
            "domain": domain,
            "time_bin": time_bin,
            "anchor_distribution_json": json.dumps([float(x) for x in dist]),
            "n_timepoints": counts.get(time_bin, 0),
        })
    return rows


def nearest_time_key(by_time, query_time):
    if not by_time:
        return None
    keys = sorted(by_time)
    query_time = int(query_time)
    pos = np.searchsorted(keys, query_time)
    candidates = []
    if pos < len(keys):
        candidates.append(keys[pos])
    if pos > 0:
        candidates.append(keys[pos - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(int(x) - query_time))


def eval_shift(qs_by_time, qt_by_time, target_times, delta):
    js_values, l2_values = [], []
    for t in target_times:
        src_t = nearest_time_key(qs_by_time, int(t) + int(delta))
        if src_t is None or t not in qt_by_time:
            continue
        js_values.append(js_divergence(qs_by_time[src_t], qt_by_time[t]))
        l2_values.append(l2_distance(qs_by_time[src_t], qt_by_time[t]))
    if not js_values:
        return float("nan"), float("nan"), 0
    return float(np.mean(js_values)), float(np.mean(l2_values)), len(js_values)


def fit_correspondence_matrix(qs_rows, qt_rows, method="least_squares"):
    k_value = qs_rows.shape[1]
    if qs_rows.shape[0] == 0:
        return np.eye(k_value, dtype=np.float64)
    if method == "projected_gd":
        matrix = np.eye(k_value, dtype=np.float64)
        lr = 0.5
        for _ in range(300):
            pred = qs_rows @ matrix
            grad = qs_rows.T @ (pred - qt_rows) / max(qs_rows.shape[0], 1)
            matrix -= lr * grad
            matrix = np.clip(matrix, 1e-8, None)
            matrix /= matrix.sum(axis=1, keepdims=True)
        return matrix
    matrix, *_ = np.linalg.lstsq(qs_rows, qt_rows, rcond=None)
    matrix = np.clip(matrix, 1e-8, None)
    row_sums = matrix.sum(axis=1, keepdims=True)
    bad = row_sums.squeeze(1) <= 1e-8
    if bad.any():
        matrix[bad] = 1.0 / k_value
        row_sums = matrix.sum(axis=1, keepdims=True)
    matrix /= row_sums
    return matrix


def eval_matrix(qs_by_time, qt_by_time, target_times, matrix, delta=0):
    js_values, l2_values = [], []
    for t in target_times:
        src_t = nearest_time_key(qs_by_time, int(t) + int(delta))
        if src_t is None or t not in qt_by_time:
            continue
        pred = qs_by_time[src_t] @ matrix
        js_values.append(js_divergence(pred, qt_by_time[t]))
        l2_values.append(l2_distance(pred, qt_by_time[t]))
    if not js_values:
        return float("nan"), float("nan"), 0
    return float(np.mean(js_values)), float(np.mean(l2_values)), len(js_values)


def temporal_block_folds(times, n_folds):
    times = sorted(set(int(t) for t in times))
    if len(times) < 2:
        return [(times, times)]
    n_folds = min(max(2, n_folds), len(times))
    fold_sizes = [len(times) // n_folds] * n_folds
    for idx in range(len(times) % n_folds):
        fold_sizes[idx] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        test = times[start:start + size]
        train = times[:start] + times[start + size:]
        if not train:
            train = test
        folds.append((train, test))
        start += size
    return folds


def paired_rows_for_times(qs_by_time, qt_by_time, times, delta=0):
    qs, qt = [], []
    for t in times:
        src_t = nearest_time_key(qs_by_time, int(t) + int(delta))
        if src_t is not None and t in qt_by_time:
            qs.append(qs_by_time[src_t])
            qt.append(qt_by_time[t])
    if not qs:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.vstack(qs), np.vstack(qt)


def best_shift_cv(qs_by_time, qt_by_time, train_times, max_shift):
    best_delta, best_js, best_l2 = 0, float("inf"), float("inf")
    for delta in range(-max_shift, max_shift + 1):
        js_value, l2_value, count = eval_shift(qs_by_time, qt_by_time, train_times, delta)
        if count and js_value < best_js:
            best_delta, best_js, best_l2 = delta, js_value, l2_value
    return best_delta, best_js, best_l2


def load_timematch_estimated_shift(v300_log_dir, task, seed):
    if not v300_log_dir:
        return None
    path = Path(v300_log_dir) / "offline" / f"{task}_seed{seed}" / "oracle_scalar_shift_summary.tsv"
    rows = read_tsv(path)
    if not rows:
        return None
    value = rows[0].get("estimated_shift_am")
    if value in {None, ""}:
        return None
    return int(float(value))


def matrix_summary(matrix, mean_source_times):
    k_value = matrix.shape[0]
    row_ent = [normalized_entropy(row) for row in matrix]
    col_sums = matrix.sum(axis=0)
    col_dist = col_sums / max(float(col_sums.sum()), 1e-12)
    col_entropy = normalized_entropy(col_dist)
    diag_mass = float(np.trace(matrix) / max(float(matrix.sum()), 1e-12))
    max_entry_mean = float(matrix.max(axis=1).mean())
    effective_matches = distribution_effective_count(matrix.reshape(-1) / max(float(matrix.sum()), 1e-12))
    order = np.argsort(mean_source_times)
    ordered = matrix[np.ix_(order, order)]
    residuals = []
    for shift in range(k_value):
        perm = np.zeros_like(ordered)
        for i in range(k_value):
            perm[i, (i + shift) % k_value] = 1.0
        residuals.append(float(np.linalg.norm(ordered - perm)))
    return {
        "row_entropy_mean": float(np.mean(row_ent)),
        "col_entropy_mean": float(col_entropy),
        "diag_mass": diag_mass,
        "max_entry_mean": max_entry_mean,
        "effective_matches": effective_matches,
        "best_shift_permutation_residual": min(residuals) if residuals else float("nan"),
    }


def anchorability_metrics(args, source_flat, source_labels_flat, source_positions_flat, q_source, hard_source):
    d2 = ((source_flat - args["_centroids"][hard_source]) ** 2).sum(axis=1)
    cluster_compactness = float(d2.mean())
    entropy = entropy_rows(q_source)
    max_probs = q_source.max(axis=1)
    dist = q_source.mean(axis=0)
    effective_anchor_count = distribution_effective_count(dist)
    if hard_source.shape[0] > 1:
        # Temporal persistence is computed before flattening by sorting the original positions per sample.
        temporal_persistence = args.get("_temporal_persistence", float("nan"))
    else:
        temporal_persistence = float("nan")

    class_entropies, class_weights = [], []
    for class_id in sorted(set(int(x) for x in source_labels_flat.tolist())):
        mask = source_labels_flat == class_id
        if not mask.any():
            continue
        class_dist = np.bincount(hard_source[mask], minlength=args["K"]).astype(np.float64)
        class_entropies.append(normalized_entropy(class_dist))
        class_weights.append(int(mask.sum()))
    if class_entropies:
        class_cond_entropy = float(np.average(class_entropies, weights=class_weights))
    else:
        class_cond_entropy = float("nan")

    purity_num = 0.0
    for anchor_id in range(args["K"]):
        mask = hard_source == anchor_id
        if not mask.any():
            continue
        counts = np.bincount(source_labels_flat[mask].astype(np.int64))
        purity_num += float(counts.max())
    class_cond_purity = purity_num / max(float(hard_source.shape[0]), 1.0)

    silhouette = float("nan")
    if args["max_silhouette_timepoints"] and len(np.unique(hard_source)) > 1:
        sample_idx = sample_rows(source_flat.shape[0], args["max_silhouette_timepoints"], args["seed"])
        sampled_labels = hard_source[sample_idx]
        if len(np.unique(sampled_labels)) > 1:
            silhouette = float(silhouette_score(source_flat[sample_idx], sampled_labels))

    return {
        "cluster_compactness": cluster_compactness,
        "silhouette_score_sampled": silhouette,
        "assignment_entropy_mean": float(entropy.mean()),
        "max_assignment_prob_mean": float(max_probs.mean()),
        "effective_anchor_count": effective_anchor_count,
        "temporal_persistence": temporal_persistence,
        "class_conditional_anchor_entropy": class_cond_entropy,
        "class_conditional_anchor_purity": class_cond_purity,
    }


def compute_temporal_persistence(hard_assign, features_shape):
    n, t, _ = features_shape
    hard = hard_assign.reshape(n, t)
    if t <= 1:
        return float("nan")
    return float((hard[:, 1:] == hard[:, :-1]).mean())


def class_oracle_shift_cv(source_q, target_q, source_positions, target_positions, source_labels, target_labels, folds, max_shift):
    classes = sorted(set(int(x) for x in target_labels.tolist()) & set(int(x) for x in source_labels.tolist()))
    rows = []
    fold_values = []
    for fold_id, (train_times, test_times) in enumerate(folds):
        weighted_js, weighted_l2, total_weight = 0.0, 0.0, 0.0
        selected = {}
        for class_id in classes:
            src_by_time, _ = aggregate_by_time(source_q, source_positions, source_labels, class_id)
            tgt_by_time, _ = aggregate_by_time(target_q, target_positions, target_labels, class_id)
            if not src_by_time or not tgt_by_time:
                continue
            delta, train_js, train_l2 = best_shift_cv(src_by_time, tgt_by_time, train_times, max_shift)
            test_js, test_l2, count = eval_shift(src_by_time, tgt_by_time, test_times, delta)
            if not count or math.isnan(test_js):
                continue
            weight = float((target_labels == class_id).sum())
            weighted_js += weight * test_js
            weighted_l2 += weight * test_l2
            total_weight += weight
            selected[class_id] = delta
        value_js = weighted_js / total_weight if total_weight else float("nan")
        value_l2 = weighted_l2 / total_weight if total_weight else float("nan")
        fold_values.append((value_js, value_l2))
        rows.append({
            "fold_id": fold_id,
            "selected_delta": json.dumps({str(k): int(v) for k, v in selected.items()}),
            "test_js": value_js,
            "test_l2": value_l2,
        })
    return rows, fold_values


def run_alignment_suite(args, qs_by_time, qt_by_time, source_q, target_q, source_positions, target_positions, source_labels, target_labels, mean_source_times):
    target_times = sorted(qt_by_time)
    folds = temporal_block_folds(target_times, args.cv_folds if len(target_times) >= args.cv_folds else 3)
    rows = []
    matrix_rows = []
    class_oracle_rows, class_oracle_values = class_oracle_shift_cv(
        source_q,
        target_q,
        source_positions,
        target_positions,
        source_labels,
        target_labels,
        folds,
        args.max_temporal_shift,
    )
    estimated_shift = load_timematch_estimated_shift(args.v300_log_dir, args.task, args.seed)

    for fold_id, (train_times, test_times) in enumerate(folds):
        no_train_js, no_train_l2, _ = eval_shift(qs_by_time, qt_by_time, train_times, 0)
        no_test_js, no_test_l2, _ = eval_shift(qs_by_time, qt_by_time, test_times, 0)
        rows.append({
            "fold_id": fold_id,
            "alignment_type": "no_alignment",
            "train_js": no_train_js,
            "test_js": no_test_js,
            "train_l2": no_train_l2,
            "test_l2": no_test_l2,
            "selected_delta": 0,
            "uses_target_label": 0,
        })

        if estimated_shift is None:
            tm_delta, _, _ = best_shift_cv(qs_by_time, qt_by_time, train_times, args.max_temporal_shift)
        else:
            tm_delta = estimated_shift
        tm_train_js, tm_train_l2, _ = eval_shift(qs_by_time, qt_by_time, train_times, tm_delta)
        tm_test_js, tm_test_l2, _ = eval_shift(qs_by_time, qt_by_time, test_times, tm_delta)
        rows.append({
            "fold_id": fold_id,
            "alignment_type": "timematch_estimated_global_shift",
            "train_js": tm_train_js,
            "test_js": tm_test_js,
            "train_l2": tm_train_l2,
            "test_l2": tm_test_l2,
            "selected_delta": tm_delta,
            "uses_target_label": 0,
        })

        best_delta, best_train_js, best_train_l2 = best_shift_cv(
            qs_by_time, qt_by_time, train_times, args.max_temporal_shift
        )
        best_test_js, best_test_l2, _ = eval_shift(qs_by_time, qt_by_time, test_times, best_delta)
        rows.append({
            "fold_id": fold_id,
            "alignment_type": "best_global_scalar_shift_cv",
            "train_js": best_train_js,
            "test_js": best_test_js,
            "train_l2": best_train_l2,
            "test_l2": best_test_l2,
            "selected_delta": best_delta,
            "uses_target_label": 0,
        })

        if fold_id < len(class_oracle_rows):
            rows.append({
                "fold_id": fold_id,
                "alignment_type": "per_class_scalar_shift_oracle",
                "train_js": "",
                "test_js": class_oracle_rows[fold_id]["test_js"],
                "train_l2": "",
                "test_l2": class_oracle_rows[fold_id]["test_l2"],
                "selected_delta": class_oracle_rows[fold_id]["selected_delta"],
                "uses_target_label": 1,
            })

        qs_train, qt_train = paired_rows_for_times(qs_by_time, qt_by_time, train_times, 0)
        matrix = fit_correspondence_matrix(qs_train, qt_train, args.matrix_fit_method)
        anchor_train_js, anchor_train_l2, _ = eval_matrix(qs_by_time, qt_by_time, train_times, matrix, 0)
        anchor_test_js, anchor_test_l2, _ = eval_matrix(qs_by_time, qt_by_time, test_times, matrix, 0)
        rows.append({
            "fold_id": fold_id,
            "alignment_type": "anchor_correspondence",
            "train_js": anchor_train_js,
            "test_js": anchor_test_js,
            "train_l2": anchor_train_l2,
            "test_l2": anchor_test_l2,
            "selected_delta": 0,
            "uses_target_label": 0,
        })
        matrix_rows.append({
            "fold_id": fold_id,
            "alignment_type": "anchor_correspondence",
            **matrix_summary(matrix, mean_source_times),
            "matrix_json": json.dumps(matrix.round(6).tolist()),
        })

        qs_train_shift, qt_train_shift = paired_rows_for_times(qs_by_time, qt_by_time, train_times, best_delta)
        matrix_shift = fit_correspondence_matrix(qs_train_shift, qt_train_shift, args.matrix_fit_method)
        shift_anchor_train_js, shift_anchor_train_l2, _ = eval_matrix(
            qs_by_time, qt_by_time, train_times, matrix_shift, best_delta
        )
        shift_anchor_test_js, shift_anchor_test_l2, _ = eval_matrix(
            qs_by_time, qt_by_time, test_times, matrix_shift, best_delta
        )
        rows.append({
            "fold_id": fold_id,
            "alignment_type": "anchor_correspondence_after_global_shift",
            "train_js": shift_anchor_train_js,
            "test_js": shift_anchor_test_js,
            "train_l2": shift_anchor_train_l2,
            "test_l2": shift_anchor_test_l2,
            "selected_delta": best_delta,
            "uses_target_label": 0,
        })
        matrix_rows.append({
            "fold_id": fold_id,
            "alignment_type": "anchor_correspondence_after_global_shift",
            **matrix_summary(matrix_shift, mean_source_times),
            "matrix_json": json.dumps(matrix_shift.round(6).tolist()),
        })
    return rows, matrix_rows


def make_random_centroids(source_flat, k_value, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(source_flat.shape[0], size=k_value, replace=False)
    return source_flat[idx].copy()


def make_time_only_assignments(positions, k_value):
    flat_pos = positions.reshape(-1)
    edges = np.linspace(flat_pos.min(), flat_pos.max() + 1e-6, k_value + 1)
    ids = np.clip(np.digitize(flat_pos, edges[1:-1], right=False), 0, k_value - 1)
    q = np.zeros((flat_pos.shape[0], k_value), dtype=np.float32)
    q[np.arange(flat_pos.shape[0]), ids] = 1.0
    return q


def run_controls(args, source_flat, source_positions, target_flat, target_positions, qs_true, qt_true, tau):
    rows = []
    fields_common = {
        "task": args.task,
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        "checkpoint_config": args.checkpoint_config,
        "K": args.K,
    }

    controls = []
    random_centroids = make_random_centroids(source_flat, args.K, args.seed + 17)
    q_source_random = soft_assign(source_flat, random_centroids, tau)
    q_target_random = soft_assign(target_flat, random_centroids, tau)
    controls.append(("random_anchor_control", q_source_random, q_target_random))

    controls.append((
        "time_only_anchor_control",
        make_time_only_assignments(source_positions, args.K),
        make_time_only_assignments(target_positions, args.K),
    ))

    rng = np.random.default_rng(args.seed + 101)
    shuffled_positions = target_positions.copy()
    unique_times = np.unique(shuffled_positions)
    shuffled_times = unique_times.copy()
    rng.shuffle(shuffled_times)
    mapping = {int(a): int(b) for a, b in zip(unique_times, shuffled_times)}
    shuffled_target_positions = np.vectorize(lambda x: mapping[int(x)])(shuffled_positions)
    controls.append(("shuffled_target_time_control", qs_true, qt_true, shuffled_target_positions))

    for control in controls:
        if control[0] == "shuffled_target_time_control":
            control_type, q_source, q_target, target_pos = control
        else:
            control_type, q_source, q_target = control
            target_pos = target_positions
        qs_by_time, _ = aggregate_by_time(q_source, source_positions)
        qt_by_time, _ = aggregate_by_time(q_target, target_pos)
        folds = temporal_block_folds(sorted(qt_by_time), args.cv_folds if len(qt_by_time) >= args.cv_folds else 3)
        for fold_id, (train_times, test_times) in enumerate(folds):
            qs_train, qt_train = paired_rows_for_times(qs_by_time, qt_by_time, train_times, 0)
            matrix = fit_correspondence_matrix(qs_train, qt_train, args.matrix_fit_method)
            train_js, train_l2, _ = eval_matrix(qs_by_time, qt_by_time, train_times, matrix, 0)
            test_js, test_l2, _ = eval_matrix(qs_by_time, qt_by_time, test_times, matrix, 0)
            rows.append({
                **fields_common,
                "control_type": control_type,
                "fold_id": fold_id,
                "alignment_type": "anchor_correspondence",
                "train_js": train_js,
                "test_js": test_js,
                "train_l2": train_l2,
                "test_l2": test_l2,
                "selected_delta": 0,
                "uses_target_label": 0,
            })
    return rows


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
        source_loader = build_loader(
            args,
            args.source,
            classes,
            splits[args.source]["train"],
            max_samples=args.max_source_samples,
        )
        target_loader = build_loader(
            args,
            args.target,
            classes,
            splits[args.target]["train"],
            max_samples=args.max_target_samples,
        )
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
        temporal_persistence = compute_temporal_persistence(hard_source, source_features.shape)

        anchor_counts = np.bincount(hard_source, minlength=args.K).astype(np.float64)
        mean_source_times = []
        codebook_rows = []
        for anchor_id in range(args.K):
            mask = hard_source == anchor_id
            times = source_flat["positions"][mask]
            mean_time = float(times.mean()) if times.size else float("nan")
            std_time = float(times.std()) if times.size else float("nan")
            mean_source_times.append(mean_time if not math.isnan(mean_time) else 0.0)
            codebook_rows.append({
                "task": args.task,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "checkpoint_config": args.checkpoint_config,
                "K": args.K,
                "anchor_id": anchor_id,
                "centroid_vector_json": json.dumps([float(x) for x in centroids[anchor_id]]),
                "source_anchor_count": int(anchor_counts[anchor_id]),
                "source_anchor_fraction": float(anchor_counts[anchor_id] / max(anchor_counts.sum(), 1.0)),
                "mean_source_time": mean_time,
                "std_source_time": std_time,
            })

        source_entropy = entropy_rows(q_source)
        target_entropy = entropy_rows(q_target)
        assignment_rows = []
        for domain, q in [("source", q_source), ("target", q_target)]:
            entropy = entropy_rows(q)
            max_prob = q.max(axis=1)
            dist = q.mean(axis=0)
            assignment_rows.append({
                "task": args.task,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "checkpoint_config": args.checkpoint_config,
                "K": args.K,
                "tau": tau,
                "domain": domain,
                "split": "train",
                "assignment_entropy_mean": float(entropy.mean()),
                "assignment_entropy_std": float(entropy.std()),
                "max_assignment_prob_mean": float(max_prob.mean()),
                "max_assignment_prob_std": float(max_prob.std()),
                "effective_anchor_count": distribution_effective_count(dist),
                "anchor_distribution_json": json.dumps([float(x) for x in dist]),
            })

        metric_args = {
            "K": args.K,
            "seed": args.seed,
            "max_silhouette_timepoints": args.max_silhouette_timepoints,
            "_centroids": centroids,
            "_temporal_persistence": temporal_persistence,
        }
        anchorability = anchorability_metrics(
            metric_args,
            source_flat["features"],
            source_flat["labels"],
            source_flat["positions"],
            q_source,
            hard_source,
        )
        anchorability_row = {
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            "K": args.K,
            **anchorability,
        }

        qs_by_time, source_time_counts = aggregate_by_time(q_source, source_data["positions"])
        qt_by_time, target_time_counts = aggregate_by_time(q_target, target_data["positions"])
        time_rows = []
        time_rows.extend(rows_for_time_distribution(
            args.task, args.source, args.target, args.seed, args.checkpoint_config, args.K,
            "source", qs_by_time, source_time_counts,
        ))
        time_rows.extend(rows_for_time_distribution(
            args.task, args.source, args.target, args.seed, args.checkpoint_config, args.K,
            "target", qt_by_time, target_time_counts,
        ))

        align_rows, matrix_rows = run_alignment_suite(
            args,
            qs_by_time,
            qt_by_time,
            q_source,
            q_target,
            source_data["positions"],
            target_data["positions"],
            source_data["labels"],
            target_data["labels"],
            mean_source_times,
        )
        control_rows = run_controls(
            args,
            source_flat["features"],
            source_data["positions"],
            target_flat["features"],
            target_data["positions"],
            q_source,
            q_target,
            tau,
        )

        probe_idx = sample_rows(source_flat["features"].shape[0], args.max_stability_points, args.seed)
        probe_keys = []
        n_samples, time_steps = source_data["positions"].shape
        for flat_idx in probe_idx:
            sample_id = flat_idx // time_steps
            time_id = flat_idx % time_steps
            probe_keys.append(f"{int(source_data['indices'][sample_id])}:{int(source_data['positions'][sample_id, time_id])}")
        probe_rows = [
            {
                "task": args.task,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "checkpoint_config": args.checkpoint_config,
                "K": args.K,
                "probe_key": key,
                "hard_anchor": int(hard_source[idx]),
            }
            for key, idx in zip(probe_keys, probe_idx)
        ]

        meta = {
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            "feature_stage": "model.spatial_encoder raw temporal feature",
            "feature_shape_source": list(source_features.shape),
            "feature_shape_target": list(target_features.shape),
            "time_axis_definition": "PixelSetData positions / days_after start_date",
            "normalization_method": "z-score fit on source train timepoints for each checkpoint",
            "normalization_mean": [float(x) for x in mean_vec],
            "normalization_std": [float(x) for x in std_vec],
            "checkpoint_path": str(checkpoint),
            "dataset_split": "source train codebook; source train and target train diagnostic",
            "K": args.K,
            "tau": tau,
            "max_source_samples": args.max_source_samples,
            "max_target_samples": args.max_target_samples,
            "max_codebook_timepoints": args.max_codebook_timepoints,
        }
        write_json(output_dir / "feature_extraction_meta.json", meta)
        write_tsv(output_dir / "anchor_codebook.tsv", codebook_rows, ANCHOR_CODEBOOK_FIELDS)
        write_tsv(output_dir / "anchor_assignment_summary.tsv", assignment_rows, ASSIGNMENT_FIELDS)
        write_tsv(output_dir / "anchorability_summary.tsv", [anchorability_row], ANCHORABILITY_FIELDS)
        write_tsv(output_dir / "anchor_time_distribution.tsv", time_rows, TIME_DIST_FIELDS)
        write_tsv(output_dir / "alignment_discrepancy_results.tsv", [
            decorate_common(row, args) for row in align_rows
        ], ALIGNMENT_FIELDS)
        write_tsv(output_dir / "anchor_correspondence_matrix_summary.tsv", [
            decorate_common(row, args) for row in matrix_rows
        ], MATRIX_FIELDS)
        write_tsv(output_dir / "alignment_control_results.tsv", control_rows, CONTROL_FIELDS)
        write_tsv(output_dir / "anchor_probe_assignments.tsv", probe_rows, PROBE_FIELDS)
        write_tsv(output_dir / "failed_runs.tsv", [], FAILED_FIELDS)
    except Exception as exc:
        failed_rows.append({
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "checkpoint_config": args.checkpoint_config,
            "K": args.K,
            "status": "failed",
            "error": repr(exc),
        })
        write_tsv(output_dir / "failed_runs.tsv", failed_rows, FAILED_FIELDS)
        raise


def decorate_common(row, args):
    return {
        "task": args.task,
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        "checkpoint_config": args.checkpoint_config,
        "K": args.K,
        **row,
    }


ANCHOR_CODEBOOK_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "anchor_id",
    "centroid_vector_json", "source_anchor_count", "source_anchor_fraction",
    "mean_source_time", "std_source_time",
]
ASSIGNMENT_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "tau", "domain",
    "split", "assignment_entropy_mean", "assignment_entropy_std",
    "max_assignment_prob_mean", "max_assignment_prob_std", "effective_anchor_count",
    "anchor_distribution_json",
]
ANCHORABILITY_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K",
    "cluster_compactness", "silhouette_score_sampled", "assignment_entropy_mean",
    "max_assignment_prob_mean", "effective_anchor_count", "temporal_persistence",
    "class_conditional_anchor_entropy", "class_conditional_anchor_purity",
]
TIME_DIST_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "domain",
    "time_bin", "anchor_distribution_json", "n_timepoints",
]
ALIGNMENT_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "fold_id",
    "alignment_type", "train_js", "test_js", "train_l2", "test_l2",
    "selected_delta", "uses_target_label",
]
MATRIX_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "fold_id",
    "alignment_type", "row_entropy_mean", "col_entropy_mean", "diag_mass",
    "max_entry_mean", "effective_matches", "best_shift_permutation_residual",
    "matrix_json",
]
CONTROL_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "control_type",
    "fold_id", "alignment_type", "train_js", "test_js", "train_l2", "test_l2",
    "selected_delta", "uses_target_label",
]
PROBE_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "probe_key", "hard_anchor",
]
FAILED_FIELDS = [
    "task", "source", "target", "seed", "checkpoint_config", "K", "status", "error",
]


def collect_per_run_files(input_root, filename):
    return sorted(Path(input_root).glob(f"*/*/{filename}")) + sorted(Path(input_root).glob(f"*/{filename}"))


def aggregate_files(input_root, output_dir, filename, fields):
    rows = []
    for path in collect_per_run_files(input_root, filename):
        rows.extend(read_tsv(path))
    write_tsv(Path(output_dir) / filename, rows, fields)
    return rows


def aggregate_failed(input_root, output_dir):
    rows = []
    for path in collect_per_run_files(input_root, "failed_runs.tsv"):
        rows.extend(read_tsv(path))
    rows = [row for row in rows if row.get("status")]
    write_tsv(Path(output_dir) / "failed_runs.tsv", rows, FAILED_FIELDS)
    return rows


def aggregate_meta(input_root, output_dir):
    metas = []
    for path in sorted(Path(input_root).glob("*/*/feature_extraction_meta.json")) + sorted(Path(input_root).glob("*/feature_extraction_meta.json")):
        with path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["_path"] = str(path)
        metas.append(meta)
    write_json(Path(output_dir) / "feature_extraction_meta.json", metas)
    return metas


def aggregate_seed_stability(probe_rows, codebook_rows, output_dir):
    by_key = defaultdict(list)
    for row in probe_rows:
        key = (row["source"], row["checkpoint_config"], row["K"], row["task"])
        by_key[key].append(row)
    codebooks = defaultdict(list)
    for row in codebook_rows:
        key = (row["source"], row["checkpoint_config"], row["K"], row["task"], row["seed"])
        codebooks[key].append(row)
    rows = []
    for key, group in by_key.items():
        source, config, k_value, task = key
        by_seed = defaultdict(dict)
        for row in group:
            by_seed[row["seed"]][row["probe_key"]] = int(float(row["hard_anchor"]))
        seeds = sorted(by_seed, key=lambda x: int(float(x)))
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                seed_a, seed_b = seeds[i], seeds[j]
                common = sorted(set(by_seed[seed_a]) & set(by_seed[seed_b]))
                if len(common) < 2:
                    nmi = ari = float("nan")
                else:
                    labels_a = [by_seed[seed_a][x] for x in common]
                    labels_b = [by_seed[seed_b][x] for x in common]
                    nmi = float(normalized_mutual_info_score(labels_a, labels_b))
                    ari = float(adjusted_rand_score(labels_a, labels_b))
                cent_a = rows_to_centroid_array(codebooks.get((source, config, k_value, task, seed_a), []))
                cent_b = rows_to_centroid_array(codebooks.get((source, config, k_value, task, seed_b), []))
                dist = matched_centroid_distance(cent_a, cent_b)
                rows.append({
                    "source": source,
                    "checkpoint_config": config,
                    "K": k_value,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "centroid_distance_after_matching": dist,
                    "assignment_nmi": nmi,
                    "assignment_ari": ari,
                })
    write_tsv(Path(output_dir) / "anchor_seed_stability.tsv", rows, [
        "source", "checkpoint_config", "K", "seed_a", "seed_b",
        "centroid_distance_after_matching", "assignment_nmi", "assignment_ari",
    ])
    return rows


def rows_to_centroid_array(rows):
    if not rows:
        return np.empty((0, 0))
    rows = sorted(rows, key=lambda r: int(float(r["anchor_id"])))
    return np.asarray([json.loads(row["centroid_vector_json"]) for row in rows], dtype=np.float64)


def matched_centroid_distance(a, b):
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return float("nan")
    try:
        from scipy.optimize import linear_sum_assignment
        cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2) ** 0.5
        row, col = linear_sum_assignment(cost)
        return float(cost[row, col].mean())
    except Exception:
        return float("nan")


def average_by_type(rows, value_field="test_js"):
    by_type = defaultdict(list)
    for row in rows:
        by_type[row.get("alignment_type", "")].append(row.get(value_field))
    return {key: mean(vals) for key, vals in by_type.items()}


def make_non_shiftness_summary(alignment_rows, control_rows, matrix_rows, output_dir):
    grouped = defaultdict(list)
    for row in alignment_rows:
        key = (
            row["task"], row["source"], row["target"], row["seed"],
            row["checkpoint_config"], row["K"],
        )
        grouped[key].append(row)
    controls = defaultdict(list)
    for row in control_rows:
        key = (
            row["task"], row["source"], row["target"], row["seed"],
            row["checkpoint_config"], row["K"], row["control_type"],
        )
        controls[key].append(row)
    matrices = defaultdict(list)
    for row in matrix_rows:
        key = (
            row["task"], row["source"], row["target"], row["seed"],
            row["checkpoint_config"], row["K"],
        )
        matrices[key].append(row)
    rows = []
    for key, group in sorted(grouped.items()):
        task, source, target, seed, config, k_value = key
        values = average_by_type(group)
        random_js = mean([r.get("test_js") for r in controls.get((*key, "random_anchor_control"), [])])
        time_js = mean([r.get("test_js") for r in controls.get((*key, "time_only_anchor_control"), [])])
        shuffled_js = mean([r.get("test_js") for r in controls.get((*key, "shuffled_target_time_control"), [])])
        anchor_js = values.get("anchor_correspondence")
        best_js = values.get("best_global_scalar_shift_cv")
        tm_js = values.get("timematch_estimated_global_shift")
        no_js = values.get("no_alignment")
        oracle_js = values.get("per_class_scalar_shift_oracle")
        anchor_shift_js = values.get("anchor_correspondence_after_global_shift")
        matrix_group = matrices.get(key, [])
        rows.append({
            "task": task,
            "source": source,
            "target": target,
            "seed": seed,
            "checkpoint_config": config,
            "K": k_value,
            "no_alignment_test_js": no_js,
            "timematch_shift_test_js": tm_js,
            "best_global_shift_test_js": best_js,
            "per_class_scalar_oracle_test_js": oracle_js,
            "anchor_correspondence_test_js": anchor_js,
            "anchor_after_global_shift_test_js": anchor_shift_js,
            "anchor_gain_vs_no_alignment": diff(no_js, anchor_js),
            "anchor_gain_vs_timematch_shift": diff(tm_js, anchor_js),
            "anchor_gain_vs_best_global_shift": diff(best_js, anchor_js),
            "anchor_gain_vs_per_class_oracle": diff(oracle_js, anchor_js),
            "anchor_after_shift_gain_vs_best_global_shift": diff(best_js, anchor_shift_js),
            "random_anchor_test_js": random_js,
            "time_only_anchor_test_js": time_js,
            "shuffled_target_test_js": shuffled_js,
            "anchor_gain_vs_random": diff(random_js, anchor_js),
            "anchor_gain_vs_time_only": diff(time_js, anchor_js),
            "anchor_gain_vs_shuffled": diff(shuffled_js, anchor_js),
            "best_shift_permutation_residual_mean": mean(
                [r.get("best_shift_permutation_residual") for r in matrix_group]
            ),
            "diag_mass_mean": mean([r.get("diag_mass") for r in matrix_group]),
            "row_entropy_mean": mean([r.get("row_entropy_mean") for r in matrix_group]),
        })
    fields = [
        "task", "source", "target", "seed", "checkpoint_config", "K",
        "no_alignment_test_js", "timematch_shift_test_js", "best_global_shift_test_js",
        "per_class_scalar_oracle_test_js", "anchor_correspondence_test_js",
        "anchor_after_global_shift_test_js", "anchor_gain_vs_no_alignment",
        "anchor_gain_vs_timematch_shift", "anchor_gain_vs_best_global_shift",
        "anchor_gain_vs_per_class_oracle", "anchor_after_shift_gain_vs_best_global_shift",
        "random_anchor_test_js", "time_only_anchor_test_js", "shuffled_target_test_js",
        "anchor_gain_vs_random", "anchor_gain_vs_time_only", "anchor_gain_vs_shuffled",
        "best_shift_permutation_residual_mean", "diag_mass_mean", "row_entropy_mean",
    ]
    write_tsv(Path(output_dir) / "non_shiftness_summary.tsv", rows, fields)
    return rows


def diff(a, b):
    a = safe_float(a)
    b = safe_float(b)
    if a is None or b is None:
        return None
    return a - b


def write_summary(output_dir, anchorability_rows, stability_rows, non_shift_rows, failed_rows):
    output_dir = Path(output_dir)
    config_groups = defaultdict(list)
    for row in anchorability_rows:
        config_groups[row["checkpoint_config"]].append(row)
    non_groups = defaultdict(list)
    for row in non_shift_rows:
        non_groups[row["checkpoint_config"]].append(row)

    def avg(rows, field):
        return mean([r.get(field) for r in rows])

    def count_positive(rows, field):
        vals = [safe_float(r.get(field)) for r in rows]
        vals = [v for v in vals if v is not None]
        return sum(1 for v in vals if v > 0), len(vals)

    lines = []
    lines.append("# v3.0.1 Anchor Correspondence Diagnostic Summary\n")
    lines.append("## 1. Experiment Setup\n")
    lines.append("- version: `v301_anchor_correspondence_diagnostic`\n")
    lines.append("- feature stage: `model.spatial_encoder` raw temporal feature\n")
    lines.append("- anchors: source-defined KMeans codebook, main K=6\n")
    lines.append("- alignment metric: heldout-time-bin JS divergence, L2 as auxiliary\n")
    lines.append(f"- failed runs: {len(failed_rows)}\n")

    lines.append("\n## 2. Anchorability: Plain vs Smooth\n")
    lines.append("| config | n | cluster compactness | assignment entropy | max assignment prob | temporal persistence | class anchor entropy | class anchor purity |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for config, rows in sorted(config_groups.items()):
        lines.append(
            f"| {config} | {len(rows)} | {fmt(avg(rows, 'cluster_compactness'))} | "
            f"{fmt(avg(rows, 'assignment_entropy_mean'))} | {fmt(avg(rows, 'max_assignment_prob_mean'))} | "
            f"{fmt(avg(rows, 'temporal_persistence'))} | {fmt(avg(rows, 'class_conditional_anchor_entropy'))} | "
            f"{fmt(avg(rows, 'class_conditional_anchor_purity'))} |\n"
        )

    lines.append("\n## 3. Alignment Discrepancy Comparison\n")
    lines.append("| config | n | no align | timematch shift | best global shift | per-class oracle | anchor | anchor after shift |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for config, rows in sorted(non_groups.items()):
        lines.append(
            f"| {config} | {len(rows)} | {fmt(avg(rows, 'no_alignment_test_js'))} | "
            f"{fmt(avg(rows, 'timematch_shift_test_js'))} | {fmt(avg(rows, 'best_global_shift_test_js'))} | "
            f"{fmt(avg(rows, 'per_class_scalar_oracle_test_js'))} | {fmt(avg(rows, 'anchor_correspondence_test_js'))} | "
            f"{fmt(avg(rows, 'anchor_after_global_shift_test_js'))} |\n"
        )

    lines.append("\n## 4. Correspondence Matrix Diagnostics\n")
    lines.append("| config | shifted diagonal residual | diag mass | row entropy |\n")
    lines.append("|---|---:|---:|---:|\n")
    for config, rows in sorted(non_groups.items()):
        lines.append(
            f"| {config} | {fmt(avg(rows, 'best_shift_permutation_residual_mean'))} | "
            f"{fmt(avg(rows, 'diag_mass_mean'))} | {fmt(avg(rows, 'row_entropy_mean'))} |\n"
        )

    lines.append("\n## 5. Control Experiments\n")
    lines.append("| config | anchor vs random positive | anchor vs time-only positive | anchor vs shuffled positive |\n")
    lines.append("|---|---:|---:|---:|\n")
    for config, rows in sorted(non_groups.items()):
        r_pos, r_n = count_positive(rows, "anchor_gain_vs_random")
        t_pos, t_n = count_positive(rows, "anchor_gain_vs_time_only")
        s_pos, s_n = count_positive(rows, "anchor_gain_vs_shuffled")
        lines.append(f"| {config} | {r_pos}/{r_n} | {t_pos}/{t_n} | {s_pos}/{s_n} |\n")

    lines.append("\n## 6. Non-shiftness Evidence\n")
    lines.append("| config | anchor > best global | anchor after shift > best global | anchor > per-class oracle |\n")
    lines.append("|---|---:|---:|---:|\n")
    for config, rows in sorted(non_groups.items()):
        a_pos, a_n = count_positive(rows, "anchor_gain_vs_best_global_shift")
        as_pos, as_n = count_positive(rows, "anchor_after_shift_gain_vs_best_global_shift")
        o_pos, o_n = count_positive(rows, "anchor_gain_vs_per_class_oracle")
        lines.append(f"| {config} | {a_pos}/{a_n} | {as_pos}/{as_n} | {o_pos}/{o_n} |\n")

    lines.append("\n## 7. Optional Pseudo-label Diagnostic\n")
    lines.append("Skipped. This diagnostic does not train DA and does not define a natural anchor-adjusted pseudo-label policy.\n")

    lines.append("\n## 8. Minimal Conclusion\n")
    lines.append("This file reports diagnostic evidence only. It does not claim a new UDA method or update the theory.\n")

    (output_dir / "v301_anchor_correspondence_diagnostic_summary.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def aggregate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = args.input_root or str(output_dir / "runs")
    aggregate_meta(input_root, output_dir)
    codebook_rows = aggregate_files(input_root, output_dir, "anchor_codebook.tsv", ANCHOR_CODEBOOK_FIELDS)
    assignment_rows = aggregate_files(input_root, output_dir, "anchor_assignment_summary.tsv", ASSIGNMENT_FIELDS)
    anchorability_rows = aggregate_files(input_root, output_dir, "anchorability_summary.tsv", ANCHORABILITY_FIELDS)
    time_rows = aggregate_files(input_root, output_dir, "anchor_time_distribution.tsv", TIME_DIST_FIELDS)
    alignment_rows = aggregate_files(input_root, output_dir, "alignment_discrepancy_results.tsv", ALIGNMENT_FIELDS)
    matrix_rows = aggregate_files(input_root, output_dir, "anchor_correspondence_matrix_summary.tsv", MATRIX_FIELDS)
    control_rows = aggregate_files(input_root, output_dir, "alignment_control_results.tsv", CONTROL_FIELDS)
    probe_rows = aggregate_files(input_root, output_dir, "anchor_probe_assignments.tsv", PROBE_FIELDS)
    failed_rows = aggregate_failed(input_root, output_dir)
    stability_rows = aggregate_seed_stability(probe_rows, codebook_rows, output_dir)
    non_shift_rows = make_non_shiftness_summary(alignment_rows, control_rows, matrix_rows, output_dir)
    write_summary(output_dir, anchorability_rows, stability_rows, non_shift_rows, failed_rows)
    # Keep optional pseudo-label explicit.
    write_tsv(output_dir / "pseudo_label_anchor_diagnostic.tsv", [], [
        "task", "source", "target", "seed", "checkpoint_config", "K",
        "pseudo_label_policy", "coverage", "confidence_mean", "class_entropy",
        "effective_class_count", "target_macro_f1_offline",
    ])


def main():
    args = parse_args()
    if args.mode == "aggregate":
        aggregate(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
