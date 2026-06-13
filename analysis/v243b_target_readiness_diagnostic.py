#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.recompute_transfer_metrics import (  # noqa: E402
    build_dataset,
    build_loader,
    build_model,
    get_task_classes,
    load_checkpoint,
)
from analysis.v243b_frozen_encoder_transfer_probe import (  # noqa: E402
    checkpoint_from_log,
    filter_jobs,
    mean,
    pearson,
    read_jobs,
    safe_float,
    spearman,
    stdev,
    write_tsv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose target feature structure of v2.4.3b raw compactness checkpoints. "
            "This is a post-hoc mechanism diagnostic, not a selector."
        )
    )
    parser.add_argument("log_dir", help="Experiment log directory containing jobs.tsv and logs.")
    parser.add_argument("--jobs_tsv", default="", help="Defaults to <log_dir>/jobs.tsv.")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--output_prefix", default="target_readiness")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--configs", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="pseltae", choices=["pseltae", "psetae", "psetcnn", "psegru"])
    parser.add_argument("--with_extra", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument("--max_metric_samples", type=int, default=2048)
    parser.add_argument("--feature_kind", default="final", choices=["final", "raw_pooled", "raw_flat"])
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8f}"
    return str(value)


def read_optional_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@torch.no_grad()
def collect_target_features(model, loader, device, max_batches, feature_kind):
    features = []
    labels = []
    logits_list = []
    for batch_idx, sample in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        pixels = sample["pixels"].to(device=device, non_blocking=True)
        valid_pixels = sample["valid_pixels"].to(device=device, non_blocking=True)
        positions = sample["positions"].to(device=device, non_blocking=True)
        extra = sample["extra"].to(device=device, non_blocking=True)
        y = sample["label"].to(device=device, non_blocking=True)

        if feature_kind == "final":
            logits, feats = model.forward(pixels, valid_pixels, positions, extra, return_feats=True)
        else:
            raw = model.spatial_encoder(pixels, valid_pixels, extra)
            logits, _ = model.forward(pixels, valid_pixels, positions, extra, return_feats=True)
            if feature_kind == "raw_pooled":
                feats = raw.mean(dim=1)
            else:
                feats = raw.reshape(raw.shape[0], -1)
        if feats.ndim > 2:
            feats = feats.reshape(feats.shape[0], -1)
        features.append(feats.detach().cpu().float())
        labels.append(y.detach().cpu().long())
        logits_list.append(logits.detach().cpu().float())

    if not features:
        return None
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "logits": torch.cat(logits_list, dim=0),
    }


def macro_f1(pred, labels, num_classes):
    scores = []
    for class_id in range(num_classes):
        tp = ((pred == class_id) & (labels == class_id)).sum().item()
        fp = ((pred == class_id) & (labels != class_id)).sum().item()
        fn = ((pred != class_id) & (labels == class_id)).sum().item()
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(scores) / max(len(scores), 1)


def effective_rank_and_spectrum(features, eps=1e-12):
    if features.shape[0] < 2:
        return {
            "target_effrank": None,
            "target_effrank_ratio": None,
            "target_top_eig_ratio": None,
            "target_cov_trace": None,
        }
    centered = features - features.mean(dim=0, keepdim=True)
    cov = centered.t().matmul(centered) / max(features.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum()
    if total <= eps:
        return {
            "target_effrank": 0.0,
            "target_effrank_ratio": 0.0,
            "target_top_eig_ratio": 1.0,
            "target_cov_trace": 0.0,
        }
    probs = eigvals / total
    entropy = -(probs * probs.clamp_min(eps).log()).sum()
    effrank = float(torch.exp(entropy).item())
    return {
        "target_effrank": effrank,
        "target_effrank_ratio": effrank / max(int(features.shape[1]), 1),
        "target_top_eig_ratio": float((eigvals.max() / total).item()),
        "target_cov_trace": float(total.item()),
    }


def subsample_for_metrics(features, pseudo, labels, max_samples, seed):
    if max_samples <= 0 or features.shape[0] <= max_samples:
        return features, pseudo, labels
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(features.shape[0], generator=generator)[:max_samples]
    return features[indices], pseudo[indices], labels[indices]


def pseudo_silhouette(features, pseudo, eps=1e-12):
    labels = torch.unique(pseudo)
    valid_labels = [label for label in labels.tolist() if int((pseudo == label).sum().item()) >= 2]
    if len(valid_labels) < 2:
        return None
    dists = torch.cdist(features, features, p=2)
    scores = []
    for idx in range(features.shape[0]):
        own = pseudo[idx]
        if own.item() not in valid_labels:
            continue
        own_mask = pseudo == own
        own_count = int(own_mask.sum().item())
        if own_count < 2:
            continue
        a = dists[idx, own_mask].sum() / max(own_count - 1, 1)
        b_values = []
        for label in valid_labels:
            if label == own.item():
                continue
            mask = pseudo == label
            b_values.append(dists[idx, mask].mean())
        if not b_values:
            continue
        b = torch.stack(b_values).min()
        denom = torch.maximum(a, b).clamp_min(eps)
        scores.append(((b - a) / denom).item())
    return None if not scores else float(sum(scores) / len(scores))


def pseudo_davies_bouldin(features, pseudo, eps=1e-12):
    labels = [label for label in torch.unique(pseudo).tolist() if int((pseudo == label).sum().item()) >= 2]
    if len(labels) < 2:
        return None
    centroids = []
    scatters = []
    for label in labels:
        group = features[pseudo == label]
        centroid = group.mean(dim=0)
        centroids.append(centroid)
        scatters.append(torch.linalg.norm(group - centroid, dim=1).mean())
    centroids = torch.stack(centroids)
    scatters = torch.stack(scatters)
    center_dists = torch.cdist(centroids, centroids, p=2).clamp_min(eps)
    ratios = (scatters[:, None] + scatters[None, :]) / center_dists
    ratios.fill_diagonal_(float("-inf"))
    return float(ratios.max(dim=1).values.mean().item())


def label_distribution_metrics(pseudo, num_classes, eps=1e-12):
    counts = torch.bincount(pseudo, minlength=num_classes).float()
    probs = counts / counts.sum().clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum()
    norm_entropy = entropy / math.log(max(num_classes, 2))
    return {
        "pseudo_class_entropy": float(norm_entropy.item()),
        "pseudo_effective_classes": float(torch.exp(entropy).item()),
        "pseudo_dominant_ratio": float(probs.max().item()),
        "pseudo_nonempty_classes": int((counts > 0).sum().item()),
    }


def compute_metrics(data, num_classes, args, row_seed):
    features = data["features"]
    logits = data["logits"]
    labels = data["labels"]
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    confidence, pseudo = probs.max(dim=1)
    pred_metrics = {
        "target_head_acc": float((pseudo == labels).float().mean().item()),
        "target_head_macro_f1": macro_f1(pseudo, labels, num_classes),
        "target_entropy": float(entropy.mean().item()),
        "target_entropy_norm": float((entropy / math.log(max(num_classes, 2))).mean().item()),
        "target_confidence": float(confidence.mean().item()),
        "target_conf_ge_090": float((confidence >= 0.90).float().mean().item()),
        "target_conf_ge_095": float((confidence >= 0.95).float().mean().item()),
        "target_feature_norm": float(torch.linalg.norm(features, dim=1).mean().item()),
        "target_feature_norm_std": float(torch.linalg.norm(features, dim=1).std().item()),
    }
    spectrum = effective_rank_and_spectrum(features)
    distribution = label_distribution_metrics(pseudo, num_classes)

    metric_features, metric_pseudo, _ = subsample_for_metrics(
        features, pseudo, labels, args.max_metric_samples, args.seed + row_seed
    )
    cluster = {
        "pseudo_silhouette": pseudo_silhouette(metric_features, metric_pseudo),
        "pseudo_davies_bouldin": pseudo_davies_bouldin(metric_features, metric_pseudo),
        "metric_sample_count": int(metric_features.shape[0]),
    }
    out = {}
    out.update(pred_metrics)
    out.update(spectrum)
    out.update(distribution)
    out.update(cluster)
    return out


def run_row(row, args, device):
    checkpoint = checkpoint_from_log(args.log_dir, row, args.outputs_root)
    out = {
        "task": row["task"],
        "seed": row["seed"],
        "config": row["config"],
        "compact_weight": safe_float(row.get("compact_weight")),
        "checkpoint": "" if checkpoint is None else str(checkpoint),
        "status": "ok",
    }
    if checkpoint is None:
        out["status"] = "missing_checkpoint"
        return out

    classes = get_task_classes(args.data_root, row["source_dataset"], closed_set=True)
    model = build_model(args.model, len(classes), args.with_extra, device)
    load_checkpoint(model, checkpoint, device)
    target_ds = build_dataset(args.data_root, row["target_dataset"], classes, True, args.with_extra)
    target_loader = build_loader(target_ds, args.batch_size, args.num_workers)
    data = collect_target_features(model, target_loader, device, args.max_batches, args.feature_kind)
    if data is None:
        out["status"] = "empty_loader"
        return out
    out.update(
        {
            "target_count": int(data["labels"].numel()),
            "feature_dim": int(data["features"].shape[1]),
        }
    )
    out.update(compute_metrics(data, len(classes), args, int(row["seed"])))
    return out


def summarize(rows, group_fields, value_fields):
    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(field) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item["n"] = len(group)
        for field in value_fields:
            values = [row.get(field) for row in group if row.get(field) is not None]
            item[f"{field}_mean"] = mean(values)
            item[f"{field}_std"] = stdev(values)
        out.append(item)
    return out


def build_curve_attribution(log_dir, readiness_curve):
    dose_curve = read_optional_tsv(Path(log_dir) / "dose_response_curve.tsv")
    if not dose_curve:
        return []

    def key(row):
        weight = safe_float(row.get("compact_weight"))
        return row.get("task"), None if weight is None else round(weight, 8)

    dose_by_key = {key(row): row for row in dose_curve}
    merged = []
    for row in readiness_curve:
        dose = dose_by_key.get(key(row))
        if dose is None:
            continue
        item = dict(row)
        for field in [
            "source_on_target_f1_mean",
            "da_f1_mean",
            "initial_all_f1_mean",
            "epoch1_all_f1_mean",
            "last_all_f1_mean",
        ]:
            item[field] = safe_float(dose.get(field))
        merged.append(item)

    out = []
    by_task = {}
    for row in merged:
        by_task.setdefault(row["task"], []).append(row)
    metrics = [
        "target_head_macro_f1_mean",
        "target_entropy_norm_mean",
        "target_confidence_mean",
        "target_effrank_ratio_mean",
        "target_top_eig_ratio_mean",
        "pseudo_class_entropy_mean",
        "pseudo_effective_classes_mean",
        "pseudo_dominant_ratio_mean",
        "pseudo_silhouette_mean",
        "pseudo_davies_bouldin_mean",
    ]
    reference_metrics = [
        "da_f1_mean",
        "source_on_target_f1_mean",
        "initial_all_f1_mean",
        "epoch1_all_f1_mean",
        "last_all_f1_mean",
    ]
    for task, group in sorted(by_task.items()):
        weights = [safe_float(row.get("compact_weight")) for row in group]
        for metric in metrics:
            values = [row.get(metric) for row in group]
            item = {
                "task": task,
                "metric": metric,
                "n": len(group),
                "pearson_with_weight": pearson(weights, values),
                "spearman_with_weight": spearman(weights, values),
            }
            for ref in reference_metrics:
                item[f"pearson_with_{ref}"] = pearson([row.get(ref) for row in group], values)
                item[f"spearman_with_{ref}"] = spearman([row.get(ref) for row in group], values)
            out.append(item)
    return out


def main():
    args = parse_args()
    args.log_dir = Path(args.log_dir)
    jobs_tsv = Path(args.jobs_tsv) if args.jobs_tsv else args.log_dir / "jobs.tsv"
    jobs = filter_jobs(read_jobs(jobs_tsv), args)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = []
    for idx, row in enumerate(jobs, start=1):
        print(
            f"[{idx}/{len(jobs)}] task={row['task']} seed={row['seed']} config={row['config']}",
            flush=True,
        )
        try:
            rows.append(run_row(row, args, device))
        except Exception as exc:
            rows.append(
                {
                    "task": row.get("task", ""),
                    "seed": row.get("seed", ""),
                    "config": row.get("config", ""),
                    "compact_weight": safe_float(row.get("compact_weight")),
                    "status": f"error:{type(exc).__name__}",
                    "error": str(exc),
                }
            )

    metric_fields = [
        "target_head_macro_f1",
        "target_head_acc",
        "target_entropy_norm",
        "target_confidence",
        "target_conf_ge_090",
        "target_conf_ge_095",
        "target_effrank_ratio",
        "target_top_eig_ratio",
        "target_cov_trace",
        "target_feature_norm",
        "target_feature_norm_std",
        "pseudo_class_entropy",
        "pseudo_effective_classes",
        "pseudo_dominant_ratio",
        "pseudo_nonempty_classes",
        "pseudo_silhouette",
        "pseudo_davies_bouldin",
    ]
    row_fields = [
        "task",
        "seed",
        "config",
        "compact_weight",
        "status",
        "target_count",
        "feature_dim",
        "metric_sample_count",
        *metric_fields,
        "checkpoint",
        "error",
    ]
    curve = summarize(rows, ["task", "compact_weight"], metric_fields)
    attribution = build_curve_attribution(args.log_dir, curve)

    prefix = args.log_dir / args.output_prefix
    write_tsv(prefix.with_name(f"{prefix.name}_rows.tsv"), rows, row_fields)
    write_tsv(prefix.with_name(f"{prefix.name}_curve.tsv"), curve, list(curve[0].keys()) if curve else ["task"])
    write_tsv(
        prefix.with_name(f"{prefix.name}_attribution.tsv"),
        attribution,
        list(attribution[0].keys()) if attribution else ["task"],
    )
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_rows.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_curve.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_attribution.tsv')}")


if __name__ == "__main__":
    main()
