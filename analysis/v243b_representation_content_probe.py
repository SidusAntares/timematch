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
    mean,
    pearson,
    safe_float,
    spearman,
    stdev,
    write_tsv,
)


JOB_FIELDS = [
    "task",
    "source_dataset",
    "target_dataset",
    "seed",
    "config",
    "compact_weight",
    "compact_distance",
    "norm_preserve_trade_off",
    "norm_preserve_target",
    "norm_preserve_value",
    "est_weight",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Content-level representation probe for v2.4.3b raw compactness. "
            "It tests what compactness gradients change in source geometry and "
            "source-target prototype compatibility."
        )
    )
    parser.add_argument("log_dir", help="Experiment log directory containing jobs.tsv and job logs.")
    parser.add_argument("--jobs_tsv", default="", help="Defaults to <log_dir>/jobs.tsv.")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--output_prefix", default="representation_content")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--configs", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="pseltae", choices=["pseltae", "psetae", "psetcnn", "psegru"])
    parser.add_argument("--with_extra", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument("--max_metric_samples", type=int, default=4096)
    parser.add_argument("--bootstrap_iters", type=int, default=200)
    parser.add_argument("--feature_kind", default="raw_pooled", choices=["final", "raw_pooled", "raw_flat"])
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def parse_set(text):
    if text is None or str(text).strip() == "":
        return None
    return {part.strip() for part in str(text).split(",") if part.strip()}


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8f}"
    return str(value)


def read_jobs(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if not parts:
                continue
            if parts[0] == "task":
                continue
            row = {field: parts[idx] if idx < len(parts) else "" for idx, field in enumerate(JOB_FIELDS)}
            rows.append(row)
    return rows


def read_optional_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def filter_jobs(rows, args):
    tasks = parse_set(args.tasks)
    seeds = parse_set(args.seeds)
    configs = parse_set(args.configs)
    out = []
    seen = set()
    for row in rows:
        if tasks is not None and row["task"] not in tasks:
            continue
        if seeds is not None and row["seed"] not in seeds:
            continue
        if configs is not None and row["config"] not in configs:
            continue
        key = (row["task"], row["seed"], row["config"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def macro_f1(pred, labels, num_classes):
    scores = []
    for class_id in range(num_classes):
        tp = ((pred == class_id) & (labels == class_id)).sum().item()
        fp = ((pred == class_id) & (labels != class_id)).sum().item()
        fn = ((pred != class_id) & (labels == class_id)).sum().item()
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def collect_features(model, loader, device, max_batches, feature_kind):
    features = []
    labels = []
    logits = []
    for batch_idx, sample in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        pixels = sample["pixels"].to(device=device, non_blocking=True)
        valid_pixels = sample["valid_pixels"].to(device=device, non_blocking=True)
        positions = sample["positions"].to(device=device, non_blocking=True)
        extra = sample["extra"].to(device=device, non_blocking=True)
        y = sample["label"].to(device=device, non_blocking=True)

        if feature_kind == "final":
            batch_logits, feats = model.forward(pixels, valid_pixels, positions, extra, return_feats=True)
        else:
            raw = model.spatial_encoder(pixels, valid_pixels, extra)
            batch_logits, _ = model.forward(pixels, valid_pixels, positions, extra, return_feats=True)
            if feature_kind == "raw_pooled":
                feats = raw.mean(dim=1)
            else:
                feats = raw.reshape(raw.shape[0], -1)
        if feats.ndim > 2:
            feats = feats.reshape(feats.shape[0], -1)
        features.append(feats.detach().cpu().float())
        labels.append(y.detach().cpu().long())
        logits.append(batch_logits.detach().cpu().float())

    if not features:
        return None
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "logits": torch.cat(logits, dim=0),
    }


def maybe_subsample(data, max_samples, seed):
    if max_samples <= 0 or data["features"].shape[0] <= max_samples:
        return data
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(data["features"].shape[0], generator=generator)[:max_samples]
    return {
        "features": data["features"][indices],
        "labels": data["labels"][indices],
        "logits": data["logits"][indices],
    }


def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).t()
    return (x_norm + y_norm - 2.0 * x @ y.t()).clamp_min(0.0)


def class_prototypes(features, labels, num_classes):
    protos = []
    counts = []
    class_ids = []
    for class_id in range(num_classes):
        group = features[labels == class_id]
        if group.numel() == 0:
            continue
        protos.append(group.mean(dim=0))
        counts.append(int(group.shape[0]))
        class_ids.append(class_id)
    if not protos:
        return None, [], []
    return torch.stack(protos, dim=0), class_ids, counts


def source_geometry(features, labels, num_classes, eps=1e-12):
    protos, class_ids, counts = class_prototypes(features, labels, num_classes)
    if protos is None:
        return {}, None, [], []

    within_total = 0.0
    within_count = 0
    for row_idx, class_id in enumerate(class_ids):
        group = features[labels == class_id]
        sq = ((group - protos[row_idx].unsqueeze(0)) ** 2).sum(dim=1)
        within_total += float(sq.sum().item())
        within_count += int(sq.numel())
    source_intra = within_total / max(within_count, 1)

    inter_mean = None
    inter_nearest = None
    if protos.shape[0] >= 2:
        proto_dists = pairwise_sq_dists(protos, protos)
        mask = ~torch.eye(protos.shape[0], dtype=torch.bool)
        inter_mean = float(proto_dists[mask].mean().item())
        proto_dists = proto_dists.masked_fill(~mask, float("inf"))
        inter_nearest = float(proto_dists.min(dim=1).values.mean().item())

    metrics = {
        "source_class_count": len(class_ids),
        "source_intra_sq": source_intra,
        "source_inter_sq_mean": inter_mean,
        "source_inter_sq_nearest": inter_nearest,
        "source_inter_intra_ratio": None if inter_mean is None else inter_mean / max(source_intra, eps),
        "source_nearest_inter_intra_ratio": None
        if inter_nearest is None
        else inter_nearest / max(source_intra, eps),
        "source_feature_norm": float(torch.linalg.norm(features, dim=1).mean().item()),
        "source_cov_trace": covariance_trace(features),
        "source_min_class_count": min(counts) if counts else None,
    }
    return metrics, protos, class_ids, counts


def covariance_trace(features):
    if features.shape[0] < 2:
        return None
    centered = features - features.mean(dim=0, keepdim=True)
    return float(centered.pow(2).sum(dim=1).sum().item() / max(features.shape[0] - 1, 1))


def bootstrap_mean_ci(values, iters, seed, confidence=0.95):
    if values is None or values.numel() == 0 or iters <= 0:
        return None, None
    generator = torch.Generator().manual_seed(seed)
    means = []
    count = int(values.numel())
    for _ in range(iters):
        indices = torch.randint(0, count, (count,), generator=generator)
        means.append(float(values[indices].float().mean().item()))
    means.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    lo_idx = int(math.floor((alpha / 2.0) * (len(means) - 1)))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * (len(means) - 1)))
    return means[lo_idx], means[hi_idx]


def target_source_compatibility(
    target_features,
    target_labels,
    source_protos,
    class_ids,
    num_classes,
    bootstrap_iters=200,
    seed=23,
    eps=1e-12,
):
    if source_protos is None or len(class_ids) < 2:
        return {}
    class_to_row = {class_id: idx for idx, class_id in enumerate(class_ids)}
    valid_mask = torch.tensor([int(label.item()) in class_to_row for label in target_labels], dtype=torch.bool)
    if int(valid_mask.sum().item()) == 0:
        return {}

    feats = target_features[valid_mask]
    labels = target_labels[valid_mask]
    dists = pairwise_sq_dists(feats, source_protos)
    top2 = torch.topk(dists, k=2, dim=1, largest=False).values
    nearest_rows = dists.argmin(dim=1)
    nearest_labels = torch.tensor([class_ids[int(idx.item())] for idx in nearest_rows], dtype=torch.long)
    nearest_correct = (nearest_labels == labels).float()
    nearest_acc = float(nearest_correct.mean().item())
    nearest_f1 = macro_f1(nearest_labels, labels, num_classes)
    nearest_margin = top2[:, 1] - top2[:, 0]

    true_rows = torch.tensor([class_to_row[int(label.item())] for label in labels], dtype=torch.long)
    true_dists = dists[torch.arange(dists.shape[0]), true_rows]
    other_mask = torch.ones_like(dists, dtype=torch.bool)
    other_mask[torch.arange(dists.shape[0]), true_rows] = False
    other_nearest = dists.masked_fill(~other_mask, float("inf")).min(dim=1).values
    oracle_margin = other_nearest - true_dists
    true_rank = (dists < true_dists.unsqueeze(1)).sum(dim=1).float() + 1.0

    proto_dists = pairwise_sq_dists(source_protos, source_protos)
    offdiag = ~torch.eye(source_protos.shape[0], dtype=torch.bool)
    source_inter_scale = proto_dists[offdiag].mean().clamp_min(eps)

    norm_target = F.normalize(feats, dim=1, eps=eps)
    norm_proto = F.normalize(source_protos, dim=1, eps=eps)
    cosine = norm_target @ norm_proto.t()
    cos_nearest_rows = cosine.argmax(dim=1)
    cos_nearest_labels = torch.tensor([class_ids[int(idx.item())] for idx in cos_nearest_rows], dtype=torch.long)
    cos_correct = (cos_nearest_labels == labels).float()
    cos_top2 = torch.topk(cosine, k=2, dim=1, largest=True).values
    cos_nearest_margin = cos_top2[:, 0] - cos_top2[:, 1]
    true_cos = cosine[torch.arange(cosine.shape[0]), true_rows]
    other_cos = cosine.masked_fill(~other_mask, float("-inf")).max(dim=1).values
    cos_oracle_margin = true_cos - other_cos

    nearest_acc_lo, nearest_acc_hi = bootstrap_mean_ci(nearest_correct, bootstrap_iters, seed)
    oracle_margin_lo, oracle_margin_hi = bootstrap_mean_ci(oracle_margin, bootstrap_iters, seed + 1)
    cos_acc_lo, cos_acc_hi = bootstrap_mean_ci(cos_correct, bootstrap_iters, seed + 2)
    cos_oracle_lo, cos_oracle_hi = bootstrap_mean_ci(cos_oracle_margin, bootstrap_iters, seed + 3)

    return {
        "target_proto_nearest_acc": nearest_acc,
        "target_proto_nearest_acc_ci_low": nearest_acc_lo,
        "target_proto_nearest_acc_ci_high": nearest_acc_hi,
        "target_proto_nearest_macro_f1": nearest_f1,
        "target_proto_nearest_margin": float(nearest_margin.mean().item()),
        "target_proto_nearest_margin_norm": float((nearest_margin / source_inter_scale).mean().item()),
        "target_oracle_source_dist": float(true_dists.mean().item()),
        "target_oracle_source_dist_norm": float((true_dists / source_inter_scale).mean().item()),
        "target_oracle_margin": float(oracle_margin.mean().item()),
        "target_oracle_margin_ci_low": oracle_margin_lo,
        "target_oracle_margin_ci_high": oracle_margin_hi,
        "target_oracle_margin_norm": float((oracle_margin / source_inter_scale).mean().item()),
        "target_true_proto_rank": float(true_rank.mean().item()),
        "target_cos_proto_nearest_acc": float(cos_correct.mean().item()),
        "target_cos_proto_nearest_acc_ci_low": cos_acc_lo,
        "target_cos_proto_nearest_acc_ci_high": cos_acc_hi,
        "target_cos_proto_nearest_macro_f1": macro_f1(cos_nearest_labels, labels, num_classes),
        "target_cos_proto_nearest_margin": float(cos_nearest_margin.mean().item()),
        "target_cos_oracle_similarity": float(true_cos.mean().item()),
        "target_cos_oracle_margin": float(cos_oracle_margin.mean().item()),
        "target_cos_oracle_margin_ci_low": cos_oracle_lo,
        "target_cos_oracle_margin_ci_high": cos_oracle_hi,
        "target_valid_count": int(valid_mask.sum().item()),
    }


def class_centroid_alignment(source_protos, source_class_ids, target_features, target_labels, num_classes, eps=1e-12):
    if source_protos is None:
        return {}
    source_by_class = {class_id: source_protos[idx] for idx, class_id in enumerate(source_class_ids)}
    shifts = []
    cosines = []
    target_intra_total = 0.0
    target_intra_count = 0
    class_count = 0
    for class_id in range(num_classes):
        if class_id not in source_by_class:
            continue
        group = target_features[target_labels == class_id]
        if group.numel() == 0:
            continue
        target_center = group.mean(dim=0)
        source_center = source_by_class[class_id]
        shifts.append(((target_center - source_center) ** 2).sum())
        cosines.append(F.cosine_similarity(target_center, source_center, dim=0, eps=eps))
        sq = ((group - target_center.unsqueeze(0)) ** 2).sum(dim=1)
        target_intra_total += float(sq.sum().item())
        target_intra_count += int(sq.numel())
        class_count += 1
    if not shifts:
        return {}
    shift = torch.stack(shifts).mean()
    cosine_values = torch.stack(cosines)
    proto_dists = pairwise_sq_dists(source_protos, source_protos)
    offdiag = ~torch.eye(source_protos.shape[0], dtype=torch.bool)
    source_inter_scale = proto_dists[offdiag].mean().clamp_min(eps) if int(offdiag.sum().item()) else shift.new_tensor(1.0)
    return {
        "source_target_centroid_shift": float(shift.item()),
        "source_target_centroid_shift_norm": float((shift / source_inter_scale).item()),
        "source_target_centroid_cosine_mean": float(cosine_values.mean().item()),
        "source_target_centroid_cosine_min": float(cosine_values.min().item()),
        "target_intra_sq": target_intra_total / max(target_intra_count, 1),
        "source_target_class_overlap_count": class_count,
    }


def target_head_metrics(logits, labels, num_classes):
    probs = F.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "target_head_acc": float((pred == labels).float().mean().item()),
        "target_head_macro_f1": macro_f1(pred, labels, num_classes),
        "target_head_confidence": float(confidence.mean().item()),
        "target_head_entropy_norm": float((entropy / math.log(max(num_classes, 2))).mean().item()),
    }


def run_row(row, args, device):
    checkpoint = checkpoint_from_log(args.log_dir, row, args.outputs_root)
    out = {
        "task": row["task"],
        "seed": row["seed"],
        "config": row["config"],
        "compact_weight": safe_float(row.get("compact_weight")),
        "compact_distance": row.get("compact_distance", ""),
        "norm_preserve_trade_off": safe_float(row.get("norm_preserve_trade_off")),
        "norm_preserve_target": row.get("norm_preserve_target", ""),
        "norm_preserve_value": safe_float(row.get("norm_preserve_value")),
        "feature_kind": args.feature_kind,
        "checkpoint": "" if checkpoint is None else str(checkpoint),
        "status": "ok",
    }
    if checkpoint is None:
        out["status"] = "missing_checkpoint"
        return out

    classes = get_task_classes(args.data_root, row["source_dataset"], closed_set=True)
    model = build_model(args.model, len(classes), args.with_extra, device)
    load_checkpoint(model, checkpoint, device)
    source_ds = build_dataset(args.data_root, row["source_dataset"], classes, True, args.with_extra)
    target_ds = build_dataset(args.data_root, row["target_dataset"], classes, True, args.with_extra)
    source_loader = build_loader(source_ds, args.batch_size, args.num_workers)
    target_loader = build_loader(target_ds, args.batch_size, args.num_workers)

    source = collect_features(model, source_loader, device, args.max_batches, args.feature_kind)
    target = collect_features(model, target_loader, device, args.max_batches, args.feature_kind)
    if source is None or target is None:
        out["status"] = "empty_loader"
        return out
    source = maybe_subsample(source, args.max_metric_samples, args.seed + int(row["seed"]) * 17)
    target = maybe_subsample(target, args.max_metric_samples, args.seed + int(row["seed"]) * 31)

    out.update(
        {
            "source_count": int(source["labels"].numel()),
            "target_count": int(target["labels"].numel()),
            "feature_dim": int(source["features"].shape[1]),
        }
    )
    source_metrics, source_protos, source_class_ids, _ = source_geometry(
        source["features"], source["labels"], len(classes)
    )
    out.update(source_metrics)
    out.update(target_head_metrics(target["logits"], target["labels"], len(classes)))
    out.update(
        target_source_compatibility(
            target["features"],
            target["labels"],
            source_protos,
            source_class_ids,
            len(classes),
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed + int(row["seed"]) * 101,
        )
    )
    out.update(
        class_centroid_alignment(
            source_protos, source_class_ids, target["features"], target["labels"], len(classes)
        )
    )
    return out


def summarize(rows, group_fields, value_fields):
    grouped = {}
    for row in rows:
        if "status" in row and row.get("status") != "ok":
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


def build_delta_vs_plain(rows, summary_rows):
    summary_map = {
        (row.get("task"), str(row.get("seed")), row.get("config")): row
        for row in summary_rows
        if row.get("status") in {"ok", ""}
    }
    metric_fields = content_metric_fields()
    by_key = {
        (row["task"], str(row["seed"]), row["config"]): row
        for row in rows
        if row.get("status") == "ok"
    }
    out = []
    missing = []
    for key, row in sorted(by_key.items()):
        task, seed, config = key
        if config == "plain":
            continue
        plain = by_key.get((task, seed, "plain"))
        if plain is None:
            missing.append(
                {
                    "task": task,
                    "seed": seed,
                    "config": config,
                    "reason": "missing_plain_content_row",
                }
            )
            continue
        item = {
            "task": task,
            "seed": seed,
            "config": config,
            "compact_weight": row.get("compact_weight"),
            "feature_kind": row.get("feature_kind"),
        }
        config_summary = summary_map.get((task, seed, config), {})
        plain_summary = summary_map.get((task, seed, "plain"), {})
        for field in ["source_on_target_f1", "da_f1", "da_gain", "initial_all_f1", "last_all_f1"]:
            left = safe_float(config_summary.get(field))
            right = safe_float(plain_summary.get(field))
            item[f"plain_{field}"] = right
            item[f"config_{field}"] = left
            item[f"delta_{field}"] = None if left is None or right is None else left - right
        for field in metric_fields:
            left = row.get(field)
            right = plain.get(field)
            item[f"delta_{field}"] = None if left is None or right is None else left - right
        out.append(item)
    return out, missing


def build_attribution(delta_rows):
    by_task = {}
    for row in delta_rows:
        by_task.setdefault((row["task"], row["feature_kind"]), []).append(row)
    metrics = [f"delta_{field}" for field in content_metric_fields()]
    refs = ["delta_da_f1", "delta_source_on_target_f1", "delta_initial_all_f1", "delta_last_all_f1"]
    out = []
    for (task, feature_kind), group in sorted(by_task.items()):
        for metric in metrics:
            values = [row.get(metric) for row in group]
            item = {"task": task, "feature_kind": feature_kind, "metric": metric, "n": len(group)}
            for ref in refs:
                ref_values = [row.get(ref) for row in group]
                item[f"pearson_with_{ref}"] = pearson(values, ref_values)
                item[f"spearman_with_{ref}"] = spearman(values, ref_values)
            out.append(item)
    return out


def content_metric_fields():
    return [
        "source_intra_sq",
        "source_inter_sq_mean",
        "source_inter_sq_nearest",
        "source_inter_intra_ratio",
        "source_nearest_inter_intra_ratio",
        "source_feature_norm",
        "source_cov_trace",
        "source_min_class_count",
        "target_proto_nearest_acc",
        "target_proto_nearest_acc_ci_low",
        "target_proto_nearest_acc_ci_high",
        "target_proto_nearest_macro_f1",
        "target_proto_nearest_margin",
        "target_proto_nearest_margin_norm",
        "target_oracle_source_dist",
        "target_oracle_source_dist_norm",
        "target_oracle_margin",
        "target_oracle_margin_ci_low",
        "target_oracle_margin_ci_high",
        "target_oracle_margin_norm",
        "target_true_proto_rank",
        "target_cos_proto_nearest_acc",
        "target_cos_proto_nearest_acc_ci_low",
        "target_cos_proto_nearest_acc_ci_high",
        "target_cos_proto_nearest_macro_f1",
        "target_cos_proto_nearest_margin",
        "target_cos_oracle_similarity",
        "target_cos_oracle_margin",
        "target_cos_oracle_margin_ci_low",
        "target_cos_oracle_margin_ci_high",
        "source_target_centroid_shift",
        "source_target_centroid_shift_norm",
        "source_target_centroid_cosine_mean",
        "source_target_centroid_cosine_min",
        "target_intra_sq",
    ]


def debug_metric_fields():
    return [
        "target_head_acc",
        "target_head_macro_f1",
        "target_head_confidence",
        "target_head_entropy_norm",
    ]


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
            f"[{idx}/{len(jobs)}] task={row['task']} seed={row['seed']} "
            f"config={row['config']} feature={args.feature_kind}",
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
                    "feature_kind": args.feature_kind,
                    "status": f"error:{type(exc).__name__}",
                    "error": str(exc),
                }
            )

    summary_rows = read_optional_tsv(args.log_dir / "raw_strength_rows.tsv")
    delta_rows, missing_delta_rows = build_delta_vs_plain(rows, summary_rows)
    curve = summarize(
        rows,
        ["task", "config", "compact_weight", "compact_distance", "norm_preserve_target", "feature_kind"],
        content_metric_fields(),
    )
    delta_curve = summarize(
        delta_rows,
        ["task", "config", "compact_weight", "feature_kind"],
        [
            "delta_da_f1",
            "delta_source_on_target_f1",
            "delta_initial_all_f1",
            "delta_last_all_f1",
            *[f"delta_{field}" for field in content_metric_fields()],
        ],
    )
    attribution = build_attribution(delta_rows)

    row_fields = [
        "task",
        "seed",
        "config",
        "compact_weight",
        "compact_distance",
        "norm_preserve_trade_off",
        "norm_preserve_target",
        "norm_preserve_value",
        "feature_kind",
        "status",
        "source_count",
        "target_count",
        "feature_dim",
        *debug_metric_fields(),
        *content_metric_fields(),
        "checkpoint",
        "error",
    ]
    delta_fields = [
        "task",
        "seed",
        "config",
        "compact_weight",
        "feature_kind",
        "plain_source_on_target_f1",
        "config_source_on_target_f1",
        "delta_source_on_target_f1",
        "plain_da_f1",
        "config_da_f1",
        "delta_da_f1",
        "plain_initial_all_f1",
        "config_initial_all_f1",
        "delta_initial_all_f1",
        "plain_last_all_f1",
        "config_last_all_f1",
        "delta_last_all_f1",
        *[f"delta_{field}" for field in content_metric_fields()],
    ]

    prefix = args.log_dir / args.output_prefix
    write_tsv(prefix.with_name(f"{prefix.name}_rows.tsv"), rows, row_fields)
    write_tsv(prefix.with_name(f"{prefix.name}_curve.tsv"), curve, list(curve[0].keys()) if curve else ["task"])
    write_tsv(prefix.with_name(f"{prefix.name}_delta_vs_plain.tsv"), delta_rows, delta_fields)
    write_tsv(
        prefix.with_name(f"{prefix.name}_missing_delta_pairs.tsv"),
        missing_delta_rows,
        ["task", "seed", "config", "reason"],
    )
    write_tsv(
        prefix.with_name(f"{prefix.name}_delta_curve.tsv"),
        delta_curve,
        list(delta_curve[0].keys()) if delta_curve else ["task"],
    )
    write_tsv(
        prefix.with_name(f"{prefix.name}_attribution.tsv"),
        attribution,
        list(attribution[0].keys()) if attribution else ["task"],
    )
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_rows.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_curve.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_delta_vs_plain.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_missing_delta_pairs.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_delta_curve.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_attribution.tsv')}")


if __name__ == "__main__":
    main()
