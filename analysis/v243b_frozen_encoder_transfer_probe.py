#!/usr/bin/env python3
import argparse
import csv
import math
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
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


JOB_FIELDS = [
    "task",
    "source_dataset",
    "target_dataset",
    "seed",
    "config",
    "compact_weight",
    "est_weight",
]

SOURCE_OUTPUT_RE = re.compile(r"output_dir='([^']*sourcephasecompact[^']*)'")
SAVE_RE = re.compile(r"Saving best model to (outputs/[^\\s]+/fold_0/model\\.pt)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-encoder transfer probe for v2.4.3b raw compactness experiments. "
            "It trains a fresh linear classifier on frozen source features and evaluates "
            "target transfer without TimeMatch iterations."
        )
    )
    parser.add_argument("log_dir", help="Experiment log directory containing jobs.tsv and job logs.")
    parser.add_argument("--jobs_tsv", default="", help="Defaults to <log_dir>/jobs.tsv.")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--output_prefix", default="frozen_probe")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--configs", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="pseltae", choices=["pseltae", "psetae", "psetcnn", "psegru"])
    parser.add_argument("--with_extra", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--probe_lr", type=float, default=0.05)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--class_weight", default="balanced", choices=["none", "balanced"])
    parser.add_argument("--feature_kind", default="final", choices=["final", "raw_pooled", "raw_flat"])
    parser.add_argument("--standardize", default=True, type=bool_arg)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def parse_set(text):
    if text is None or str(text).strip() == "":
        return None
    return {part.strip() for part in str(text).split(",") if part.strip()}


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


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


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def stdev(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def rank_values(values):
    indexed = [(idx, value) for idx, value in enumerate(values) if value is not None]
    ranks = [None] * len(values)
    if len(indexed) < 2:
        return ranks
    indexed.sort(key=lambda item: item[1])
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx, _ in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    var_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def spearman(xs, ys):
    return pearson(rank_values(xs), rank_values(ys))


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


def job_log_path(log_dir, row):
    pattern = f"gpu*_{row['task']}_seed{row['seed']}_{row['config']}.log"
    matches = sorted(Path(log_dir).glob(pattern))
    return matches[0] if matches else None


def checkpoint_from_log(log_dir, row, outputs_root):
    log_path = job_log_path(log_dir, row)
    if log_path is None or not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    match = SOURCE_OUTPUT_RE.search(text)
    if match:
        checkpoint = Path(match.group(1)) / "fold_0" / "model.pt"
        if checkpoint.exists():
            return checkpoint
        rooted = Path(outputs_root) / checkpoint.relative_to("outputs") if checkpoint.parts[0] == "outputs" else checkpoint
        if rooted.exists():
            return rooted

    match = SAVE_RE.search(text)
    if match:
        checkpoint = Path(match.group(1))
        if checkpoint.exists():
            return checkpoint
        rooted = Path(outputs_root) / checkpoint.relative_to("outputs") if checkpoint.parts[0] == "outputs" else checkpoint
        if rooted.exists():
            return rooted
    return None


@torch.no_grad()
def collect_features(model, loader, device, max_batches, feature_kind):
    xs = []
    ys = []
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
        xs.append(feats.detach().cpu().float())
        ys.append(y.detach().cpu().long())
        logits_list.append(logits.detach().cpu().float())

    if not xs:
        return None
    return {
        "x": torch.cat(xs, dim=0),
        "y": torch.cat(ys, dim=0),
        "logits": torch.cat(logits_list, dim=0),
    }


def macro_f1(pred, labels, num_classes):
    scores = []
    for class_id in range(num_classes):
        tp = ((pred == class_id) & (labels == class_id)).sum().item()
        fp = ((pred == class_id) & (labels != class_id)).sum().item()
        fn = ((pred != class_id) & (labels == class_id)).sum().item()
        denom = 2 * tp + fp + fn
        if denom == 0:
            scores.append(0.0)
        else:
            scores.append((2 * tp) / denom)
    return sum(scores) / max(len(scores), 1)


def eval_logits(logits, labels, num_classes):
    pred = logits.argmax(dim=1)
    return {
        "acc": float((pred == labels).float().mean().item()),
        "macro_f1": macro_f1(pred, labels, num_classes),
    }


def standardize(train_x, *others):
    mean_x = train_x.mean(dim=0, keepdim=True)
    std_x = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return [(x - mean_x) / std_x for x in (train_x, *others)]


def class_weights(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    weights = labels.numel() / (num_classes * counts)
    return weights / weights.mean().clamp_min(1e-6)


def train_linear_probe(source_x, source_y, target_x, num_classes, args, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if args.standardize:
        source_x, target_x = standardize(source_x, target_x)

    source_x = source_x.to(device)
    source_y = source_y.to(device)
    target_x = target_x.to(device)
    probe = nn.Linear(source_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay
    )
    weight = None
    if args.class_weight == "balanced":
        weight = class_weights(source_y.detach().cpu(), num_classes).to(device)

    for _ in range(args.probe_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(source_x)
        loss = F.cross_entropy(logits, source_y, weight=weight)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        return probe(source_x).detach().cpu(), probe(target_x).detach().cpu()


def run_row(row, args, device):
    checkpoint_path = checkpoint_from_log(args.log_dir, row, args.outputs_root)
    out = {
        "task": row["task"],
        "seed": row["seed"],
        "config": row["config"],
        "compact_weight": safe_float(row.get("compact_weight")),
        "checkpoint": "" if checkpoint_path is None else str(checkpoint_path),
        "status": "ok",
    }
    if checkpoint_path is None:
        out["status"] = "missing_checkpoint"
        return out

    classes = get_task_classes(args.data_root, row["source_dataset"], closed_set=True)
    model = build_model(args.model, len(classes), args.with_extra, device)
    load_checkpoint(model, checkpoint_path, device)
    source_ds = build_dataset(args.data_root, row["source_dataset"], classes, True, args.with_extra)
    target_ds = build_dataset(args.data_root, row["target_dataset"], classes, True, args.with_extra)
    source_loader = build_loader(source_ds, args.batch_size, args.num_workers)
    target_loader = build_loader(target_ds, args.batch_size, args.num_workers)

    source = collect_features(model, source_loader, device, args.max_batches, args.feature_kind)
    target = collect_features(model, target_loader, device, args.max_batches, args.feature_kind)
    if source is None or target is None:
        out["status"] = "empty_loader"
        return out

    num_classes = len(classes)
    head_source = eval_logits(source["logits"], source["y"], num_classes)
    head_target = eval_logits(target["logits"], target["y"], num_classes)
    probe_source_logits, probe_target_logits = train_linear_probe(
        source["x"], source["y"], target["x"], num_classes, args, args.seed + int(row["seed"])
    )
    probe_source = eval_logits(probe_source_logits, source["y"], num_classes)
    probe_target = eval_logits(probe_target_logits, target["y"], num_classes)

    out.update(
        {
            "source_count": int(source["y"].numel()),
            "target_count": int(target["y"].numel()),
            "feature_dim": int(source["x"].shape[1]),
            "head_source_acc": head_source["acc"],
            "head_source_macro_f1": head_source["macro_f1"],
            "head_target_acc": head_target["acc"],
            "head_target_macro_f1": head_target["macro_f1"],
            "probe_source_acc": probe_source["acc"],
            "probe_source_macro_f1": probe_source["macro_f1"],
            "probe_target_acc": probe_target["acc"],
            "probe_target_macro_f1": probe_target["macro_f1"],
        }
    )
    return out


def build_deltas(rows):
    base = {
        (row["task"], row["seed"]): row
        for row in rows
        if row.get("status") == "ok" and row.get("config") == "plain"
    }
    fields = [
        "head_target_macro_f1",
        "probe_target_macro_f1",
        "head_source_macro_f1",
        "probe_source_macro_f1",
    ]
    out = []
    for row in rows:
        if row.get("status") != "ok" or row.get("config") == "plain":
            continue
        plain = base.get((row["task"], row["seed"]))
        if plain is None:
            continue
        item = {
            "task": row["task"],
            "seed": row["seed"],
            "config": row["config"],
            "compact_weight": row["compact_weight"],
        }
        for field in fields:
            left = row.get(field)
            right = plain.get(field)
            item[f"plain_{field}"] = right
            item[f"config_{field}"] = left
            item[f"delta_{field}"] = None if left is None or right is None else left - right
        out.append(item)
    return out


def summarize(rows, group_fields, value_fields):
    grouped = {}
    for row in rows:
        if row.get("status") not in {None, "ok"}:
            continue
        key = tuple(row.get(field) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item["n"] = len(group)
        for field in value_fields:
            values = [row.get(field) for row in group]
            item[f"{field}_mean"] = mean(values)
            item[f"{field}_std"] = stdev(values)
            item[f"{field}_pos"] = sum(1 for value in values if value is not None and value > 0)
        out.append(item)
    return out


def read_optional_curve(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_curve_attribution(log_dir, probe_curve):
    dose_curve = read_optional_curve(Path(log_dir) / "dose_response_curve.tsv")
    if not dose_curve:
        return []
    def curve_key(row):
        weight = safe_float(row.get("compact_weight"))
        if weight is None:
            return row.get("task"), None
        return row.get("task"), round(weight, 8)

    dose_by_key = {
        curve_key(row): row
        for row in dose_curve
    }
    merged = []
    for row in probe_curve:
        key = curve_key(row)
        dose = dose_by_key.get(key)
        if dose is None:
            continue
        merged.append(
            {
                "task": row.get("task"),
                "compact_weight": row.get("compact_weight"),
                "probe_target_macro_f1_mean": row.get("probe_target_macro_f1_mean"),
                "head_target_macro_f1_mean": row.get("head_target_macro_f1_mean"),
                "da_f1_mean": safe_float(dose.get("da_f1_mean")),
                "source_on_target_f1_mean": safe_float(dose.get("source_on_target_f1_mean")),
                "initial_all_f1_mean": safe_float(dose.get("initial_all_f1_mean")),
                "last_all_f1_mean": safe_float(dose.get("last_all_f1_mean")),
            }
        )

    out = []
    by_task = {}
    for row in merged:
        by_task.setdefault(row["task"], []).append(row)
    for task, group in sorted(by_task.items()):
        weights = [row["compact_weight"] for row in group]
        for metric in [
            "probe_target_macro_f1_mean",
            "head_target_macro_f1_mean",
            "da_f1_mean",
            "source_on_target_f1_mean",
            "initial_all_f1_mean",
            "last_all_f1_mean",
        ]:
            out.append(
                {
                    "task": task,
                    "metric": metric,
                    "n": len(group),
                    "pearson_with_weight": pearson(weights, [row.get(metric) for row in group]),
                    "spearman_with_weight": spearman(weights, [row.get(metric) for row in group]),
                    "pearson_with_da_f1": pearson(
                        [row.get("da_f1_mean") for row in group],
                        [row.get(metric) for row in group],
                    ),
                    "spearman_with_da_f1": spearman(
                        [row.get("da_f1_mean") for row in group],
                        [row.get(metric) for row in group],
                    ),
                }
            )
    return out


def main():
    args = parse_args()
    args.log_dir = Path(args.log_dir)
    jobs_tsv = Path(args.jobs_tsv) if args.jobs_tsv else args.log_dir / "jobs.tsv"
    jobs = filter_jobs(read_jobs(jobs_tsv), args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows = []
    for idx, row in enumerate(jobs, start=1):
        print(
            f"[{idx}/{len(jobs)}] task={row['task']} seed={row['seed']} config={row['config']}",
            flush=True,
        )
        try:
            rows.append(run_row(row, args, device))
        except Exception as exc:  # Keep the diagnostic robust across partial log/checkpoint sets.
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

    row_fields = [
        "task",
        "seed",
        "config",
        "compact_weight",
        "status",
        "source_count",
        "target_count",
        "feature_dim",
        "head_source_macro_f1",
        "head_target_macro_f1",
        "probe_source_macro_f1",
        "probe_target_macro_f1",
        "head_source_acc",
        "head_target_acc",
        "probe_source_acc",
        "probe_target_acc",
        "checkpoint",
        "error",
    ]
    delta_fields = [
        "task",
        "seed",
        "config",
        "compact_weight",
        "plain_head_target_macro_f1",
        "config_head_target_macro_f1",
        "delta_head_target_macro_f1",
        "plain_probe_target_macro_f1",
        "config_probe_target_macro_f1",
        "delta_probe_target_macro_f1",
        "plain_head_source_macro_f1",
        "config_head_source_macro_f1",
        "delta_head_source_macro_f1",
        "plain_probe_source_macro_f1",
        "config_probe_source_macro_f1",
        "delta_probe_source_macro_f1",
    ]

    deltas = build_deltas(rows)
    curve = summarize(
        rows,
        ["task", "compact_weight"],
        [
            "head_source_macro_f1",
            "head_target_macro_f1",
            "probe_source_macro_f1",
            "probe_target_macro_f1",
        ],
    )
    delta_curve = summarize(
        deltas,
        ["task", "compact_weight"],
        [
            "delta_head_target_macro_f1",
            "delta_probe_target_macro_f1",
            "delta_head_source_macro_f1",
            "delta_probe_source_macro_f1",
        ],
    )
    attribution = build_curve_attribution(args.log_dir, curve)

    prefix = args.log_dir / args.output_prefix
    write_tsv(prefix.with_name(f"{prefix.name}_rows.tsv"), rows, row_fields)
    write_tsv(prefix.with_name(f"{prefix.name}_deltas.tsv"), deltas, delta_fields)
    write_tsv(prefix.with_name(f"{prefix.name}_curve.tsv"), curve, list(curve[0].keys()) if curve else ["task"])
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
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_delta_curve.tsv')}")
    print(f"Wrote: {prefix.with_name(f'{prefix.name}_attribution.tsv')}")


if __name__ == "__main__":
    main()
