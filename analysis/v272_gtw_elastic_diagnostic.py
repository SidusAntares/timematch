import argparse
import csv
import glob
import json
import math
import os
import random
from collections import defaultdict
from types import SimpleNamespace

import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_adapters.factory import (
    build_dataset,
    get_classes_for_config,
    get_dataset_length,
    make_eval_transform,
)
from dataset import GroupByShapesBatchSampler
from models.stclassifier import PseLTae


REMOTE_TASKS = {
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


MODE_KEYS = {
    "G_global": "global",
    "G_gtw_r1": "gtw_r1",
    "G_gtw_r2": "gtw_r2",
    "G_gtw_r4": "gtw_r4",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline GTW diagnostic: test whether class-level temporal elasticity "
            "explains already observed GTW gains."
        )
    )
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--output_root", required=True, help="GTW controller output root")
    parser.add_argument("--log_root", default="", help="GTW controller log root; optional")
    parser.add_argument(
        "--tasks",
        default="FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1",
        help="Comma-separated remote tasks",
    )
    parser.add_argument("--checkpoint_mode", default="G_global", choices=sorted(MODE_KEYS))
    parser.add_argument("--band_ratios", default="0.10,0.30,0.50")
    parser.add_argument("--max_source_batches", type=int, default=24)
    parser.add_argument("--max_target_batches", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_pixels", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_class_count", type=int, default=8)
    parser.add_argument("--out_csv", required=True)
    return parser.parse_args()


def make_config(args, source, target):
    return SimpleNamespace(
        dataset_type="remote_sensing",
        data_root=args.data_root,
        source=source,
        target=target,
        closed_set=True,
        combine_spring_and_winter=False,
        num_folds=1,
        val_ratio=0.1,
        test_ratio=0.2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_dim=10,
        num_pixels=args.num_pixels,
        seq_length=args.seq_length,
        model="pseltae",
        with_extra=False,
        sample_pixels_val=True,
        seed=args.seed,
    )


def create_train_val_test_folds(datasets, num_indices, val_ratio=0.1, test_ratio=0.2):
    splits = {}
    for dataset_name in datasets:
        indices = list(range(num_indices[dataset_name]))
        n = len(indices)
        n_test = int(test_ratio * n)
        n_val = int(val_ratio * n)
        n_train = n - n_test - n_val
        random.shuffle(indices)
        splits[dataset_name] = {
            "train": set(indices[:n_train]),
            "val": set(indices[n_train : n_train + n_val]),
            "test": set(indices[-n_test:]) if n_test > 0 else set(),
        }
    return splits


def find_source_checkpoint(output_root, task_key, mode):
    mode_key = MODE_KEYS[mode]
    pattern = os.path.join(
        output_root,
        "remote",
        mode_key,
        f"remote_{task_key}_{mode_key}_*_source_*",
        "fold_0",
        "model.pt",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No source checkpoint matched: {pattern}")
    return matches[-1]


def build_model(config, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PseLTae(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        with_extra=config.with_extra,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def move_batch(sample, device):
    return (
        sample["pixels"].to(device, non_blocking=True),
        sample["valid_pixels"].to(device, non_blocking=True),
        sample["positions"].to(device, non_blocking=True),
        sample.get("extra", None).to(device, non_blocking=True) if "extra" in sample else None,
        sample["label"].to(device, non_blocking=True),
    )


@torch.no_grad()
def collect_curves(model, loader, device, max_batches):
    curves, positions, labels = [], [], []
    for batch_idx, sample in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        pixels, valid_pixels, pos, extra, y = move_batch(sample, device)
        feat = model.spatial_encoder(pixels, valid_pixels, extra)
        curves.append(feat.detach().cpu())
        positions.append(pos.detach().cpu().float())
        labels.append(y.detach().cpu().long())
    if not curves:
        raise RuntimeError("No curves collected; check dataset/loader settings.")
    return torch.cat(curves), torch.cat(positions), torch.cat(labels)


def make_loader(config, dataset_name, indices):
    dataset = build_dataset(
        config,
        dataset_name,
        config.classes,
        transform=make_eval_transform(config, sample_pixels=True, sample_time=False),
        indices=indices,
        split="train",
    )
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(dataset, config.batch_size, by_pixel_dim=False),
    )


def class_prototypes(curves, positions, labels, min_class_count):
    out = {}
    for class_id in labels.unique(sorted=True).tolist():
        mask = labels == int(class_id)
        if int(mask.sum().item()) < min_class_count:
            continue
        out[int(class_id)] = {
            "curve": curves[mask].mean(dim=0).float(),
            "grid": positions[mask].median(dim=0).values.float(),
            "count": int(mask.sum().item()),
        }
    return out


def interp_curve(curve, grid, query, eps=1e-8):
    if curve.shape[0] == 1:
        return curve[:1].expand(query.shape[0], curve.shape[-1])
    grid = grid.float()
    query = query.float().clamp(min=float(grid.min().item()), max=float(grid.max().item()))
    order = torch.argsort(grid)
    grid = grid[order]
    curve = curve[order]
    right = torch.searchsorted(grid.contiguous(), query.contiguous(), right=False).clamp(
        min=1,
        max=grid.numel() - 1,
    )
    left = right - 1
    left_t = grid[left]
    right_t = grid[right]
    alpha = ((query - left_t) / (right_t - left_t).clamp_min(eps)).clamp(0.0, 1.0)
    return curve[left] * (1.0 - alpha.unsqueeze(-1)) + curve[right] * alpha.unsqueeze(-1)


def normalize_grid(grid):
    grid = grid.float()
    span = (grid.max() - grid.min()).clamp_min(1e-8)
    return (grid - grid.min()) / span


def rigid_distance(source_curve, source_grid, target_curve, target_grid):
    source_norm = normalize_grid(source_grid)
    target_norm = normalize_grid(target_grid)
    source_on_target = interp_curve(source_curve, source_norm, target_norm)
    return float((source_on_target - target_curve).pow(2).mean().item())


def constrained_dtw_distance(source_curve, source_grid, target_curve, target_grid, band_ratio):
    source_norm = normalize_grid(source_grid).numpy()
    target_norm = normalize_grid(target_grid).numpy()
    source_np = source_curve.float().numpy()
    target_np = target_curve.float().numpy()
    n, m = len(source_np), len(target_np)
    band_ratio = float(band_ratio)

    cost = np.sum((source_np[:, None, :] - target_np[None, :, :]) ** 2, axis=2)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.full((n + 1, m + 1), 0.0, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(source_norm[i - 1] - target_norm[j - 1]) > band_ratio:
                continue
            candidates = (
                (dp[i - 1, j], steps[i - 1, j]),
                (dp[i, j - 1], steps[i, j - 1]),
                (dp[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            prev_cost, prev_steps = min(candidates, key=lambda item: item[0])
            if math.isfinite(prev_cost):
                dp[i, j] = prev_cost + float(cost[i - 1, j - 1])
                steps[i, j] = prev_steps + 1.0

    if not math.isfinite(float(dp[n, m])):
        return float("nan")
    return float(dp[n, m] / max(steps[n, m], 1.0) / max(source_curve.shape[-1], 1))


def assign_target_to_source_classes(target_curves, target_positions, source_protos, min_class_count):
    assigned = defaultdict(lambda: {"curves": [], "positions": []})
    for curve, grid in zip(target_curves, target_positions):
        best_class, best_distance = None, float("inf")
        for class_id, proto in source_protos.items():
            distance = rigid_distance(proto["curve"], proto["grid"], curve, grid)
            if distance < best_distance:
                best_class = class_id
                best_distance = distance
        if best_class is not None:
            assigned[best_class]["curves"].append(curve)
            assigned[best_class]["positions"].append(grid)

    target_protos = {}
    for class_id, items in assigned.items():
        if len(items["curves"]) < min_class_count:
            continue
        target_protos[class_id] = {
            "curve": torch.stack(items["curves"]).mean(dim=0).float(),
            "grid": torch.stack(items["positions"]).median(dim=0).values.float(),
            "count": len(items["curves"]),
        }
    return target_protos


def parse_summary_results(log_root):
    summary = os.path.join(log_root, "summary.tsv")
    if not log_root or not os.path.isfile(summary):
        return {}, {}

    macro = {}
    class_f1 = {}
    with open(summary, "r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("dataset") != "REMOTE" or row.get("stage") != "da" or row.get("status") != "ok":
                continue
            task = row.get("task", "")
            mode = row.get("mode", "")
            f1 = row.get("f1", "")
            if f1:
                macro[(task, mode)] = float(f1)
            log_path = row.get("log", "")
            log_path = localize_logged_path(log_path, log_root)
            parsed = parse_class_report(log_path)
            if parsed:
                class_f1[(task, mode)] = parsed
    return macro, class_f1


def localize_logged_path(log_path, log_root):
    if not log_path or os.path.isfile(log_path):
        return log_path
    run_name = os.path.basename(os.path.abspath(log_root))
    marker = f"/logs/{run_name}/"
    normalized = log_path.replace("\\", "/")
    if marker in normalized:
        candidate = os.path.join(log_root, normalized.split(marker, 1)[1].replace("/", os.sep))
        if os.path.isfile(candidate):
            return candidate
    return log_path


def parse_class_report(log_path):
    if not log_path or not os.path.isfile(log_path):
        return {}
    text = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    out = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("precision", "accuracy", "macro avg", "weighted avg")):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        try:
            precision = float(parts[-4])
            recall = float(parts[-3])
            f1 = float(parts[-2])
            support = int(float(parts[-1]))
        except ValueError:
            continue
        name = " ".join(parts[:-4])
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def rankdata(values):
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
            end += 1
        rank = (idx + end - 1) / 2.0 + 1.0
        for pair_idx in range(idx, end):
            ranks[pairs[pair_idx][0]] = rank
        idx = end
    return ranks


def spearman(xs, ys):
    if len(xs) < 3:
        return ""
    rx = np.asarray(rankdata(xs), dtype=np.float64)
    ry = np.asarray(rankdata(ys), dtype=np.float64)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return ""
    return float(np.corrcoef(rx, ry)[0, 1])


def best_gtw_class_gain(task, class_name, class_f1):
    global_report = class_f1.get((task, "G_global"), {})
    if class_name not in global_report:
        return ""
    base = global_report[class_name]["f1"]
    gains = []
    for mode in ("G_gtw_r1", "G_gtw_r2", "G_gtw_r4"):
        report = class_f1.get((task, mode), {})
        if class_name in report:
            gains.append(report[class_name]["f1"] - base)
    return max(gains) if gains else ""


def best_gtw_macro_gain(task, macro):
    base = macro.get((task, "G_global"))
    if base is None:
        return ""
    values = [macro[(task, mode)] - base for mode in ("G_gtw_r1", "G_gtw_r2", "G_gtw_r4") if (task, mode) in macro]
    return max(values) if values else ""


def run_task(args, task_key, macro, class_f1):
    if task_key not in REMOTE_TASKS:
        raise ValueError(f"Unsupported remote task: {task_key}")
    source, target = REMOTE_TASKS[task_key]
    config = make_config(args, source, target)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config.classes = get_classes_for_config(config)
    config.num_classes = len(config.classes)

    indices = {
        source: get_dataset_length(config, source, split="train"),
        target: get_dataset_length(config, target, split="train"),
    }
    splits = create_train_val_test_folds([source, target], indices, config.val_ratio, config.test_ratio)
    source_loader = make_loader(config, source, splits[source]["train"])
    target_loader = make_loader(config, target, splits[target]["train"])

    checkpoint_path = find_source_checkpoint(args.output_root, task_key, args.checkpoint_mode)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = build_model(config, checkpoint_path, device)

    source_curves, source_positions, source_labels = collect_curves(
        model,
        source_loader,
        device,
        args.max_source_batches,
    )
    target_curves, target_positions, _ = collect_curves(
        model,
        target_loader,
        device,
        args.max_target_batches,
    )

    source_protos = class_prototypes(source_curves, source_positions, source_labels, args.min_class_count)
    target_protos = assign_target_to_source_classes(
        target_curves,
        target_positions,
        source_protos,
        args.min_class_count,
    )

    rows = []
    band_ratios = [float(x) for x in args.band_ratios.split(",") if x.strip()]
    for class_id, source_proto in source_protos.items():
        if class_id not in target_protos:
            continue
        class_name = config.classes[class_id]
        target_proto = target_protos[class_id]
        rigid = rigid_distance(
            source_proto["curve"],
            source_proto["grid"],
            target_proto["curve"],
            target_proto["grid"],
        )
        class_gain = best_gtw_class_gain(task_key, class_name, class_f1)
        for band_ratio in band_ratios:
            elastic = constrained_dtw_distance(
                source_proto["curve"],
                source_proto["grid"],
                target_proto["curve"],
                target_proto["grid"],
                band_ratio=band_ratio,
            )
            gain = "" if rigid <= 1e-12 or math.isnan(elastic) else (rigid - elastic) / rigid
            rows.append(
                {
                    "task": task_key,
                    "class_id": class_id,
                    "class_name": class_name,
                    "band_ratio": band_ratio,
                    "source_count": source_proto["count"],
                    "target_assigned_count": target_proto["count"],
                    "rigid_distance": rigid,
                    "elastic_distance": elastic,
                    "elastic_gain": gain,
                    "best_gtw_class_f1_gain": class_gain,
                    "best_gtw_macro_f1_gain": best_gtw_macro_gain(task_key, macro),
                    "checkpoint": checkpoint_path,
                }
            )

    for band_ratio in band_ratios:
        xs, ys = [], []
        for row in rows:
            if float(row["band_ratio"]) != band_ratio:
                continue
            if row["elastic_gain"] == "" or row["best_gtw_class_f1_gain"] == "":
                continue
            xs.append(float(row["elastic_gain"]))
            ys.append(float(row["best_gtw_class_f1_gain"]))
        rho = spearman(xs, ys)
        rows.append(
            {
                "task": task_key,
                "class_id": "__summary__",
                "class_name": "__spearman__",
                "band_ratio": band_ratio,
                "source_count": "",
                "target_assigned_count": "",
                "rigid_distance": "",
                "elastic_distance": "",
                "elastic_gain": "",
                "best_gtw_class_f1_gain": "",
                "best_gtw_macro_f1_gain": best_gtw_macro_gain(task_key, macro),
                "checkpoint": checkpoint_path,
                "spearman_elastic_vs_gtw_class_gain": rho,
            }
        )
    return rows


def main():
    args = parse_args()
    macro, class_f1 = parse_summary_results(args.log_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]

    fieldnames = [
        "task",
        "class_id",
        "class_name",
        "band_ratio",
        "source_count",
        "target_assigned_count",
        "rigid_distance",
        "elastic_distance",
        "elastic_gain",
        "best_gtw_class_f1_gain",
        "best_gtw_macro_f1_gain",
        "spearman_elastic_vs_gtw_class_gain",
        "checkpoint",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task_key in tasks:
            print(f"[GTW_DIAG] task={task_key}")
            rows = run_task(args, task_key, macro, class_f1)
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()

    meta_path = os.path.splitext(args.out_csv)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(f"[GTW_DIAG] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
