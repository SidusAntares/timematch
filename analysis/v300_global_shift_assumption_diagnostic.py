#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
from torch.utils import data
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import PixelSetData, count_pixelset_samples
from models.stclassifier import PseLTae
from timematch import collect_shift_softmaxes, estimate_class_distribution, score_shift_softmaxes
from transforms import Normalize, RandomSamplePixels, ToTensor
from utils import label_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="v3.0.0 offline diagnostic for TimeMatch global scalar temporal shift."
    )
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--max_temporal_shift", type=int, default=60)
    parser.add_argument("--pseudo_threshold", type=float, default=0.9)
    return parser.parse_args()


def bool_value(text):
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return ""
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


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
            assert len(train_indices) + len(val_indices) + len(test_indices) == n
            splits[dataset_name] = {"train": train_indices, "val": val_indices, "test": test_indices}
        folds.append(splits)
    return folds


def resolve_classes(args):
    source_classes = label_utils.get_classes(
        args.source.split("/")[0],
        combine_spring_and_winter=False,
    )
    if bool_value(args.closed_set):
        source_classes = [cls for cls in source_classes if cls != "unknown"]
    source_data = PixelSetData(args.data_root, args.source, source_classes, closed_set=bool_value(args.closed_set))
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    return [source_classes[i] for i in labels[counts >= 200]]


def build_target_loader(args, classes, split_indices):
    transform = transforms.Compose([
        RandomSamplePixels(args.num_pixels),
        Normalize(),
        ToTensor(),
    ])
    dataset = PixelSetData(
        args.data_root,
        args.target,
        classes,
        transform,
        indices=split_indices,
        closed_set=bool_value(args.closed_set),
    )
    return data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )


def load_model(args, classes):
    model = PseLTae(
        input_dim=args.input_dim,
        num_classes=len(classes),
        with_extra=bool_value(args.with_extra),
    )
    checkpoint = Path("outputs") / args.source_model / "fold_0" / "model.pt"
    if not checkpoint.exists():
        checkpoint = Path(args.source_model)
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)["state_dict"]
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()
    return model, checkpoint


def percentile_iqr(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = resolve_classes(args)
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
    target_loader = build_target_loader(args, classes, splits[args.target]["train"])
    model, checkpoint = load_model(args, classes)

    min_shift, max_shift = -args.max_temporal_shift, args.max_temporal_shift
    shifts, shift_softmaxes, labels = collect_shift_softmaxes(
        model,
        target_loader,
        args.device,
        min_shift=min_shift,
        max_shift=max_shift,
        sample_size=args.sample_size,
    )
    actual_distribution = estimate_class_distribution(labels, len(classes))
    scores = score_shift_softmaxes(
        shifts,
        shift_softmaxes,
        labels,
        len(classes),
        class_distribution=actual_distribution,
        pseudo_threshold=args.pseudo_threshold,
    )

    curve_rows = []
    for row in scores["rows"]:
        curve_rows.append(
            {
                "task": args.task,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "epoch": "source_only",
                **row,
            }
        )
    curve_fields = [
        "task",
        "source",
        "target",
        "seed",
        "epoch",
        "shift_delta",
        "is_score",
        "am_score",
        "entropy_mean",
        "prediction_confidence_mean",
        "prediction_confidence_std",
        "marginal_entropy",
        "marginal_kl_to_uniform",
        "pseudo_label_coverage_tau",
        "pseudo_label_class_entropy",
        "pseudo_label_effective_class_count",
        "pseudo_label_class_distribution_json",
        "target_macro_f1_at_shift",
        "target_accuracy_at_shift",
        "target_mean_confidence_at_shift",
    ]
    write_tsv(output_dir / "shift_score_curves_source_only.tsv", curve_rows, curve_fields)

    estimated_is_shift = scores["best_is_shift"]
    estimated_am_shift = scores["best_am_shift"]
    oracle_shift = scores["best_f1_shift"]
    shift_to_idx = {shift: idx for idx, shift in enumerate(shifts)}
    estimated_idx = shift_to_idx[estimated_am_shift]
    oracle_idx = shift_to_idx[oracle_shift]
    no_shift_idx = shift_to_idx.get(0, estimated_idx)
    oracle_row = {
        "task": args.task,
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        "estimated_shift_is": estimated_is_shift,
        "estimated_shift_am": estimated_am_shift,
        "oracle_scalar_shift_by_f1": oracle_shift,
        "estimated_shift_f1": float(scores["f1_scores"][estimated_idx]),
        "oracle_scalar_f1": float(scores["f1_scores"][oracle_idx]),
        "no_shift_f1": float(scores["f1_scores"][no_shift_idx]),
        "abs_shift_error": abs(int(estimated_am_shift) - int(oracle_shift)),
        "shift_estimation_gap": float(scores["f1_scores"][estimated_idx] - scores["f1_scores"][oracle_idx]),
        "is_top1_top2_margin": scores["is_top1_top2_margin"],
        "am_top1_top2_margin": scores["am_top1_top2_margin"],
        "is_curve_sharpness": scores["is_curve_sharpness"],
        "am_curve_sharpness": scores["am_curve_sharpness"],
        "checkpoint": str(checkpoint),
    }
    write_tsv(
        output_dir / "oracle_scalar_shift_summary.tsv",
        [oracle_row],
        [
            "task",
            "source",
            "target",
            "seed",
            "estimated_shift_is",
            "estimated_shift_am",
            "oracle_scalar_shift_by_f1",
            "estimated_shift_f1",
            "oracle_scalar_f1",
            "no_shift_f1",
            "abs_shift_error",
            "shift_estimation_gap",
            "is_top1_top2_margin",
            "am_top1_top2_margin",
            "is_curve_sharpness",
            "am_curve_sharpness",
            "checkpoint",
        ],
    )

    preds = np.argmax(shift_softmaxes, axis=2)
    class_rows = []
    class_shifts, class_weights = [], []
    for class_id, class_name in enumerate(classes):
        true_mask = labels == class_id
        n_target = int(true_mask.sum())
        if n_target <= 0:
            continue
        class_f1 = []
        class_recall = []
        class_acc = []
        for shift_idx in range(len(shifts)):
            pred_mask = preds[:, shift_idx] == class_id
            class_f1.append(
                sklearn.metrics.f1_score(true_mask, pred_mask, average="binary", zero_division=0)
            )
            class_recall.append(
                sklearn.metrics.recall_score(true_mask, pred_mask, average="binary", zero_division=0)
            )
            class_acc.append(float((preds[true_mask, shift_idx] == class_id).mean()))
        best_idx = int(np.argsort(class_f1)[::-1][0])
        best_shift = int(shifts[best_idx])
        class_shifts.append(best_shift)
        class_weights.append(n_target)
        class_rows.append(
            {
                "task": args.task,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "class_id": class_id,
                "class_name_if_available": class_name,
                "n_target_samples": n_target,
                "oracle_class_shift": best_shift,
                "oracle_class_f1": float(class_f1[best_idx]),
                "oracle_class_recall": float(class_recall[best_idx]),
                "oracle_class_accuracy": float(class_acc[best_idx]),
                "global_oracle_shift": oracle_shift,
                "class_shift_minus_global": best_shift - int(oracle_shift),
                "estimated_am_shift": estimated_am_shift,
                "class_shift_minus_estimated": best_shift - int(estimated_am_shift),
            }
        )
    class_fields = [
        "task",
        "source",
        "target",
        "seed",
        "class_id",
        "class_name_if_available",
        "n_target_samples",
        "oracle_class_shift",
        "oracle_class_f1",
        "oracle_class_recall",
        "oracle_class_accuracy",
        "global_oracle_shift",
        "class_shift_minus_global",
        "estimated_am_shift",
        "class_shift_minus_estimated",
    ]
    write_tsv(output_dir / "classwise_oracle_shift.tsv", class_rows, class_fields)

    class_shifts_np = np.asarray(class_shifts, dtype=np.float64)
    class_weights_np = np.asarray(class_weights, dtype=np.float64)
    if class_shifts_np.size:
        weighted_mean = float(np.average(class_shifts_np, weights=class_weights_np))
        weighted_var = float(np.average((class_shifts_np - weighted_mean) ** 2, weights=class_weights_np))
        dispersion_row = {
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "n_classes_valid": int(class_shifts_np.size),
            "global_oracle_shift": oracle_shift,
            "estimated_am_shift": estimated_am_shift,
            "class_shift_mean": float(class_shifts_np.mean()),
            "class_shift_std": float(class_shifts_np.std()),
            "class_shift_range": float(class_shifts_np.max() - class_shifts_np.min()),
            "class_shift_iqr": percentile_iqr(class_shifts_np),
            "weighted_class_shift_std": math.sqrt(weighted_var),
            "mean_abs_class_shift_minus_global": float(np.mean(np.abs(class_shifts_np - int(oracle_shift)))),
            "mean_abs_class_shift_minus_estimated": float(np.mean(np.abs(class_shifts_np - int(estimated_am_shift)))),
        }
    else:
        dispersion_row = {
            "task": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "n_classes_valid": 0,
        }
    write_tsv(
        output_dir / "classwise_shift_dispersion_summary.tsv",
        [dispersion_row],
        [
            "task",
            "source",
            "target",
            "seed",
            "n_classes_valid",
            "global_oracle_shift",
            "estimated_am_shift",
            "class_shift_mean",
            "class_shift_std",
            "class_shift_range",
            "class_shift_iqr",
            "weighted_class_shift_std",
            "mean_abs_class_shift_minus_global",
            "mean_abs_class_shift_minus_estimated",
        ],
    )
    print("Wrote v300 offline diagnostics to", output_dir)


if __name__ == "__main__":
    main()
