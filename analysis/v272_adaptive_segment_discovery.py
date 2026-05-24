import argparse
import csv
import json
import os
import random
import sys
from argparse import Namespace

import numpy as np
import torch
from torch.utils import data

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import GroupByShapesBatchSampler
from data_adapters.factory import (
    build_dataset,
    get_classes_for_config,
    get_dataset_length,
    is_har,
    make_eval_transform,
)
from ideas.source_feature_reshaper import build_source_feature_reshaper
from ideas.v271_adaptive_support_discovery import (
    build_atomic_partition_spec,
    compute_source_segment_prototypes,
    construct_adaptive_segments,
    construct_pair_adaptive_supports,
    describe_atomic_partition_spec,
    discover_target_pair_segments,
)
from models.stclassifier import PseGru, PseLTae, PseTae, PseTempCNN
from timematch import estimate_temporal_shift
from utils.train_utils import bool_flag


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "v2.7.2b offline support-bank discovery: build shift-aligned, "
            "class-pair-conditioned adaptive temporal supports."
        )
    )
    parser.add_argument("--run_dir", required=True, help="Source training run directory containing train_config.json.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to run_dir/fold_0/model.pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default="", help="Override data_root from train_config.json.")
    parser.add_argument("--source_max_batches", default=64, type=int)
    parser.add_argument("--target_max_batches", default=64, type=int)
    parser.add_argument("--shift_sample_size", default=40, type=int)
    parser.add_argument("--target_to_source_shift", default=None, type=int)
    parser.add_argument("--shift_jitter", default=3, type=int)
    parser.add_argument("--atomic_bins", default=12, type=int)
    parser.add_argument("--top_k_segments", default=8, type=int)
    parser.add_argument("--top_m_per_pair", default=2, type=int)
    parser.add_argument("--max_adaptive_supports", default=12, type=int)
    parser.add_argument("--segment_score_quantile", default=0.75, type=float)
    parser.add_argument("--min_segment_score", default=0.0, type=float)
    parser.add_argument("--min_segment_ratio", default=1.0, type=float)
    parser.add_argument("--soft_evidence", default=True, type=bool_flag)
    parser.add_argument("--max_margin", default=0.20, type=float)
    parser.add_argument("--min_top2_mass", default=0.35, type=float)
    parser.add_argument("--prototype_temperature", default=1.0, type=float)
    parser.add_argument("--baseline_pairs_per_sample", default=4, type=int)
    parser.add_argument("--baseline_mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--apply_source_reshaper", default=False, type=bool_flag)
    parser.add_argument("--shuffle_pair_baseline", default=True, type=bool_flag)
    parser.add_argument("--seed", default=None, type=int)
    return parser.parse_args()


def load_config(run_dir, data_root_override, device_override, seed_override):
    config_path = os.path.join(run_dir, "train_config.json")
    with open(config_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if data_root_override:
        payload["data_root"] = data_root_override
    if device_override:
        payload["device"] = device_override
    if seed_override is not None:
        payload["seed"] = int(seed_override)
    return Namespace(**payload)


def build_model(config):
    model_name = str(config.model).lower()
    if model_name == "pseltae":
        return PseLTae(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    if model_name == "psetae":
        return PseTae(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    if model_name == "psetcnn":
        return PseTempCNN(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    if model_name == "psegru":
        return PseGru(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    raise ValueError(f"Unsupported model: {config.model}")


def load_model_and_reshaper(config, checkpoint_path, device):
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    reshaper = build_source_feature_reshaper(
        kind=getattr(config, "source_feature_reshaper", "none"),
        feature_dim=model.spatial_encoder.output_dim,
        strength=getattr(config, "source_feature_reshaper_strength", 0.10),
        kernel_size=getattr(config, "source_feature_reshaper_kernel_size", 3),
    )
    if reshaper is not None:
        reshaper.to(device)
        if "source_feature_reshaper_state_dict" in checkpoint:
            reshaper.load_state_dict(checkpoint["source_feature_reshaper_state_dict"])
        reshaper.eval()
    return model, reshaper


def create_train_val_test_folds_local(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset_name in datasets:
            indices = list(range(num_indices[dataset_name]))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val
            random.shuffle(indices)
            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:]) if n_test > 0 else set()
            splits[dataset_name] = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
        folds.append(splits)
    return folds


def make_splits(config):
    classes = get_classes_for_config(config)
    config.classes = classes
    config.num_classes = len(classes)
    indices = {
        config.source: get_dataset_length(config, config.source, split="train"),
        config.target: get_dataset_length(config, config.target, split="train"),
    }
    return create_train_val_test_folds_local(
        [config.source, config.target],
        config.num_folds,
        indices,
        config.val_ratio,
        config.test_ratio,
    )[0]


def create_diagnostic_dataset(config, dataset_name, splits):
    transform = make_eval_transform(config, sample_pixels=True, sample_time=False)
    return build_dataset(
        config,
        dataset_name,
        config.classes,
        transform=transform,
        indices=splits[dataset_name]["train"],
        split="train",
    )


def create_diagnostic_loader(dataset, config):
    if is_har(config):
        return data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return data.DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(
            dataset,
            config.batch_size,
            by_pixel_dim=False,
        ),
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main():
    args = parse_args()
    config = load_config(args.run_dir, args.data_root, args.device, args.seed)
    seed = int(getattr(config, "seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    splits = make_splits(config)
    source_dataset = create_diagnostic_dataset(config, config.source, splits)
    target_dataset = create_diagnostic_dataset(config, config.target, splits)
    source_loader = create_diagnostic_loader(source_dataset, config)
    target_loader = create_diagnostic_loader(target_dataset, config)

    checkpoint_path = args.checkpoint or os.path.join(args.run_dir, "fold_0", "model.pt")
    model, reshaper = load_model_and_reshaper(config, checkpoint_path, device)

    partition_spec = build_atomic_partition_spec(source_dataset.date_positions, args.atomic_bins)
    print("atomic partition:", describe_atomic_partition_spec(partition_spec))

    if args.target_to_source_shift is None:
        max_shift = int(getattr(config, "max_temporal_shift", 60))
        shift = estimate_temporal_shift(
            model,
            target_loader,
            device,
            min_shift=-max_shift,
            max_shift=max_shift,
            sample_size=args.shift_sample_size,
            shift_estimator="IS",
        )
    else:
        shift = int(args.target_to_source_shift)
    print(f"target_to_source_shift={shift}")

    prototypes, variances, counts = compute_source_segment_prototypes(
        model,
        reshaper,
        source_loader,
        partition_spec,
        config.num_classes,
        device,
        max_batches=args.source_max_batches,
        apply_reshaper=bool(args.apply_source_reshaper),
    )
    discovery = discover_target_pair_segments(
        model,
        reshaper if bool(args.apply_source_reshaper) else None,
        target_loader,
        prototypes,
        variances,
        counts,
        partition_spec,
        device,
        target_to_source_shift=shift,
        max_batches=args.target_max_batches,
        max_margin=args.max_margin,
        min_top2_mass=args.min_top2_mass,
        prototype_temperature=args.prototype_temperature,
        shift_jitter=args.shift_jitter,
        soft_evidence=bool(args.soft_evidence),
        shuffle_pair_baseline=bool(args.shuffle_pair_baseline),
        baseline_pairs_per_sample=args.baseline_pairs_per_sample,
        baseline_mode=args.baseline_mode,
    )
    segment_threshold, adaptive_segments = construct_adaptive_segments(
        discovery["segment_rows_by_index"],
        score_quantile=args.segment_score_quantile,
        min_score=args.min_segment_score,
        min_ratio=args.min_segment_ratio,
    )
    pair_threshold, adaptive_supports = construct_pair_adaptive_supports(
        discovery["pair_segment_rows"],
        score_quantile=args.segment_score_quantile,
        min_score=args.min_segment_score,
        min_ratio=args.min_segment_ratio,
        top_m_per_pair=args.top_m_per_pair,
        max_supports=args.max_adaptive_supports,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "source": config.source,
        "target": config.target,
        "run_dir": args.run_dir,
        "checkpoint": checkpoint_path,
        "target_to_source_shift": int(shift),
        "candidate_shifts": discovery.get("shifts", []),
        "partition": partition_spec,
        "source_max_batches": args.source_max_batches,
        "target_max_batches": args.target_max_batches,
        "soft_evidence": bool(args.soft_evidence),
        "max_margin": args.max_margin,
        "min_top2_mass": args.min_top2_mass,
        "segment_score_threshold": segment_threshold,
        "pair_score_threshold": pair_threshold,
        "min_segment_ratio": args.min_segment_ratio,
        "top_m_per_pair": args.top_m_per_pair,
        "max_adaptive_supports": args.max_adaptive_supports,
        "seen_target_samples": discovery["seen_target_samples"],
        "accepted_ambiguous_samples": discovery["accepted_ambiguous_samples"],
        "accepted_fraction": discovery["accepted_fraction"],
        "accepted_evidence_weight": discovery["accepted_evidence_weight"],
        "top_segments": discovery["segment_rows"][: args.top_k_segments],
        "pair_rows": discovery["pair_rows"],
        "adaptive_segments": adaptive_segments,
        "adaptive_supports": adaptive_supports,
    }

    with open(os.path.join(args.output_dir, "adaptive_segments.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    with open(os.path.join(args.output_dir, "adaptive_supports.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    segment_fields = [
        "segment",
        "start",
        "end",
        "score",
        "support_count",
        "source_separability",
        "target_explainability",
        "ambiguity",
        "shift_stability",
        "actual_raw_score",
        "baseline_raw_score",
        "relative_score",
        "ratio_score",
        "baseline_separability",
        "baseline_explainability",
        "best_pair",
        "best_pair_score",
    ]
    pair_fields = [
        "pair",
        "score",
        "support_count",
        "source_separability",
        "target_explainability",
        "ambiguity",
        "shift_stability",
    ]
    pair_segment_fields = [
        "pair",
        "segment",
        "start",
        "end",
        "score",
        "support_count",
        "source_separability",
        "target_explainability",
        "ambiguity",
        "shift_stability",
        "actual_raw_score",
        "baseline_raw_score",
        "relative_score",
        "ratio_score",
        "baseline_separability",
        "baseline_explainability",
    ]
    write_csv(os.path.join(args.output_dir, "segment_scores.csv"), discovery["segment_rows"], segment_fields)
    write_csv(
        os.path.join(args.output_dir, "segment_scores_by_index.csv"),
        discovery["segment_rows_by_index"],
        segment_fields,
    )
    write_csv(
        os.path.join(args.output_dir, "baseline_segment_scores.csv"),
        discovery["baseline_segment_rows"],
        segment_fields,
    )
    write_csv(os.path.join(args.output_dir, "pair_scores.csv"), discovery["pair_rows"], pair_fields)
    write_csv(
        os.path.join(args.output_dir, "pair_segment_scores.csv"),
        discovery["pair_segment_rows"],
        pair_segment_fields,
    )

    print(
        f"adaptive supports: count={len(adaptive_supports)}, "
        f"pair_threshold={pair_threshold:.6f}, segment_threshold={segment_threshold:.6f}"
    )
    for item in adaptive_supports:
        print(
            f"  pair={item['class_pair']} interval=[{item['start']},{item['end']}] "
            f"score={item['score']:.6f} gate={item['gate']:.3f} atoms={item['atomic_segments']}"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
