import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from timematch import estimate_temporal_shift
from v272_adaptive_segment_discovery import (
    build_atomic_partition_spec,
    compute_source_segment_prototypes,
    construct_adaptive_segments,
    create_diagnostic_dataset,
    create_diagnostic_loader,
    describe_atomic_partition_spec,
    discover_target_segments,
    load_config,
    load_model_and_reshaper,
    make_splits,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "v2.7.3a diagnostic: estimate DA-stage confusion-view reliability "
            "without changing training loss."
        )
    )
    parser.add_argument("--run_dir", required=True, help="Source or DA run directory containing train_config.json.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to run_dir/fold_0/model.pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default="", help="Override data_root from train_config.json.")
    parser.add_argument("--source_max_batches", default=64, type=int)
    parser.add_argument("--target_max_batches", default=64, type=int)
    parser.add_argument("--shift_sample_size", default=40, type=int)
    parser.add_argument("--target_to_source_shift", default=None, type=int)
    parser.add_argument("--atomic_bins", default=12, type=int)
    parser.add_argument("--segment_score_quantile", default=0.75, type=float)
    parser.add_argument("--min_segment_score", default=0.0, type=float)
    parser.add_argument("--min_segment_ratio", default=1.0, type=float)
    parser.add_argument("--max_margin", default=0.20, type=float)
    parser.add_argument("--min_top2_mass", default=0.35, type=float)
    parser.add_argument("--prototype_temperature", default=1.0, type=float)
    parser.add_argument("--baseline_pairs_per_sample", default=4, type=int)
    parser.add_argument("--baseline_mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--apply_source_reshaper", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes"})
    parser.add_argument("--shuffle_pair_baseline", default=True, type=lambda x: str(x).lower() in {"1", "true", "yes"})
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gate_score_high", default=0.10, type=float)
    parser.add_argument("--gate_score_low", default=0.02, type=float)
    parser.add_argument("--gate_ratio_low", default=1.00, type=float)
    parser.add_argument("--gate_ratio_high", default=1.20, type=float)
    parser.add_argument("--gate_accept_low", default=0.01, type=float)
    parser.add_argument("--gate_accept_high", default=0.08, type=float)
    parser.add_argument("--gate_accept_too_high", default=0.16, type=float)
    parser.add_argument("--gate_overlap_high", default=0.50, type=float)
    parser.add_argument("--gate_post_pre_low", default=0.15, type=float)
    parser.add_argument("--gate_post_pre_high", default=0.70, type=float)
    return parser.parse_args()


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _ramp(value, low, high):
    if float(high) <= float(low):
        return 1.0 if float(value) >= float(high) else 0.0
    return _clamp01((float(value) - float(low)) / (float(high) - float(low)))


def _band_gate(value, low, high, too_high):
    value = float(value)
    if value <= float(low):
        return 0.0
    if value <= float(high):
        return _ramp(value, low, high)
    if value <= float(too_high):
        return 1.0
    return _clamp01(1.0 - (value - float(too_high)) / max(float(too_high), 1e-6))


def _rows_to_positive_atoms(rows, min_score, min_ratio):
    atoms = set()
    for row in rows:
        if row.get("score", 0.0) > float(min_score) and row.get("ratio_score", 0.0) >= float(min_ratio):
            atoms.add(int(row["segment"]))
    return atoms


def _overlap_score(pre_rows_by_index, post_rows_by_index, min_score, min_ratio):
    pre_atoms = _rows_to_positive_atoms(pre_rows_by_index, min_score, min_ratio)
    post_atoms = _rows_to_positive_atoms(post_rows_by_index, min_score, min_ratio)
    if not pre_atoms and not post_atoms:
        return 0.0, 0, 0, 0
    inter = pre_atoms & post_atoms
    union = pre_atoms | post_atoms
    return len(inter) / max(len(union), 1), len(pre_atoms), len(post_atoms), len(inter)


def _top_signal(rows):
    if not rows:
        return {"score": 0.0, "ratio_score": 0.0, "support_count": 0, "start": None, "end": None}
    row = max(rows, key=lambda item: item.get("score", 0.0))
    return {
        "score": float(row.get("score", 0.0)),
        "ratio_score": float(row.get("ratio_score", 0.0)),
        "support_count": int(row.get("support_count", 0)),
        "start": row.get("start"),
        "end": row.get("end"),
        "best_pair": row.get("best_pair", row.get("dominant_pair", "")),
    }


def _compute_gate(args, pre_top, post_discovery, post_top, overlap):
    score_gate = _ramp(post_top["score"], args.gate_score_low, args.gate_score_high)
    ratio_gate = _ramp(post_top["ratio_score"], args.gate_ratio_low, args.gate_ratio_high)
    accept_gate = _band_gate(
        post_discovery["accepted_fraction"],
        args.gate_accept_low,
        args.gate_accept_high,
        args.gate_accept_too_high,
    )
    stability_gate = _clamp01(overlap / max(float(args.gate_overlap_high), 1e-6))
    post_pre_ratio = float(post_top["score"]) / max(float(pre_top["score"]), 1e-6)
    post_pre_gate = _ramp(post_pre_ratio, args.gate_post_pre_low, args.gate_post_pre_high)
    gate = score_gate * ratio_gate * accept_gate * stability_gate * post_pre_gate
    return {
        "gate": gate,
        "score_gate": score_gate,
        "ratio_gate": ratio_gate,
        "accepted_fraction_gate": accept_gate,
        "pre_post_overlap_gate": stability_gate,
        "post_pre_score_ratio": post_pre_ratio,
        "post_pre_score_gate": post_pre_gate,
    }


def _run_discovery(
    args,
    model,
    reshaper,
    target_loader,
    prototypes,
    variances,
    counts,
    phase_partition_spec,
    config,
    device,
    shift,
    label,
):
    print(f"running {label} discovery with target_to_source_shift={shift}")
    discovery = discover_target_segments(
        model,
        reshaper if bool(args.apply_source_reshaper) else None,
        target_loader,
        prototypes,
        variances,
        counts,
        phase_partition_spec,
        config,
        device,
        target_to_source_shift=int(shift),
        max_batches=args.target_max_batches,
        max_margin=args.max_margin,
        min_top2_mass=args.min_top2_mass,
        prototype_temperature=args.prototype_temperature,
        shift_jitter=0,
        shuffle_pair_baseline=bool(args.shuffle_pair_baseline),
        baseline_pairs_per_sample=args.baseline_pairs_per_sample,
        baseline_mode=args.baseline_mode,
    )
    threshold, adaptive_segments = construct_adaptive_segments(
        discovery["segment_rows_by_index"],
        score_quantile=args.segment_score_quantile,
        min_score=args.min_segment_score,
        min_ratio=args.min_segment_ratio,
    )
    return discovery, threshold, adaptive_segments


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

    phase_partition_spec = build_atomic_partition_spec(source_dataset.date_positions, args.atomic_bins)
    print("atomic partition:", describe_atomic_partition_spec(phase_partition_spec))

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
    print(f"estimated target_to_source_shift={shift}")

    prototypes, variances, counts = compute_source_segment_prototypes(
        model,
        reshaper,
        source_loader,
        phase_partition_spec,
        config.num_classes,
        device,
        max_batches=args.source_max_batches,
        apply_reshaper=bool(args.apply_source_reshaper),
    )

    pre_discovery, pre_threshold, pre_segments = _run_discovery(
        args,
        model,
        reshaper,
        target_loader,
        prototypes,
        variances,
        counts,
        phase_partition_spec,
        config,
        device,
        shift=0,
        label="pre-shift",
    )
    post_discovery, post_threshold, post_segments = _run_discovery(
        args,
        model,
        reshaper,
        target_loader,
        prototypes,
        variances,
        counts,
        phase_partition_spec,
        config,
        device,
        shift=shift,
        label="post-shift",
    )

    overlap, pre_atom_count, post_atom_count, overlap_count = _overlap_score(
        pre_discovery["segment_rows_by_index"],
        post_discovery["segment_rows_by_index"],
        min_score=args.min_segment_score,
        min_ratio=args.min_segment_ratio,
    )
    pre_top = _top_signal(pre_discovery["segment_rows_by_index"])
    post_top = _top_signal(post_discovery["segment_rows_by_index"])
    gate = _compute_gate(args, pre_top, post_discovery, post_top, overlap)

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "source": config.source,
        "target": config.target,
        "run_dir": args.run_dir,
        "checkpoint": checkpoint_path,
        "target_to_source_shift": int(shift),
        "segment_partition": phase_partition_spec,
        "pre_shift": {
            "accepted_fraction": pre_discovery["accepted_fraction"],
            "accepted_ambiguous_samples": pre_discovery["accepted_ambiguous_samples"],
            "seen_target_samples": pre_discovery["seen_target_samples"],
            "score_threshold": pre_threshold,
            "top_signal": pre_top,
            "adaptive_segments": pre_segments,
        },
        "post_shift": {
            "accepted_fraction": post_discovery["accepted_fraction"],
            "accepted_ambiguous_samples": post_discovery["accepted_ambiguous_samples"],
            "seen_target_samples": post_discovery["seen_target_samples"],
            "score_threshold": post_threshold,
            "top_signal": post_top,
            "adaptive_segments": post_segments,
        },
        "pre_post": {
            "positive_atom_overlap": overlap,
            "pre_positive_atom_count": pre_atom_count,
            "post_positive_atom_count": post_atom_count,
            "overlap_atom_count": overlap_count,
        },
        "confusion_gate": gate,
        "gate_config": {
            "score_low": args.gate_score_low,
            "score_high": args.gate_score_high,
            "ratio_low": args.gate_ratio_low,
            "ratio_high": args.gate_ratio_high,
            "accept_low": args.gate_accept_low,
            "accept_high": args.gate_accept_high,
            "accept_too_high": args.gate_accept_too_high,
            "overlap_high": args.gate_overlap_high,
            "post_pre_low": args.gate_post_pre_low,
            "post_pre_high": args.gate_post_pre_high,
        },
    }
    json_path = os.path.join(args.output_dir, "confusion_gate_diagnostic.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    write_csv(
        os.path.join(args.output_dir, "pre_shift_atomic_bin_scores.csv"),
        pre_discovery["segment_rows"],
        [
            "segment",
            "start",
            "end",
            "score",
            "support_count",
            "source_separability",
            "target_explainability",
            "ambiguity",
            "actual_raw_score",
            "baseline_raw_score",
            "relative_score",
            "ratio_score",
            "baseline_separability",
            "baseline_explainability",
            "best_pair",
            "best_pair_score",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "post_shift_atomic_bin_scores.csv"),
        post_discovery["segment_rows"],
        [
            "segment",
            "start",
            "end",
            "score",
            "support_count",
            "source_separability",
            "target_explainability",
            "ambiguity",
            "actual_raw_score",
            "baseline_raw_score",
            "relative_score",
            "ratio_score",
            "baseline_separability",
            "baseline_explainability",
            "best_pair",
            "best_pair_score",
        ],
    )
    with open(os.path.join(args.output_dir, "summary.tsv"), "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "source",
                "target",
                "shift",
                "gate",
                "post_score",
                "post_ratio",
                "post_accepted_fraction",
                "pre_score",
                "pre_ratio",
                "overlap",
                "post_pre_score_ratio",
                "pre_atoms",
                "post_atoms",
                "overlap_atoms",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "source": config.source,
                "target": config.target,
                "shift": int(shift),
                "gate": gate["gate"],
                "post_score": post_top["score"],
                "post_ratio": post_top["ratio_score"],
                "post_accepted_fraction": post_discovery["accepted_fraction"],
                "pre_score": pre_top["score"],
                "pre_ratio": pre_top["ratio_score"],
                "overlap": overlap,
                "post_pre_score_ratio": gate["post_pre_score_ratio"],
                "pre_atoms": pre_atom_count,
                "post_atoms": post_atom_count,
                "overlap_atoms": overlap_count,
            }
        )

    print(f"Saved {json_path}")
    print(
        "CONF_GATE "
        f"gate={gate['gate']:.4f}, shift={shift}, "
        f"post_score={post_top['score']:.4f}, post_ratio={post_top['ratio_score']:.3f}, "
        f"post_acc={post_discovery['accepted_fraction']:.4f}, "
        f"pre_score={pre_top['score']:.4f}, pre_ratio={pre_top['ratio_score']:.3f}, "
        f"overlap={overlap:.3f} ({overlap_count}/{max(pre_atom_count + post_atom_count - overlap_count, 1)}), "
        f"post_pre={gate['post_pre_score_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
