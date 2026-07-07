import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.recompute_transfer_metrics import (  # noqa: E402
    DATASET_TO_TAG,
    build_dataset,
    build_loader,
    build_model,
    extract_domain_statistics,
    get_task_classes,
    load_checkpoint,
)
from utils.train_utils import bool_flag  # noqa: E402


NON_METRIC_FIELDS = {
    "source",
    "target",
    "source_dataset",
    "target_dataset",
    "target_f1",
    "source_checkpoint",
}


def coeff_var(values, dim=None, eps=1e-6):
    mean = values.mean(dim=dim)
    std = values.std(dim=dim, unbiased=False)
    return std / mean.abs().clamp_min(eps)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze source-side encoded feature curve dimensions against transfer performance."
    )
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data", type=str)
    parser.add_argument("--outputs_root", default="outputs", type=str)
    parser.add_argument("--transfer_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--source_experiment_suffix",
        required=True,
        type=str,
        help="Suffix appended after 'pseltae_<SRC>_closedset_noshift'. Example: '_sourcephasecompact_p5_v223_current_s010_rel003'.",
    )
    parser.add_argument("--closed_set", default=True, type=bool_flag)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument("--model", default="pseltae", choices=["pseltae", "psetae", "psetcnn", "psegru"])
    parser.add_argument("--with_extra", default=False, type=bool_flag)
    parser.add_argument("--temporal_grid_size", default=30, type=int)
    parser.add_argument("--max_acf_lag", default=10, type=int)
    parser.add_argument("--phase_count", default=5, type=int)
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(text)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_class_indices(domain_stats):
    return torch.where(domain_stats["curve_counts"] > 0)[0]


def uniform_phase_slices(total_steps, phase_count):
    return torch.tensor_split(torch.arange(total_steps, dtype=torch.long), phase_count)


def compute_phase_means(mean_curves, phase_slices):
    phase_means = []
    for phase_indices in phase_slices:
        phase_means.append(mean_curves[:, phase_indices, :].mean(dim=1))
    return torch.stack(phase_means, dim=1)


def compute_within_radius_curves(domain_stats, valid_classes, eps=1e-6):
    mean_curves = domain_stats["mean_curves"][valid_classes]
    counts = domain_stats["curve_counts"][valid_classes].unsqueeze(-1).to(dtype=mean_curves.dtype).clamp_min(1.0)
    mean_sq_norm = domain_stats["time_feature_sq_norm_sums"][valid_classes] / counts
    curve_sq_norm = mean_curves.pow(2).sum(dim=2)
    radius_sq = (mean_sq_norm - curve_sq_norm).clamp_min(0.0)
    return torch.sqrt(radius_sq + eps)


def compute_dimension_metrics(domain_stats, phase_count):
    valid_classes = valid_class_indices(domain_stats)
    if valid_classes.numel() == 0:
        raise ValueError("No valid classes found in source domain statistics.")

    mean_curves = domain_stats["mean_curves"][valid_classes].float()
    total_steps = int(mean_curves.shape[1])
    phase_slices = uniform_phase_slices(total_steps, phase_count)
    phase_means = compute_phase_means(mean_curves, phase_slices)

    amplitude_excursion = torch.norm(mean_curves.max(dim=1).values - mean_curves.min(dim=1).values, dim=1)
    curve_norms = torch.norm(mean_curves, dim=2)
    phase_energy = torch.stack([curve_norms[:, phase_indices].mean(dim=1) for phase_indices in phase_slices], dim=1)
    early_phase_energy = phase_energy[:, 0]
    late_phase_energy = phase_energy[:, -1]

    adjacent_phase_diffs = torch.norm(phase_means[:, 1:, :] - phase_means[:, :-1, :], dim=2)
    phase_pairs = []
    for phase_idx in range(phase_means.shape[1]):
        for other_idx in range(phase_idx + 1, phase_means.shape[1]):
            phase_pairs.append(torch.norm(phase_means[:, phase_idx, :] - phase_means[:, other_idx, :], dim=1))
    global_phase_contrast = torch.stack(phase_pairs, dim=1).mean(dim=1)
    early_late_distance = torch.norm(phase_means[:, -1, :] - phase_means[:, 0, :], dim=1)
    boundary_sharpness = adjacent_phase_diffs.max(dim=1).values / adjacent_phase_diffs.mean(dim=1).clamp_min(1e-6)

    within_radius_curves = compute_within_radius_curves(domain_stats, valid_classes)
    phase_radius_means = []
    phase_internal_step_norms = []
    for phase_indices in phase_slices:
        phase_radius_means.append(within_radius_curves[:, phase_indices].mean(dim=1))
        phase_curve = mean_curves[:, phase_indices, :]
        if phase_curve.shape[1] >= 2:
            phase_internal_step_norms.append(torch.norm(phase_curve[:, 1:, :] - phase_curve[:, :-1, :], dim=2).mean(dim=1))
    phase_radius_means = torch.stack(phase_radius_means, dim=1)
    phase_internal_step_norms = torch.stack(phase_internal_step_norms, dim=1) if phase_internal_step_norms else None

    step_norms = torch.norm(mean_curves[:, 1:, :] - mean_curves[:, :-1, :], dim=2)
    path_length = step_norms.sum(dim=1)
    displacement = torch.norm(mean_curves[:, -1, :] - mean_curves[:, 0, :], dim=1)
    directionality = displacement / path_length.clamp_min(1e-6)
    if step_norms.shape[1] >= 2:
        second_diffs = mean_curves[:, 2:, :] - 2.0 * mean_curves[:, 1:-1, :] + mean_curves[:, :-2, :]
        curvature = torch.norm(second_diffs, dim=2).mean(dim=1)
    else:
        curvature = torch.zeros_like(path_length)
    burstiness = step_norms.max(dim=1).values / step_norms.mean(dim=1).clamp_min(1e-6)

    metrics = {
        "source_encoded_amplitude_excursion_mean": float(amplitude_excursion.mean().item()),
        "source_encoded_amplitude_excursion_cv": float(coeff_var(amplitude_excursion).item()),
        "source_encoded_amplitude_phase_energy_cv_mean": float(coeff_var(phase_energy, dim=1).mean().item()),
        "source_encoded_amplitude_late_early_energy_ratio_mean": float(
            (late_phase_energy / early_phase_energy.clamp_min(1e-6)).mean().item()
        ),
        "source_encoded_interphase_adjacent_jump_mean": float(adjacent_phase_diffs.mean().item()),
        "source_encoded_interphase_adjacent_jump_cv": float(coeff_var(adjacent_phase_diffs, dim=1).mean().item()),
        "source_encoded_interphase_boundary_sharpness_mean": float(boundary_sharpness.mean().item()),
        "source_encoded_interphase_early_late_shift_mean": float(early_late_distance.mean().item()),
        "source_encoded_interphase_global_separation_mean": float(global_phase_contrast.mean().item()),
        "source_encoded_intraphase_radius_mean": float(within_radius_curves.mean().item()),
        "source_encoded_intraphase_radius_phase_cv_mean": float(coeff_var(phase_radius_means, dim=1).mean().item()),
        "source_encoded_intraphase_internal_step_mean": float(
            phase_internal_step_norms.mean().item() if phase_internal_step_norms is not None else 0.0
        ),
        "source_encoded_intraphase_late_early_radius_ratio_mean": float(
            (phase_radius_means[:, -1] / phase_radius_means[:, 0].clamp_min(1e-6)).mean().item()
        ),
        "source_encoded_trend_path_length_mean": float(path_length.mean().item()),
        "source_encoded_trend_directionality_mean": float(directionality.mean().item()),
        "source_encoded_trend_curvature_mean": float(curvature.mean().item()),
        "source_encoded_trend_burstiness_mean": float(burstiness.mean().item()),
    }
    return metrics


def infer_family(metric_name):
    if "_amplitude_" in metric_name:
        return "amplitude_spread"
    if "_interphase_" in metric_name:
        return "inter_phase_contrast"
    if "_intraphase_" in metric_name:
        return "intra_phase_compact_smooth"
    if "_trend_" in metric_name:
        return "trend_activity"
    return "other"


def infer_metric_fields(rows):
    if not rows:
        return []
    result = []
    for field in rows[0].keys():
        if field in NON_METRIC_FIELDS:
            continue
        if safe_float(rows[0].get(field)) is not None:
            result.append(field)
    return result


def compute_corr(rows, metric_field, score_field):
    pairs = []
    for row in rows:
        x = safe_float(row.get(metric_field))
        y = safe_float(row.get(score_field))
        if x is None or y is None:
            continue
        pairs.append((x, y))
    if len(pairs) < 2:
        return float("nan"), len(pairs)
    xs = np.asarray([x for x, _ in pairs], dtype=np.float64)
    ys = np.asarray([y for _, y in pairs], dtype=np.float64)
    if np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return float("nan"), len(pairs)
    return float(np.corrcoef(xs, ys)[0, 1]), len(pairs)


def build_correlation_rows(rows, metric_fields, score_field, score_label, group_name, group_value):
    result = []
    for metric in metric_fields:
        corr, pair_count = compute_corr(rows, metric, score_field)
        result.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "metric": metric,
                "family": infer_family(metric),
                score_label: corr,
                "abs_corr": abs(corr) if np.isfinite(corr) else float("nan"),
                "pair_count": pair_count,
                "task_count": len(rows),
            }
        )
    result.sort(
        key=lambda row: (
            1 if not np.isfinite(row[score_label]) else 0,
            0 if not np.isfinite(row[score_label]) else -abs(row[score_label]),
            row["metric"],
        )
    )
    return result


def build_family_summary(overall_rows, score_label):
    buckets = defaultdict(list)
    for row in overall_rows:
        if np.isfinite(row["abs_corr"]):
            buckets[row["family"]].append(row)

    summary_rows = []
    for family, family_rows in sorted(buckets.items()):
        family_rows = sorted(family_rows, key=lambda row: row["abs_corr"], reverse=True)
        summary_rows.append(
            {
                "family": family,
                "top_metric": family_rows[0]["metric"],
                "top_abs_corr": family_rows[0]["abs_corr"],
                "top_corr": family_rows[0][score_label],
                "avg_abs_corr": float(np.mean([row["abs_corr"] for row in family_rows])),
                "metric_count": len(family_rows),
            }
        )
    summary_rows.sort(key=lambda row: row["avg_abs_corr"], reverse=True)
    return summary_rows


def build_source_summary_rows(metric_rows, metric_fields):
    grouped = defaultdict(list)
    for row in metric_rows:
        grouped[row["source"]].append(row)

    summary_rows = []
    for source, rows in sorted(grouped.items()):
        row = {
            "source": source,
            "source_dataset": rows[0]["source_dataset"],
            "source_checkpoint": rows[0]["source_checkpoint"],
            "source_mean_f1": float(np.mean([safe_float(item["target_f1"]) for item in rows])),
            "source_f1_std": float(np.std([safe_float(item["target_f1"]) for item in rows])),
            "target_count": len(rows),
        }
        for metric in metric_fields:
            row[metric] = rows[0][metric]
        summary_rows.append(row)
    return summary_rows


def write_markdown(path, family_rows, source_mean_rows, task_rows, metric_fields, score_label):
    lines = [
        "# Encoded Structure Dimension Analysis",
        "",
        "This report analyzes source-side encoded feature curves only.",
        "The goal is to compare four structural dimension families against downstream transfer `macro_f1`.",
        "Primary correlations are computed against `source_mean_f1`, which matches the source-only design goal.",
        "",
        "## Family Definitions",
        "",
        "- `amplitude_spread`: excursion magnitude, phase energy distribution, and late/early energy bias of encoded class curves.",
        "- `inter_phase_contrast`: adjacent phase jumps, boundary sharpness, and long-range phase displacement.",
        "- `intra_phase_compact_smooth`: within-phase radius, phase-wise compactness heterogeneity, and internal step smoothness.",
        "- `trend_activity`: path length, directionality, curvature, and burstiness across encoded time trajectories.",
        "",
        "## Family Summary",
        "",
        "| family | top_metric | top_corr | top_abs_corr | avg_abs_corr | metric_count |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['top_metric']} | {row['top_corr']:.4f} | "
            f"{row['top_abs_corr']:.4f} | {row['avg_abs_corr']:.4f} | {row['metric_count']} |"
        )

    lines.extend(
        [
            "",
            "## Top Source-Mean Metrics",
            "",
            f"| metric | family | {score_label} | abs_corr |",
            "|---|---|---:|---:|",
        ]
    )
    for row in source_mean_rows[:12]:
        lines.append(
            f"| {row['metric']} | {row['family']} | {row[score_label]:.4f} | {row['abs_corr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Source Summary Table",
            "",
            "| source | source_mean_f1 | source_f1_std | amplitude_excursion_mean | interphase_adjacent_jump_mean | intraphase_radius_mean | trend_path_length_mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_rows:
        lines.append(
            f"| {row['source']} | {float(row['source_mean_f1']):.4f} | {float(row['source_f1_std']):.4f} | "
            f"{float(row['source_encoded_amplitude_excursion_mean']):.4f} | "
            f"{float(row['source_encoded_interphase_adjacent_jump_mean']):.4f} | "
            f"{float(row['source_encoded_intraphase_radius_mean']):.4f} | "
            f"{float(row['source_encoded_trend_path_length_mean']):.4f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def source_checkpoint_path(outputs_root, source_dataset, suffix):
    tile = source_dataset.split("/")[1]
    model_name = f"pseltae_{tile}_closedset_noshift{suffix}"
    return Path(outputs_root) / model_name / "fold_0" / "model.pt"


def main():
    args = parse_args()
    transfer_rows = read_csv(args.transfer_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    source_metric_cache = {}

    for source_dataset in sorted({row["source_dataset"] for row in transfer_rows}):
        classes = get_task_classes(args.data_root, source_dataset, args.closed_set)
        dataset = build_dataset(args.data_root, source_dataset, classes, args.closed_set, args.with_extra)
        loader = build_loader(dataset, args.batch_size, args.num_workers)
        model = build_model(args.model, len(classes), args.with_extra, device)
        checkpoint_path = source_checkpoint_path(args.outputs_root, source_dataset, args.source_experiment_suffix)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing source checkpoint for {source_dataset}: {checkpoint_path}")
        load_checkpoint(model, str(checkpoint_path), device)
        domain_stats = extract_domain_statistics(
            model=model,
            loader=loader,
            device=device,
            num_classes=len(classes),
            temporal_grid_size=args.temporal_grid_size,
            max_acf_lag=args.max_acf_lag,
            phase_only=True,
        )
        metrics = compute_dimension_metrics(domain_stats, args.phase_count)
        metrics["source_checkpoint"] = str(checkpoint_path)
        source_metric_cache[source_dataset] = metrics

    metric_rows = []
    for transfer_row in transfer_rows:
        source_dataset = transfer_row["source_dataset"]
        row = {
            "source": transfer_row["source"],
            "target": transfer_row["target"],
            "source_dataset": source_dataset,
            "target_dataset": transfer_row["target_dataset"],
            "target_f1": safe_float(transfer_row.get("target_f1")),
        }
        row.update(source_metric_cache[source_dataset])
        metric_rows.append(row)

    metric_fields = [
        field
        for field in metric_rows[0].keys()
        if field not in NON_METRIC_FIELDS and field != "source_checkpoint"
    ]
    write_csv(
        output_dir / "encoded_structure_dimension_rows.csv",
        metric_rows,
        ["source", "target", "source_dataset", "target_dataset", "target_f1", "source_checkpoint"] + metric_fields,
    )

    metric_fields = infer_metric_fields(metric_rows)
    source_summary_rows = build_source_summary_rows(metric_rows, metric_fields)
    write_csv(
        output_dir / "source_mean_dimension_rows.csv",
        source_summary_rows,
        ["source", "source_dataset", "source_checkpoint", "source_mean_f1", "source_f1_std", "target_count"] + metric_fields,
    )

    source_mean_corr_rows = build_correlation_rows(
        source_summary_rows,
        metric_fields,
        "source_mean_f1",
        "corr_with_source_mean_f1",
        "overall",
        "all_sources",
    )
    write_csv(
        output_dir / "source_mean_metric_correlations.csv",
        source_mean_corr_rows,
        [
            "group_name",
            "group_value",
            "metric",
            "family",
            "corr_with_source_mean_f1",
            "abs_corr",
            "pair_count",
            "task_count",
        ],
    )

    overall_rows = build_correlation_rows(
        metric_rows,
        metric_fields,
        "target_f1",
        "corr_with_target_f1",
        "overall",
        "all",
    )
    write_csv(
        output_dir / "overall_metric_correlations.csv",
        overall_rows,
        ["group_name", "group_value", "metric", "family", "corr_with_target_f1", "abs_corr", "pair_count", "task_count"],
    )

    by_source_rows = []
    grouped_by_source = defaultdict(list)
    for row in metric_rows:
        grouped_by_source[row["source"]].append(row)
    for source, rows in sorted(grouped_by_source.items()):
        by_source_rows.extend(
            build_correlation_rows(rows, metric_fields, "target_f1", "corr_with_target_f1", "source", source)
        )
    write_csv(
        output_dir / "by_source_metric_correlations.csv",
        by_source_rows,
        ["group_name", "group_value", "metric", "family", "corr_with_target_f1", "abs_corr", "pair_count", "task_count"],
    )

    by_target_rows = []
    grouped_by_target = defaultdict(list)
    for row in metric_rows:
        grouped_by_target[row["target"]].append(row)
    for target, rows in sorted(grouped_by_target.items()):
        by_target_rows.extend(
            build_correlation_rows(rows, metric_fields, "target_f1", "corr_with_target_f1", "target", target)
        )
    write_csv(
        output_dir / "by_target_metric_correlations.csv",
        by_target_rows,
        ["group_name", "group_value", "metric", "family", "corr_with_target_f1", "abs_corr", "pair_count", "task_count"],
    )

    family_rows = build_family_summary(source_mean_corr_rows, "corr_with_source_mean_f1")
    write_csv(
        output_dir / "dimension_family_summary.csv",
        family_rows,
        ["family", "top_metric", "top_abs_corr", "top_corr", "avg_abs_corr", "metric_count"],
    )
    write_markdown(
        output_dir / "encoded_structure_dimension_summary.md",
        family_rows,
        source_mean_corr_rows,
        source_summary_rows,
        metric_fields,
        "corr_with_source_mean_f1",
    )
    print(f"[INFO] Saved encoded structure dimension analysis to: {output_dir}")


if __name__ == "__main__":
    main()
