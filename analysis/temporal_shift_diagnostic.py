#!/usr/bin/env python3
"""
Diagnose whether source-target temporal mismatch is global or locally structured.

This script is intentionally offline and model-free. It compares source and target
class prototype curves in the raw input space, then asks whether different
partition views explain the temporal distribution of the domain mismatch.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import zarr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import PixelSetData
from ideas.source_phase_compactness import (
    build_source_segment_partition_spec,
    describe_source_segment_partition_spec,
)
from utils import label_utils


DEFAULT_TASKS = [
    "FR1_to_FR2|france/30TXT/2017|france/31TCJ/2017",
    "FR2_to_FR1|france/31TCJ/2017|france/30TXT/2017",
    "DK1_to_FR1|denmark/32VNH/2017|france/30TXT/2017",
]


def _parse_task(text):
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 3:
        raise ValueError(
            "Task must be formatted as name|source_dataset|target_dataset, "
            f"got: {text}"
        )
    return {"name": parts[0], "source": parts[1], "target": parts[2]}


def _country(dataset_name):
    return dataset_name.split("/")[-3]


def _closed_set_classes(source_dataset, target_dataset, combine_spring_and_winter=False):
    source_country = _country(source_dataset)
    target_country = _country(target_dataset)
    source_classes = label_utils.get_classes(
        source_country,
        combine_spring_and_winter=combine_spring_and_winter,
    )
    target_classes = label_utils.get_classes(
        target_country,
        combine_spring_and_winter=combine_spring_and_winter,
    )
    common = [cls for cls in source_classes if cls in set(target_classes)]
    return [cls for cls in common if cls != "unknown"]


def _collect_class_curve_stats(dataset, max_samples_per_class):
    sums = {}
    sum_squares = {}
    counts = defaultdict(int)
    for path, _parcel_idx, label, _extra in dataset.samples:
        if counts[label] >= max_samples_per_class:
            continue
        pixels = zarr.load(path).astype(np.float64)  # (T, C, S)
        curve = np.nanmean(pixels, axis=-1)  # (T, C)
        if label not in sums:
            sums[label] = np.zeros_like(curve, dtype=np.float64)
            sum_squares[label] = np.zeros_like(curve, dtype=np.float64)
        sums[label] += curve
        sum_squares[label] += np.square(curve)
        counts[label] += 1

    means = {}
    dispersions = {}
    for label, total in sums.items():
        if counts[label] > 0:
            count = float(counts[label])
            mean = total / count
            var = np.maximum((sum_squares[label] / count) - np.square(mean), 0.0)
            means[int(label)] = mean
            dispersions[int(label)] = np.sqrt(var)
    return means, dispersions, {int(label): int(count) for label, count in counts.items()}


def _common_time_grid(source_positions, target_positions):
    source_positions = np.asarray(source_positions, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    start = max(float(source_positions.min()), float(target_positions.min()))
    end = min(float(source_positions.max()), float(target_positions.max()))
    source_overlap = source_positions[
        (source_positions >= start) & (source_positions <= end)
    ]
    target_overlap = target_positions[
        (target_positions >= start) & (target_positions <= end)
    ]
    grid = np.unique(np.concatenate([source_overlap, target_overlap]))
    if grid.size >= 3:
        return grid
    length = max(3, min(source_positions.size, target_positions.size))
    return np.linspace(start, end, length)


def _interp_curve(curve, positions, grid):
    positions = np.asarray(positions, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    channels = []
    for channel_idx in range(curve.shape[1]):
        channels.append(np.interp(grid, positions, curve[:, channel_idx]))
    return np.stack(channels, axis=1)


def _standardized_curve_distance(source_curve, target_curve):
    stacked = np.concatenate([source_curve, target_curve], axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    source_z = (source_curve - mean) / std
    target_z = (target_curve - mean) / std
    return np.linalg.norm(source_z - target_z, axis=1)


def _source_mad_curve_distance(source_curve, target_curve, source_scale):
    scale = np.asarray(source_scale, dtype=np.float64)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return np.mean(np.abs(source_curve - target_curve) / scale.reshape(1, -1), axis=1)


def _task_distance_curve(source_dataset, target_dataset, max_samples_per_class):
    source_means, source_dispersions, source_counts = _collect_class_curve_stats(
        source_dataset,
        max_samples_per_class=max_samples_per_class,
    )
    target_means, target_dispersions, target_counts = _collect_class_curve_stats(
        target_dataset,
        max_samples_per_class=max_samples_per_class,
    )
    grid = _common_time_grid(source_dataset.date_positions, target_dataset.date_positions)
    class_pairs = []
    class_details = []
    for label in sorted(set(source_means) & set(target_means)):
        source_curve = _interp_curve(source_means[label], source_dataset.date_positions, grid)
        target_curve = _interp_curve(target_means[label], target_dataset.date_positions, grid)
        source_dispersion = _interp_curve(
            source_dispersions[label],
            source_dataset.date_positions,
            grid,
        )
        target_dispersion = _interp_curve(
            target_dispersions[label],
            target_dataset.date_positions,
            grid,
        )
        class_pairs.append((label, source_curve, target_curve, source_dispersion, target_dispersion))
        class_details.append(
            {
                "label": int(label),
                "class_name": source_dataset.classes[label],
                "source_samples": int(source_counts.get(label, 0)),
                "target_samples": int(target_counts.get(label, 0)),
            }
        )

    if not class_pairs:
        raise ValueError(
            f"No shared classes with samples for {source_dataset.dataset_name} -> "
            f"{target_dataset.dataset_name}"
        )

    source_stack = np.concatenate([pair[1] for pair in class_pairs], axis=0)
    source_median = np.median(source_stack, axis=0, keepdims=True)
    source_scale = np.median(np.abs(source_stack - source_median), axis=0)
    source_scale = np.where(source_scale < 1e-6, 1.0, source_scale)

    distance_by_mode = {
        "source_mad_l1": [],
        "joint_z_l2": [],
    }
    source_dispersion_curves = []
    target_dispersion_curves = []
    for idx, (_label, source_curve, target_curve, source_dispersion, target_dispersion) in enumerate(class_pairs):
        source_mad_distance = _source_mad_curve_distance(source_curve, target_curve, source_scale)
        joint_z_distance = _standardized_curve_distance(source_curve, target_curve)
        distance_by_mode["source_mad_l1"].append(source_mad_distance)
        distance_by_mode["joint_z_l2"].append(joint_z_distance)
        source_dispersion_curve = np.mean(source_dispersion / source_scale.reshape(1, -1), axis=1)
        target_dispersion_curve = np.mean(target_dispersion / source_scale.reshape(1, -1), axis=1)
        source_dispersion_curves.append(source_dispersion_curve)
        target_dispersion_curves.append(target_dispersion_curve)
        class_details[idx]["source_mad_l1_mean"] = float(source_mad_distance.mean())
        class_details[idx]["joint_z_l2_mean"] = float(joint_z_distance.mean())
        class_details[idx]["source_dispersion_mean"] = float(source_dispersion_curve.mean())
        class_details[idx]["target_dispersion_mean"] = float(target_dispersion_curve.mean())

    distance_curves = {
        mode: np.mean(np.stack(curves, axis=0), axis=0)
        for mode, curves in distance_by_mode.items()
    }
    dispersion = {
        "source": np.mean(np.stack(source_dispersion_curves, axis=0), axis=0),
        "target": np.mean(np.stack(target_dispersion_curves, axis=0), axis=0),
    }
    return distance_curves, dispersion, grid, class_details


def _parse_partition(text):
    if ":" in text:
        mode, count = text.split(":", 1)
    else:
        mode, count = text, "1"
    return mode.strip(), int(count)


def _partition_segments(mode, count, positions, source_positions):
    indices = np.arange(len(positions))
    if count <= 1:
        return [indices], "global"

    mode = mode.lower()
    if mode == "uniform":
        segments = [segment for segment in np.array_split(indices, count) if segment.size > 0]
        return segments, f"uniform:{len(segments)}"

    spec = build_source_segment_partition_spec(
        source_positions,
        mode=mode,
        segment_count=count,
    )
    segments = []
    for start, end in spec["intervals"]:
        segment = indices[(positions >= start) & (positions <= end)]
        if segment.size > 0:
            segments.append(segment)
    if not segments:
        segments = [indices]
    return segments, describe_source_segment_partition_spec(spec)


def _boundary_mask(segments, length, window):
    mask = np.zeros(length, dtype=bool)
    for segment in segments[:-1]:
        if segment.size == 0:
            continue
        boundary = int(segment[-1])
        start = max(0, boundary - window)
        end = min(length, boundary + window + 1)
        mask[start:end] = True
    return mask


def _diagnostic_hint(row):
    if row["locality_peak_ratio"] < 1.35 and row["locality_top20_mass"] < 0.35:
        return "global_compactness"
    if row["complexity_adjusted_compression"] >= 0.10:
        return "segmented_compactness"
    if row["temporal_variation_ratio"] >= 0.35:
        return "trajectory_dynamics_probe"
    return "coarse_or_global_compactness"


def _partition_metrics(distance_curve, positions, segments, boundary_window):
    eps = 1e-8
    distance_curve = np.asarray(distance_curve, dtype=np.float64)
    mean_distance = float(distance_curve.mean())
    total_distance = float(distance_curve.sum())
    top_k = max(1, int(math.ceil(distance_curve.size * 0.2)))
    top20_mass = float(np.sort(distance_curve)[-top_k:].sum() / max(total_distance, eps))
    peak_ratio = float(distance_curve.max() / max(mean_distance, eps))
    variation_ratio = float(
        np.abs(np.diff(distance_curve)).mean() / max(mean_distance, eps)
    ) if distance_curve.size > 1 else 0.0
    global_var = float(distance_curve.var())

    weighted_within_var = 0.0
    segment_means = []
    for segment in segments:
        values = distance_curve[segment]
        if values.size == 0:
            continue
        weighted_within_var += float(values.var()) * (float(values.size) / distance_curve.size)
        segment_means.append(float(values.mean()))
    compression = 0.0
    if global_var > eps:
        compression = float(1.0 - weighted_within_var / global_var)
    segment_count = max(1, len(segments))
    complexity_adjusted_compression = float(compression / math.sqrt(segment_count))

    boundary = _boundary_mask(segments, distance_curve.size, boundary_window)
    if boundary.any():
        boundary_mass_ratio = float(distance_curve[boundary].mean() / max(mean_distance, eps))
        boundary_mass_fraction = float(distance_curve[boundary].sum() / max(total_distance, eps))
    else:
        boundary_mass_ratio = 0.0
        boundary_mass_fraction = 0.0

    segment_contrast = 0.0
    if len(segment_means) > 1:
        segment_contrast = float(np.std(segment_means) / max(mean_distance, eps))

    return {
        "common_points": int(distance_curve.size),
        "mean_domain_distance": mean_distance,
        "locality_peak_ratio": peak_ratio,
        "locality_top20_mass": top20_mass,
        "temporal_variation_ratio": variation_ratio,
        "partition_compression": compression,
        "complexity_adjusted_compression": complexity_adjusted_compression,
        "segment_contrast": segment_contrast,
        "boundary_mass_ratio": boundary_mass_ratio,
        "boundary_mass_fraction": boundary_mass_fraction,
        "segment_sizes": [int(segment.size) for segment in segments],
        "segment_count": int(segment_count),
        "position_start": float(positions[0]),
        "position_end": float(positions[-1]),
    }


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "task",
        "source",
        "target",
        "distance_mode",
        "partition",
        "partition_description",
        "class_count",
        "common_points",
        "mean_domain_distance",
        "locality_peak_ratio",
        "locality_top20_mass",
        "temporal_variation_ratio",
        "partition_compression",
        "complexity_adjusted_compression",
        "segment_contrast",
        "boundary_mass_ratio",
        "boundary_mass_fraction",
        "source_dispersion_mean",
        "target_dispersion_mean",
        "domain_to_source_dispersion_ratio",
        "diagnostic_hint",
        "segment_sizes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["segment_sizes"] = json.dumps(out["segment_sizes"], ensure_ascii=False)
            writer.writerow(out)


def _write_markdown(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best_by_task = {}
    for row in rows:
        key = (row["task"], row["distance_mode"])
        current = best_by_task.get(key)
        if current is None or row["complexity_adjusted_compression"] > current["complexity_adjusted_compression"]:
            best_by_task[key] = row

    lines = [
        "# v2.5 Temporal Structure Diagnostic",
        "",
        "This report diagnoses raw source-target temporal mismatch. It is a guide for",
        "choosing structure views, not a replacement for DA validation.",
        "",
        "| task | distance | best partition | adj. compression | raw compression | peak ratio | top20 mass | variation | hint |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in sorted(best_by_task):
        row = best_by_task[key]
        lines.append(
            "| {task} | {distance_mode} | {partition} | {adjusted:.4f} | "
            "{compression:.4f} | {peak:.4f} | {top20:.4f} | {variation:.4f} | {hint} |".format(
                task=row["task"],
                distance_mode=row["distance_mode"],
                partition=row["partition"],
                adjusted=row["complexity_adjusted_compression"],
                compression=row["partition_compression"],
                peak=row["locality_peak_ratio"],
                top20=row["locality_top20_mass"],
                variation=row["temporal_variation_ratio"],
                hint=row["diagnostic_hint"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `global_compactness`: mismatch is diffuse; no-seg/global compactness is a plausible first view.",
            "- `segmented_compactness`: mismatch is locally structured; segment-wise compactness has a diagnostic basis.",
            "- `trajectory_dynamics_probe`: mismatch changes sharply over time; dynamics should be tested carefully.",
            "- `source_mad_l1`: preserves source-target offsets after scaling channels by source-domain MAD.",
            "- `joint_z_l2`: shape-only view; useful as a contrast because it removes class-wise scale/offset.",
            "- `partition_compression`: how much a partition reduces within-segment variance of the mismatch curve.",
            "- `adj. compression`: compression divided by sqrt(segment count), to reduce the trivial advantage of finer partitions.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["uniform:1", "uniform:2", "uniform:5", "uniform:10", "doy_gap:5"],
        help="Partition views to diagnose, formatted as mode:count.",
    )
    parser.add_argument("--max_samples_per_class", type=int, default=128)
    parser.add_argument("--boundary_window", type=int, default=1)
    parser.add_argument("--combine_spring_and_winter", action="store_true")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    rows = []
    details = {}
    for task_text in args.tasks:
        task = _parse_task(task_text)
        classes = _closed_set_classes(
            task["source"],
            task["target"],
            combine_spring_and_winter=args.combine_spring_and_winter,
        )
        if not classes:
            raise ValueError(f"No shared closed-set classes for {task_text}")
        source_dataset = PixelSetData(
            args.data_root,
            task["source"],
            classes,
            closed_set=True,
        )
        target_dataset = PixelSetData(
            args.data_root,
            task["target"],
            classes,
            closed_set=True,
        )
        distance_curves, dispersion, positions, class_details = _task_distance_curve(
            source_dataset,
            target_dataset,
            max_samples_per_class=args.max_samples_per_class,
        )

        details[task["name"]] = {
            "source": task["source"],
            "target": task["target"],
            "classes": classes,
            "class_details": class_details,
            "positions": [float(pos) for pos in positions.tolist()],
            "distance_curves": {
                mode: [float(value) for value in curve.tolist()]
                for mode, curve in distance_curves.items()
            },
            "source_dispersion_curve": [float(value) for value in dispersion["source"].tolist()],
            "target_dispersion_curve": [float(value) for value in dispersion["target"].tolist()],
        }

        for partition_text in args.partitions:
            mode, count = _parse_partition(partition_text)
            segments, description = _partition_segments(
                mode,
                count,
                positions=positions,
                source_positions=source_dataset.date_positions,
            )
            for distance_mode, distance_curve in distance_curves.items():
                metrics = _partition_metrics(
                    distance_curve,
                    positions,
                    segments,
                    boundary_window=args.boundary_window,
                )
                source_dispersion_mean = float(dispersion["source"].mean())
                target_dispersion_mean = float(dispersion["target"].mean())
                row = {
                    "task": task["name"],
                    "source": task["source"],
                    "target": task["target"],
                    "distance_mode": distance_mode,
                    "partition": f"{mode}:{count}",
                    "partition_description": description,
                    "class_count": len(class_details),
                    "source_dispersion_mean": source_dispersion_mean,
                    "target_dispersion_mean": target_dispersion_mean,
                    "domain_to_source_dispersion_ratio": float(
                        metrics["mean_domain_distance"] / max(source_dispersion_mean, 1e-8)
                    ),
                    **metrics,
                }
                row["diagnostic_hint"] = _diagnostic_hint(row)
                rows.append(row)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    _write_csv(args.output_csv, rows)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "details": details}, handle, indent=2, ensure_ascii=False)
    if args.output_md:
        _write_markdown(args.output_md, rows)

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSON: {args.output_json}")
    if args.output_md:
        print(f"Wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
