import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import zarr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import PixelSetData
from ideas.source_phase_compactness import build_source_segment_partition_spec, describe_source_segment_partition_spec
from utils import label_utils


ORACLE_F1_BY_TASK_EPOCH = {
    "30TXT->31TCJ": {30: 0.7904, 50: 0.8020, 70: 0.7982, 100: 0.7976},
    "30TXT->32VNH": {30: 0.6455, 50: 0.6490, 70: 0.6570, 100: 0.6444},
    "30TXT->33UVP": {30: 0.6986, 50: 0.7605, 70: 0.7388, 100: 0.7637},
    "31TCJ->30TXT": {30: 0.7672, 50: 0.7553, 70: 0.7164, 100: 0.7334},
    "31TCJ->32VNH": {30: 0.5724, 50: 0.6957, 70: 0.5004, 100: 0.5526},
    "31TCJ->33UVP": {30: 0.6676, 50: 0.6027, 70: 0.5888, 100: 0.6509},
    "32VNH->30TXT": {30: 0.5856, 50: 0.5834, 70: 0.5927, 100: 0.5807},
    "32VNH->31TCJ": {30: 0.5270, 50: 0.5474, 70: 0.5295, 100: 0.5361},
    "32VNH->33UVP": {30: 0.6674, 50: 0.4531, 70: 0.5654, 100: 0.5789},
    "33UVP->30TXT": {30: 0.6694, 50: 0.6786, 70: 0.6815, 100: 0.6780},
    "33UVP->31TCJ": {30: 0.6081, 50: 0.6158, 70: 0.6501, 100: 0.6357},
    "33UVP->32VNH": {30: 0.7765, 50: 0.7311, 70: 0.7034, 100: 0.7047},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "france/30TXT/2017",
            "france/31TCJ/2017",
            "denmark/32VNH/2017",
            "austria/33UVP/2017",
        ],
    )
    parser.add_argument("--closed_set", default=True, type=bool_arg)
    parser.add_argument("--max_samples_per_class", default=128, type=int)
    parser.add_argument("--source_segment_partition_mode", default="doy_gap")
    parser.add_argument("--source_segment_count", default=5, type=int)
    parser.add_argument("--source_phase_gap_threshold", default=45, type=int)
    parser.add_argument("--source_phase_min_points", default=3, type=int)
    parser.add_argument("--source_phase_max_points", default=8, type=int)
    parser.add_argument("--source_phase_max_span", default=120, type=int)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    rows = []
    for source in args.sources:
        rows.append(compute_source_row(args, source))

    payload = {
        "sources": rows,
        "oracle_epoch_summary": build_oracle_epoch_summary(),
        "descriptor_epoch_alignment": summarize_descriptor_epoch_alignment(rows),
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2))


def bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def compute_source_row(args, source):
    classes = label_utils.get_classes(source.split("/")[0], combine_spring_and_winter=False)
    if args.closed_set:
        classes = [cls for cls in classes if cls != "unknown"]
    all_data = PixelSetData(args.data_root, source, classes, closed_set=args.closed_set)
    labels, counts = np.unique(all_data.get_labels(), return_counts=True)
    kept_classes = [classes[i] for i in labels[counts >= 200]]
    dataset = PixelSetData(args.data_root, source, kept_classes, closed_set=args.closed_set)

    partition_spec = build_source_segment_partition_spec(
        dataset.date_positions,
        dataset=dataset,
        mode=args.source_segment_partition_mode,
        segment_count=args.source_segment_count,
        gap_threshold=args.source_phase_gap_threshold,
        min_points=args.source_phase_min_points,
        max_points=args.source_phase_max_points,
        max_span=args.source_phase_max_span,
        semantic_max_samples_per_class=args.max_samples_per_class,
    )
    curves = collect_class_curves(dataset, args.max_samples_per_class)
    descriptors = compute_descriptors(dataset, partition_spec, curves)
    source_tile = source.split("/")[1]
    return {
        "source": source,
        "source_tile": source_tile,
        "num_classes": len(kept_classes),
        "num_samples": len(dataset),
        "classes": kept_classes,
        "partition": {
            "description": describe_source_segment_partition_spec(partition_spec),
            "mode": partition_spec.get("mode"),
            "segment_count": int(partition_spec.get("phase_count", partition_spec.get("segment_count", 0))),
            "intervals": partition_spec.get("intervals"),
        },
        "descriptors": descriptors,
        "oracle_targets": oracle_targets_for_source(source_tile),
    }


def collect_class_curves(dataset, max_samples_per_class):
    sums = {}
    sq_sums = {}
    counts = defaultdict(int)
    max_samples_per_class = max(1, int(max_samples_per_class))
    for path, _parcel_idx, label, _extra in dataset.samples:
        label = int(label)
        if counts[label] >= max_samples_per_class:
            continue
        pixels = zarr.load(path)
        if pixels.ndim != 3:
            continue
        curve = pixels.mean(axis=-1).astype(np.float64, copy=False)
        if label not in sums:
            sums[label] = curve.copy()
            sq_sums[label] = curve * curve
        else:
            sums[label] += curve
            sq_sums[label] += curve * curve
        counts[label] += 1

    curves = {}
    for label, summed in sums.items():
        count = max(counts[label], 1)
        mean = summed / float(count)
        second = sq_sums[label] / float(count)
        variance = np.maximum(second - mean * mean, 0.0)
        curves[label] = {
            "mean": mean,
            "variance": variance,
            "count": count,
        }
    return curves


def compute_descriptors(dataset, partition_spec, curves):
    positions = np.asarray(dataset.date_positions, dtype=np.float64)
    intervals = partition_spec.get("intervals")
    segment_count = int(partition_spec.get("phase_count", partition_spec.get("segment_count", 0)))
    if intervals:
        spans = np.asarray([end - start for start, end in intervals], dtype=np.float64)
        point_counts = np.asarray([
            np.sum((positions >= start) & (positions <= end)) for start, end in intervals
        ], dtype=np.float64)
    else:
        split_indices = np.array_split(np.arange(len(positions)), max(segment_count, 1))
        point_counts = np.asarray([len(item) for item in split_indices], dtype=np.float64)
        spans = np.asarray(
            [positions[item[-1]] - positions[item[0]] if len(item) else 0.0 for item in split_indices],
            dtype=np.float64,
        )

    class_means = [standardize_curve(item["mean"]) for item in curves.values()]
    class_variances = [item["variance"] for item in curves.values()]
    boundary_scores = []
    roughness_values = []
    for curve in class_means:
        diffs = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
        boundary_scores.append(diffs)
        roughness_values.append(float(np.mean(diffs)) if diffs.size else 0.0)

    mean_boundary_scores = np.mean(np.stack(boundary_scores, axis=0), axis=0) if boundary_scores else np.zeros(0)
    boundary_concentration = boundary_focus_ratio(mean_boundary_scores, positions, intervals)
    separability = class_separability(class_means)
    within_variability = float(np.mean([np.mean(var) for var in class_variances])) if class_variances else 0.0
    span_mean = safe_mean(spans)
    point_count_mean = safe_mean(point_counts)

    return {
        "segment_count": float(segment_count),
        "segment_span_mean": span_mean,
        "segment_point_count_mean": point_count_mean,
        "segment_point_count_cv": safe_cv(point_counts),
        "temporal_roughness_mean": float(np.mean(roughness_values)) if roughness_values else 0.0,
        "boundary_concentration": boundary_concentration,
        "class_curve_separability": separability,
        "within_class_variability": within_variability,
    }


def standardize_curve(curve, eps=1e-6):
    mean = curve.mean(axis=0, keepdims=True)
    std = curve.std(axis=0, keepdims=True)
    return (curve - mean) / np.clip(std, eps, None)


def boundary_focus_ratio(scores, positions, intervals):
    if scores.size == 0 or not intervals:
        return 0.0
    boundary_indices = []
    for _start, end in intervals[:-1]:
        idx = int(np.searchsorted(positions, end, side="right") - 1)
        if 0 <= idx < scores.size:
            boundary_indices.append(idx)
    if not boundary_indices:
        return 0.0
    boundary_mean = float(np.mean(scores[boundary_indices]))
    global_mean = float(np.mean(scores)) if scores.size else 0.0
    return boundary_mean / max(global_mean, 1e-8)


def class_separability(class_means):
    if len(class_means) < 2:
        return 0.0
    flattened = [curve.reshape(-1) for curve in class_means]
    distances = []
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            distances.append(float(np.linalg.norm(flattened[i] - flattened[j]) / math.sqrt(flattened[i].size)))
    return float(np.mean(distances)) if distances else 0.0


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else 0.0


def safe_cv(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.std(values) / max(np.mean(values), 1e-8))


def oracle_targets_for_source(source_tile):
    rows = []
    prefix = f"{source_tile}->"
    for task, by_epoch in sorted(ORACLE_F1_BY_TASK_EPOCH.items()):
        if not task.startswith(prefix):
            continue
        best_epoch, best_f1 = max(by_epoch.items(), key=lambda item: item[1])
        rows.append({
            "task": task,
            "best_epoch": int(best_epoch),
            "best_f1": float(best_f1),
            "epoch_f1": {str(epoch): float(value) for epoch, value in sorted(by_epoch.items())},
        })
    return rows


def build_oracle_epoch_summary():
    summary = defaultdict(lambda: defaultdict(int))
    for task, by_epoch in ORACLE_F1_BY_TASK_EPOCH.items():
        source = task.split("->")[0]
        best_epoch = max(by_epoch.items(), key=lambda item: item[1])[0]
        summary[source][int(best_epoch)] += 1
    return {
        source: {str(epoch): count for epoch, count in sorted(counts.items())}
        for source, counts in sorted(summary.items())
    }


def summarize_descriptor_epoch_alignment(rows):
    # With only four source domains this is descriptive only, not a statistical claim.
    result = []
    for row in rows:
        best_epochs = [item["best_epoch"] for item in row["oracle_targets"]]
        if not best_epochs:
            continue
        result.append({
            "source_tile": row["source_tile"],
            "mean_best_epoch": float(np.mean(best_epochs)),
            "min_best_epoch": int(np.min(best_epochs)),
            "max_best_epoch": int(np.max(best_epochs)),
            "descriptors": row["descriptors"],
        })
    return result


def write_json(path, payload):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_markdown(path, payload):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    lines = ["# v2.5.5a Source Structure Descriptor Audit", ""]
    lines.append("## Source Descriptors")
    lines.append("")
    lines.append("| source | samples | classes | segments | roughness | boundary concentration | separability | within var | oracle best epochs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["sources"]:
        desc = row["descriptors"]
        best_epochs = ", ".join(f"{item['task'].split('->')[1]}:ep{item['best_epoch']}" for item in row["oracle_targets"])
        lines.append(
            f"| `{row['source_tile']}` | {row['num_samples']} | {row['num_classes']} | "
            f"{desc['segment_count']:.0f} | {desc['temporal_roughness_mean']:.4f} | "
            f"{desc['boundary_concentration']:.4f} | {desc['class_curve_separability']:.4f} | "
            f"{desc['within_class_variability']:.4f} | {best_epochs} |"
        )
    lines.append("")
    lines.append("## Oracle Epoch Summary")
    lines.append("")
    lines.append("| source | best-epoch counts |")
    lines.append("|---|---|")
    for source, counts in payload["oracle_epoch_summary"].items():
        text = ", ".join(f"ep{epoch}: {count}" for epoch, count in counts.items())
        lines.append(f"| `{source}` | {text} |")
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("")
    lines.append("These descriptors are diagnostic only. Do not convert them into a training gate until they show a stable relationship with source checkpoint preference or downstream transfer behavior.")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
