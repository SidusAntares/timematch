import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


NON_METRIC_FIELDS = {
    "source",
    "target",
    "source_dataset",
    "target_dataset",
    "closed_set",
    "num_classes",
    "fold_count",
    "timematch_experiment",
    "target_f1",
    "target_accuracy",
    "shared_classes_for_proto",
    "shared_classes_for_relation",
    "shared_classes_for_curves",
    "shared_classes_for_curve_structure",
    "source_samples",
    "target_samples",
    "checkpoint_dir",
    "source_phase_partition_mode",
    "source_phase_count",
    "target_phase_partition_mode",
    "target_phase_count",
    "raw_source_phase_partition_mode",
    "raw_source_phase_count",
    "raw_target_phase_partition_mode",
    "raw_target_phase_count",
}


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize metric correlations overall and by source/target.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--target_field", default="target_f1", type=str)
    parser.add_argument("--min_group_size", default=2, type=int)
    return parser.parse_args()


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_metric_fields(rows, target_field):
    if not rows:
        return []
    metric_fields = []
    for field in rows[0].keys():
        if field == target_field or field in NON_METRIC_FIELDS:
            continue
        sample_values = [row.get(field) for row in rows[: min(len(rows), 5)]]
        numeric_seen = False
        valid = True
        for value in sample_values:
            parsed = safe_float(value)
            if parsed is None:
                continue
            numeric_seen = True
        if not numeric_seen:
            valid = False
        if not valid:
            continue
        metric_fields.append(field)
    return metric_fields


def compute_corr(rows, metric_field, target_field):
    pairs = []
    for row in rows:
        x = safe_float(row.get(metric_field))
        y = safe_float(row.get(target_field))
        if x is None or y is None:
            continue
        pairs.append((x, y))
    if len(pairs) < 2:
        return float("nan"), len(pairs)
    xs = np.array([x for x, _ in pairs], dtype=np.float64)
    ys = np.array([y for _, y in pairs], dtype=np.float64)
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan"), len(pairs)
    return float(np.corrcoef(xs, ys)[0, 1]), len(pairs)


def build_summary(rows, metric_fields, target_field, group_name, group_value):
    summary_rows = []
    for metric_field in metric_fields:
        corr, pair_count = compute_corr(rows, metric_field, target_field)
        summary_rows.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "metric": metric_field,
                "corr_with_target_f1": corr,
                "abs_corr": abs(corr) if np.isfinite(corr) else float("nan"),
                "pair_count": pair_count,
                "task_count": len(rows),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            1 if not np.isfinite(row["corr_with_target_f1"]) else 0,
            0 if not np.isfinite(row["corr_with_target_f1"]) else -abs(row["corr_with_target_f1"]),
            row["metric"],
        )
    )
    return summary_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    rows = load_rows(input_path)
    metric_fields = infer_metric_fields(rows, args.target_field)

    overall_rows = build_summary(rows, metric_fields, args.target_field, "overall", "all")
    write_csv(output_dir / "overall_metric_correlations.csv", overall_rows)

    by_source = defaultdict(list)
    by_target = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
        by_target[row["target"]].append(row)

    source_rows = []
    for source, source_group in sorted(by_source.items()):
        if len(source_group) < args.min_group_size:
            continue
        source_rows.extend(build_summary(source_group, metric_fields, args.target_field, "source", source))
    write_csv(output_dir / "by_source_metric_correlations.csv", source_rows)

    target_rows = []
    for target, target_group in sorted(by_target.items()):
        if len(target_group) < args.min_group_size:
            continue
        target_rows.extend(build_summary(target_group, metric_fields, args.target_field, "target", target))
    write_csv(output_dir / "by_target_metric_correlations.csv", target_rows)

    spread_rows = []
    source_metric_map = defaultdict(dict)
    for row in source_rows:
        if np.isfinite(row["corr_with_target_f1"]):
            source_metric_map[row["metric"]][row["group_value"]] = row["corr_with_target_f1"]
    for metric, corr_map in source_metric_map.items():
        if len(corr_map) < 2:
            continue
        values = list(corr_map.values())
        spread_rows.append(
            {
                "metric": metric,
                "max_source_corr": max(values),
                "min_source_corr": min(values),
                "source_corr_spread": max(values) - min(values),
                "source_count": len(values),
            }
        )
    spread_rows.sort(key=lambda row: -row["source_corr_spread"])
    write_csv(output_dir / "source_correlation_spread.csv", spread_rows)

    print(f"[INFO] Saved correlation summaries under: {output_dir}")


if __name__ == "__main__":
    main()
