import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


SUMMARY_LABELS = {"accuracy", "macro avg", "weighted avg"}


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize per-class correlations between structural metrics and target-class F1.")
    parser.add_argument("--metrics_csv", required=True, type=str)
    parser.add_argument("--outputs_root", default="outputs", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--min_group_size", default=2, type=int)
    return parser.parse_args()


def load_metrics_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_metric_fields(rows):
    if not rows:
        return []
    excluded = {
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
    }
    fields = []
    for field in rows[0].keys():
        if field in excluded:
            continue
        numeric_seen = False
        for row in rows[: min(len(rows), 5)]:
            parsed = safe_float(row.get(field))
            if parsed is not None:
                numeric_seen = True
                break
        if numeric_seen:
            fields.append(field)
    return fields


def parse_class_report(report_path):
    class_rows = []
    pattern = re.compile(
        r"^\s*(?P<label>[A-Za-z0-9_ ]+?)\s+"
        r"(?P<precision>\d+\.\d+)\s+"
        r"(?P<recall>\d+\.\d+)\s+"
        r"(?P<f1>\d+\.\d+)\s+"
        r"(?P<support>\d+)\s*$"
    )
    with open(report_path, encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.rstrip("\n"))
            if not match:
                continue
            label = match.group("label").strip()
            if label in SUMMARY_LABELS:
                continue
            class_rows.append(
                {
                    "class_name": label,
                    "precision": float(match.group("precision")),
                    "recall": float(match.group("recall")),
                    "f1": float(match.group("f1")),
                    "support": int(match.group("support")),
                }
            )
    return class_rows


def read_fold_class_reports(outputs_root, experiment_name, target_dataset):
    target_name = target_dataset.replace("/", "_")
    experiment_dir = Path(outputs_root) / experiment_name
    report_paths = sorted(experiment_dir.glob(f"fold_*/class_report_{target_name}.txt"))
    if not report_paths:
        raise FileNotFoundError(f"Missing class reports for {experiment_name} -> {target_name}")

    class_to_rows = defaultdict(list)
    for report_path in report_paths:
        for row in parse_class_report(report_path):
            class_to_rows[row["class_name"]].append(row)

    aggregated = []
    for class_name, rows in sorted(class_to_rows.items()):
        aggregated.append(
            {
                "class_name": class_name,
                "precision": sum(row["precision"] for row in rows) / len(rows),
                "recall": sum(row["recall"] for row in rows) / len(rows),
                "f1": sum(row["f1"] for row in rows) / len(rows),
                "support": sum(row["support"] for row in rows) / len(rows),
                "fold_count": len(rows),
            }
        )
    return aggregated


def build_long_rows(metrics_rows, outputs_root, metric_fields):
    long_rows = []
    for metrics_row in metrics_rows:
        class_rows = read_fold_class_reports(
            outputs_root=outputs_root,
            experiment_name=metrics_row["timematch_experiment"],
            target_dataset=metrics_row["target_dataset"],
        )
        for class_row in class_rows:
            row = {
                "source": metrics_row["source"],
                "target": metrics_row["target"],
                "source_dataset": metrics_row["source_dataset"],
                "target_dataset": metrics_row["target_dataset"],
                "timematch_experiment": metrics_row["timematch_experiment"],
                "class_name": class_row["class_name"],
                "class_f1": class_row["f1"],
                "class_precision": class_row["precision"],
                "class_recall": class_row["recall"],
                "class_support": class_row["support"],
                "class_fold_count": class_row["fold_count"],
                "target_f1": safe_float(metrics_row["target_f1"]),
            }
            for metric in metric_fields:
                row[metric] = safe_float(metrics_row.get(metric))
            long_rows.append(row)
    return long_rows


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
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    var_x = sum(x * x for x in centered_x)
    var_y = sum(y * y for y in centered_y)
    if var_x == 0 or var_y == 0:
        return float("nan"), len(pairs)
    cov = sum(x * y for x, y in zip(centered_x, centered_y))
    corr = cov / math.sqrt(var_x * var_y)
    return float(corr), len(pairs)


def build_summary(rows, metric_fields, target_field, group_name, group_value):
    summary_rows = []
    for metric_field in metric_fields:
        corr, pair_count = compute_corr(rows, metric_field, target_field)
        summary_rows.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "metric": metric_field,
                "corr_with_class_f1": corr,
                "abs_corr": abs(corr) if math.isfinite(corr) else float("nan"),
                "pair_count": pair_count,
                "row_count": len(rows),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            1 if not math.isfinite(row["corr_with_class_f1"]) else 0,
            0 if not math.isfinite(row["corr_with_class_f1"]) else -abs(row["corr_with_class_f1"]),
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
    metrics_rows = load_metrics_rows(args.metrics_csv)
    metric_fields = infer_metric_fields(metrics_rows)
    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)

    long_rows = build_long_rows(metrics_rows, outputs_root, metric_fields)
    write_csv(output_dir / "per_class_long_table.csv", long_rows)

    by_class = defaultdict(list)
    by_source = defaultdict(list)
    for row in long_rows:
        by_class[row["class_name"]].append(row)
        by_source[row["source"]].append(row)

    overall_rows = build_summary(long_rows, metric_fields, "class_f1", "overall", "all")
    write_csv(output_dir / "overall_metric_correlations.csv", overall_rows)

    class_rows = []
    for class_name, class_group in sorted(by_class.items()):
        if len(class_group) < args.min_group_size:
            continue
        class_rows.extend(build_summary(class_group, metric_fields, "class_f1", "class", class_name))
    write_csv(output_dir / "by_class_metric_correlations.csv", class_rows)

    source_rows = []
    for source, source_group in sorted(by_source.items()):
        if len(source_group) < args.min_group_size:
            continue
        source_rows.extend(build_summary(source_group, metric_fields, "class_f1", "source", source))
    write_csv(output_dir / "by_source_metric_correlations.csv", source_rows)

    class_metric_map = defaultdict(dict)
    for row in class_rows:
        corr = row["corr_with_class_f1"]
        if math.isfinite(corr):
            class_metric_map[row["metric"]][row["group_value"]] = corr

    spread_rows = []
    for metric, corr_map in class_metric_map.items():
        if len(corr_map) < 2:
            continue
        values = list(corr_map.values())
        spread_rows.append(
            {
                "metric": metric,
                "max_class_corr": max(values),
                "min_class_corr": min(values),
                "class_corr_spread": max(values) - min(values),
                "class_count": len(values),
            }
        )
    spread_rows.sort(key=lambda row: -row["class_corr_spread"])
    write_csv(output_dir / "class_correlation_spread.csv", spread_rows)

    print(f"[INFO] Saved per-class correlation summaries under: {output_dir}")


if __name__ == "__main__":
    main()
