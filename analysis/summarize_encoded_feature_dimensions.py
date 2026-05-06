import argparse
import csv
import os
from collections import defaultdict


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_encoded_source_metric(metric_name):
    if "_gap" in metric_name:
        return False
    if metric_name.startswith("raw_") or metric_name.startswith("target_") or metric_name.startswith("raw_target_"):
        return False
    return metric_name.startswith("source_")


def infer_dimension(metric_name):
    if metric_name.startswith("source_phase_compactness_"):
        return "phase-intra-compactness"
    if metric_name.startswith("source_phase_separability_") or metric_name.startswith("source_phase_margin_"):
        return "phase-inter-contrast"
    if metric_name in {
        "source_curve_spread",
        "source_curve_activity_range",
        "source_ndvi_q90_range_mean",
        "source_ndvi_q90_range_std",
        "source_class_curve_variance_mean",
    }:
        return "global-feature-structure"
    if metric_name == "source_fisher_ratio":
        return "effect-discriminability"
    return "other-encoded"


def load_corr_map(path):
    rows = read_csv(path)
    result = {}
    for row in rows:
        metric = row.get("metric")
        if not metric or not is_encoded_source_metric(metric):
            continue
        result[metric] = (
            safe_float(row.get("corr_with_target_f1"))
            or safe_float(row.get("corr_with_class_f1"))
            or safe_float(row.get("correlation"))
        )
    return result


def load_spread_map(path):
    rows = read_csv(path)
    result = {}
    for row in rows:
        metric = row.get("metric")
        if not metric or not is_encoded_source_metric(metric):
            continue
        result[metric] = (
            safe_float(row.get("source_corr_spread"))
            or safe_float(row.get("class_corr_spread"))
            or safe_float(row.get("spread"))
        )
    return result


def combined_score(transfer_corr, phase_corr, per_class_corr, source_spread, class_spread):
    score = 0.0
    if transfer_corr is not None:
        score += 0.5 * abs(transfer_corr)
    if phase_corr is not None:
        score += 0.3 * abs(phase_corr)
    if per_class_corr is not None:
        score += 0.2 * abs(per_class_corr)
    source_penalty = 1.0 / (1.0 + (source_spread or 0.0))
    class_penalty = 1.0 / (1.0 + (class_spread or 0.0))
    return score * source_penalty * class_penalty


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, ranked_rows, dimension_rows):
    lines = [
        "# Encoded Feature-Curve Dimension Summary",
        "",
        "This report keeps only encoded source feature-curve metrics.",
        "Raw-source metrics, target-only metrics, and gap metrics are excluded.",
        "",
        "## Ranked encoded metrics",
        "",
        "| metric | dimension | transfer_corr | phase_corr | per_class_corr | source_spread | class_spread | combined_score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked_rows:
        lines.append(
            f"| {row['metric']} | {row['dimension']} | {row['transfer_overall_corr']:.4f} | "
            f"{row['phase_overall_corr']:.4f} | {row['per_class_overall_corr']:.4f} | "
            f"{row['source_spread']:.4f} | {row['class_spread']:.4f} | {row['combined_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Dimension summary",
            "",
            "| dimension | top_metric | top_score | avg_score | metric_count |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in dimension_rows:
        lines.append(
            f"| {row['dimension']} | {row['top_metric']} | {row['top_score']:.4f} | "
            f"{row['avg_score']:.4f} | {row['metric_count']} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer_overall_csv", required=True)
    parser.add_argument("--transfer_source_spread_csv", required=True)
    parser.add_argument("--phase_overall_csv", required=True)
    parser.add_argument("--phase_source_spread_csv", required=True)
    parser.add_argument("--per_class_overall_csv", required=True)
    parser.add_argument("--per_class_spread_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    transfer_overall = load_corr_map(args.transfer_overall_csv)
    transfer_spread = load_spread_map(args.transfer_source_spread_csv)
    phase_overall = load_corr_map(args.phase_overall_csv)
    phase_spread = load_spread_map(args.phase_source_spread_csv)
    per_class_overall = load_corr_map(args.per_class_overall_csv)
    per_class_spread = load_spread_map(args.per_class_spread_csv)

    all_metrics = sorted(
        set(transfer_overall)
        | set(transfer_spread)
        | set(phase_overall)
        | set(phase_spread)
        | set(per_class_overall)
        | set(per_class_spread)
    )

    ranked_rows = []
    dimension_buckets = defaultdict(list)
    for metric in all_metrics:
        row = {
            "metric": metric,
            "dimension": infer_dimension(metric),
            "transfer_overall_corr": transfer_overall.get(metric) or 0.0,
            "phase_overall_corr": phase_overall.get(metric) or 0.0,
            "per_class_overall_corr": per_class_overall.get(metric) or 0.0,
            "source_spread": max(transfer_spread.get(metric) or 0.0, phase_spread.get(metric) or 0.0),
            "class_spread": per_class_spread.get(metric) or 0.0,
        }
        row["combined_score"] = combined_score(
            transfer_overall.get(metric),
            phase_overall.get(metric),
            per_class_overall.get(metric),
            row["source_spread"],
            row["class_spread"],
        )
        ranked_rows.append(row)
        dimension_buckets[row["dimension"]].append(row)

    ranked_rows.sort(key=lambda x: x["combined_score"], reverse=True)

    dimension_rows = []
    for dimension, rows in sorted(dimension_buckets.items()):
        rows = sorted(rows, key=lambda x: x["combined_score"], reverse=True)
        avg_score = sum(r["combined_score"] for r in rows) / len(rows)
        dimension_rows.append(
            {
                "dimension": dimension,
                "top_metric": rows[0]["metric"],
                "top_score": rows[0]["combined_score"],
                "avg_score": avg_score,
                "metric_count": len(rows),
            }
        )
    dimension_rows.sort(key=lambda x: x["avg_score"], reverse=True)

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(
        os.path.join(args.output_dir, "ranked_encoded_feature_dimensions.csv"),
        ranked_rows,
        [
            "metric",
            "dimension",
            "transfer_overall_corr",
            "phase_overall_corr",
            "per_class_overall_corr",
            "source_spread",
            "class_spread",
            "combined_score",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "encoded_dimension_family_summary.csv"),
        dimension_rows,
        ["dimension", "top_metric", "top_score", "avg_score", "metric_count"],
    )
    write_markdown(
        os.path.join(args.output_dir, "encoded_feature_dimension_summary.md"),
        ranked_rows,
        dimension_rows,
    )


if __name__ == "__main__":
    main()
