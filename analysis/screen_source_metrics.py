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


def is_source_metric(metric_name):
    if "_gap" in metric_name:
        return False
    if metric_name.startswith("target_") or metric_name.startswith("raw_target_"):
        return False
    return metric_name.startswith("source_") or metric_name.startswith("raw_source_")


def infer_family(metric_name):
    if metric_name.startswith("raw_source_phase_"):
        return "raw-phase"
    if metric_name.startswith("source_phase_"):
        return "encoded-phase"
    if metric_name.startswith("raw_source_"):
        return "raw-shape"
    if metric_name.startswith("source_"):
        return "encoded-discriminability"
    return "other"


def load_overall_map(path):
    rows = read_csv(path)
    result = {}
    for row in rows:
        metric = row.get("metric")
        if not metric or not is_source_metric(metric):
            continue
        result[metric] = safe_float(row.get("corr_with_target_f1")) or safe_float(row.get("corr_with_class_f1")) or safe_float(row.get("correlation"))
    return result


def load_spread_map(path):
    rows = read_csv(path)
    result = {}
    for row in rows:
        metric = row.get("metric")
        if not metric or not is_source_metric(metric):
            continue
        result[metric] = safe_float(row.get("source_corr_spread")) or safe_float(row.get("class_corr_spread")) or safe_float(row.get("spread"))
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


def write_markdown(path, ranked_rows, family_rows):
    lines = [
        "# Source Metric Screening",
        "",
        "This report screens source-side metrics only.",
        "Target-only and gap-style indicators are intentionally excluded.",
        "",
        "## Top ranked source metrics",
        "",
        "| metric | family | transfer_corr | phase_corr | per_class_corr | source_spread | class_spread | combined_score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked_rows[:12]:
        lines.append(
            f"| {row['metric']} | {row['family']} | {row['transfer_overall_corr']:.4f} | "
            f"{row['phase_overall_corr']:.4f} | {row['per_class_overall_corr']:.4f} | "
            f"{row['source_spread']:.4f} | {row['class_spread']:.4f} | {row['combined_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Family summary",
            "",
            "| family | top_metric | top_score | avg_score | metric_count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['top_metric']} | {row['top_score']:.4f} | "
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

    transfer_overall = load_overall_map(args.transfer_overall_csv)
    transfer_spread = load_spread_map(args.transfer_source_spread_csv)
    phase_overall = load_overall_map(args.phase_overall_csv)
    phase_spread = load_spread_map(args.phase_source_spread_csv)
    per_class_overall = load_overall_map(args.per_class_overall_csv)
    per_class_spread = load_spread_map(args.per_class_spread_csv)

    all_metrics = sorted(
        set(transfer_overall) | set(transfer_spread) | set(phase_overall) | set(phase_spread) | set(per_class_overall) | set(per_class_spread)
    )

    ranked_rows = []
    family_buckets = defaultdict(list)
    for metric in all_metrics:
        row = {
            "metric": metric,
            "family": infer_family(metric),
            "transfer_overall_corr": transfer_overall.get(metric) or 0.0,
            "phase_overall_corr": phase_overall.get(metric) or 0.0,
            "per_class_overall_corr": per_class_overall.get(metric) or 0.0,
            "source_spread": max(
                transfer_spread.get(metric) or 0.0,
                phase_spread.get(metric) or 0.0,
            ),
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
        family_buckets[row["family"]].append(row)

    ranked_rows.sort(key=lambda x: x["combined_score"], reverse=True)

    family_rows = []
    for family, rows in sorted(family_buckets.items()):
        rows = sorted(rows, key=lambda x: x["combined_score"], reverse=True)
        avg_score = sum(r["combined_score"] for r in rows) / len(rows)
        family_rows.append(
            {
                "family": family,
                "top_metric": rows[0]["metric"],
                "top_score": rows[0]["combined_score"],
                "avg_score": avg_score,
                "metric_count": len(rows),
            }
        )
    family_rows.sort(key=lambda x: x["avg_score"], reverse=True)

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(
        os.path.join(args.output_dir, "ranked_source_metrics.csv"),
        ranked_rows,
        [
            "metric",
            "family",
            "transfer_overall_corr",
            "phase_overall_corr",
            "per_class_overall_corr",
            "source_spread",
            "class_spread",
            "combined_score",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "source_metric_family_summary.csv"),
        family_rows,
        ["family", "top_metric", "top_score", "avg_score", "metric_count"],
    )
    write_markdown(
        os.path.join(args.output_dir, "source_metric_screening.md"),
        ranked_rows,
        family_rows,
    )


if __name__ == "__main__":
    main()
