import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


REMOTE_BEST = {
    ("remote", "FR1->FR2"): {
        "baseline": 0.7946,
        "best": 0.8101,
        "best_view": "global",
        "best_strength": "strong",
        "structure_view": "global_compact",
        "segment": "noseg",
        "intra": 5.0,
        "trend": 0.0,
        "segment_inter": 0.0,
        "boundary": 0.0,
        "dynamics": 0.0,
    },
    ("remote", "FR1->DK1"): {
        "baseline": 0.6581,
        "best": 0.6977,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.02,
        "segment_inter": 0.02,
        "boundary": 0.2,
        "dynamics": 0.0,
    },
    ("remote", "FR1->AT1"): {
        "baseline": 0.8136,
        "best": 0.8211,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.03,
        "segment_inter": 0.01,
        "boundary": 0.1,
        "dynamics": 0.0,
    },
    ("remote", "FR2->FR1"): {
        "baseline": 0.6513,
        "best": 0.7957,
        "best_view": "dynamics",
        "best_strength": "light",
        "structure_view": "global_trajectory",
        "segment": "noseg",
        "intra": 1.0,
        "trend": 0.0,
        "segment_inter": 0.0,
        "boundary": 0.0,
        "dynamics": 0.01,
    },
    ("remote", "FR2->DK1"): {
        "baseline": 0.5156,
        "best": 0.6862,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.05,
        "segment_inter": 0.02,
        "boundary": 0.2,
        "dynamics": 0.0,
    },
    ("remote", "FR2->AT1"): {
        "baseline": 0.6545,
        "best": 0.6872,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.05,
        "segment_inter": 0.02,
        "boundary": 0.2,
        "dynamics": 0.0,
    },
    ("remote", "DK1->FR1"): {
        "baseline": 0.5683,
        "best": 0.6359,
        "best_view": "dynamics",
        "best_strength": "light",
        "structure_view": "pointwise_dynamics",
        "segment": "noseg",
        "intra": 1.0,
        "trend": 0.0,
        "segment_inter": 0.0,
        "boundary": 0.0,
        "dynamics": 0.05,
    },
    ("remote", "DK1->FR2"): {
        "baseline": 0.4531,
        "best": 0.5090,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.05,
        "segment_inter": 0.02,
        "boundary": 0.2,
        "dynamics": 0.0,
    },
    ("remote", "DK1->AT1"): {
        "baseline": 0.6703,
        "best": 0.6816,
        "best_view": "segmented",
        "best_strength": "light",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.05,
        "segment_inter": 0.005,
        "boundary": 0.0,
        "dynamics": 0.0,
    },
    ("remote", "AT1->FR1"): {
        "baseline": 0.7143,
        "best": 0.7230,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.05,
        "segment_inter": 0.02,
        "boundary": 0.2,
        "dynamics": 0.0,
    },
    ("remote", "AT1->FR2"): {
        "baseline": 0.6214,
        "best": 0.6323,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.02,
        "segment_inter": 0.02,
        "boundary": 0.2,
        "dynamics": 0.0,
    },
    ("remote", "AT1->DK1"): {
        "baseline": 0.7483,
        "best": 0.8623,
        "best_view": "segmented",
        "best_strength": "medium",
        "structure_view": "segmented_compact",
        "segment": "segmented",
        "intra": 1.0,
        "trend": 0.05,
        "segment_inter": 0.02,
        "boundary": 0.2,
        "dynamics": 0.0,
    },
}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=4):
    if value is None:
        return ""
    value = to_float(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def dominant_view(text):
    text = str(text)
    if "dynamics" in text:
        return "dynamics"
    if "noseg" in text or "global" in text:
        return "global"
    return "segmented"


def strength_from_har_view(text):
    text = str(text)
    if "intra_light" in text or "boundary_light" in text:
        return "light"
    if "compact_k8" in text:
        return "medium_high"
    return "medium"


def load_har_best(path):
    best = {}
    for row in read_csv(path):
        key = (row["dataset"], row["task"])
        views = row.get("da_best_views", "")
        baseline = to_float(row.get("da_baseline"))
        result = to_float(row.get("da_best"))
        best[key] = {
            "baseline": baseline,
            "best": result,
            "best_view": dominant_view(views),
            "best_strength": strength_from_har_view(views),
            "structure_view": views,
            "segment": "noseg" if "noseg" in views else "segmented",
            "intra": "",
            "trend": "",
            "segment_inter": "",
            "boundary": "",
            "dynamics": "yes" if "dynamics" in views else "",
        }
    return best


def load_pse_audit(paths):
    rows = {}
    for path in paths:
        for row in read_csv(path):
            rows[(row["dataset"], row["task"])] = row
    return rows


def strength_rank(label):
    return {
        "light": 0.25,
        "medium": 0.5,
        "medium_high": 0.75,
        "strong": 1.0,
    }.get(str(label), math.nan)


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 3:
        return math.nan
    xs, ys = zip(*pairs)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def group_means(rows, group_key, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    output = {}
    for group, items in grouped.items():
        output[group] = {metric: mean([to_float(item.get(metric)) for item in items]) for metric in metrics}
        output[group]["count"] = len(items)
    return output


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, rows, correlations, by_view, by_strength):
    lines = [
        "# v2.6.0b Metric-Best Alignment",
        "",
        "## Conclusion",
        "",
        "The PSE feature-space audit is useful for strength adaptation, but the current view heuristic is not reliable enough for automatic view selection.",
        "",
        "- `suggested_view` collapses toward `global`, while empirical best views are mostly segmented and partly dynamics.",
        "- `source_reliability`, `target_margin_ratio`, and `temporal_mismatch_cv` are usable safety signals for conservative strength adaptation, but none of them is a standalone strength predictor.",
        "- `uniform_k*_adjusted_compression` is better interpreted as a future window/locality signal, not as a global strength multiplier.",
        "- `dynamics_delta_cosine_mean` is not sufficient to select dynamics by itself.",
        "",
        "## Best-View Distribution",
        "",
        "| source | global | segmented | dynamics |",
        "| --- | ---: | ---: | ---: |",
    ]
    total_counter = Counter(row["best_view"] for row in rows)
    lines.append(f"| all | {total_counter['global']} | {total_counter['segmented']} | {total_counter['dynamics']} |")
    for dataset in sorted(set(row["dataset"] for row in rows)):
        counter = Counter(row["best_view"] for row in rows if row["dataset"] == dataset)
        lines.append(f"| {dataset} | {counter['global']} | {counter['segmented']} | {counter['dynamics']} |")

    lines += [
        "",
        "## Metric Correlation With Best Strength",
        "",
        "Strength is encoded as `light=0.25`, `medium=0.5`, `medium_high=0.75`, `strong=1.0`. This is only a coarse ordinal target.",
        "",
        "| metric | Pearson r | reading |",
        "| --- | ---: | --- |",
    ]
    readings = {
        "source_reliability": "higher source reliability can support stronger compactness, but dataset effects are strong",
        "target_margin_ratio": "useful as a safety threshold, but not a linear standalone strength predictor",
        "temporal_mismatch_cv": "high locality/mismatch variation argues against globally stronger stiffness",
        "uniform_k5_adjusted_compression": "locality/window signal, not a direct strength signal",
        "uniform_k8_adjusted_compression": "locality/window signal, useful for v2.6.2",
        "dynamics_delta_cosine_mean": "not reliable as a dynamics selector",
    }
    for metric, value in correlations.items():
        lines.append(f"| `{metric}` | {fmt(value)} | {readings.get(metric, '')} |")

    lines += [
        "",
        "## Group Means By Empirical Best View",
        "",
        "| best view | n | src_rel | tgt_margin | mismatch_cv | k5_comp | k8_comp | dyn_cos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in ["global", "segmented", "dynamics"]:
        item = by_view.get(group, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(item.get("count", 0)),
                    fmt(item.get("source_reliability")),
                    fmt(item.get("target_margin_ratio")),
                    fmt(item.get("temporal_mismatch_cv")),
                    fmt(item.get("uniform_k5_adjusted_compression")),
                    fmt(item.get("uniform_k8_adjusted_compression")),
                    fmt(item.get("dynamics_delta_cosine_mean")),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Group Means By Empirical Best Strength",
        "",
        "| strength | n | src_rel | tgt_margin | mismatch_cv | k5_comp | factor_old |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in ["light", "medium", "medium_high", "strong"]:
        item = by_strength.get(group, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(item.get("count", 0)),
                    fmt(item.get("source_reliability")),
                    fmt(item.get("target_margin_ratio")),
                    fmt(item.get("temporal_mismatch_cv")),
                    fmt(item.get("uniform_k5_adjusted_compression")),
                    fmt(item.get("suggested_strength_factor")),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Task Alignment Table",
        "",
        "| dataset | task | best view | strength | gain | src_rel | tgt_margin | mismatch_cv | k5_comp | k8_comp | dyn_cos | old suggested | old factor |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (r["dataset"], r["task"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["task"],
                    row["best_view"],
                    row["best_strength"],
                    fmt(row["best_gain"]),
                    fmt(row["source_reliability"]),
                    fmt(row["target_margin_ratio"]),
                    fmt(row["temporal_mismatch_cv"]),
                    fmt(row["uniform_k5_adjusted_compression"]),
                    fmt(row["uniform_k8_adjusted_compression"]),
                    fmt(row["dynamics_delta_cosine_mean"]),
                    row["suggested_view"],
                    fmt(row["suggested_strength_factor"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## v2.6 Decision",
        "",
        "Proceed to v2.6.1, but only after adopting this narrower definition:",
        "",
        "> v2.6.1 should adapt the strength of intra compactness first, using target-aware feature-geometry signals. It should not adapt view/window/component weights yet.",
        "",
        "Recommended first factor inputs:",
        "",
        "- `source_reliability`",
        "- `target_margin_ratio` as a safety threshold",
        "- `temporal_mismatch_cv`",
        "",
        "Use `uniform_k*_adjusted_compression` later for v2.6.2 window adaptation.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="result/_summary/v260/v260_metric_best_alignment")
    parser.add_argument(
        "--pse_csv",
        action="append",
        default=[
            "result/_summary/v260/v260_structure_reliability_audit_20260521_105553/v260_structure_reliability_audit_full.csv",
            "result/_summary/v260/v260_structure_reliability_audit_20260521_111004/v260_structure_reliability_audit_feature_failed_20260521.csv",
            "result/_summary/v260/v260_structure_reliability_audit_20260521_111727/v260_structure_reliability_audit_har_2_11.csv",
        ],
    )
    parser.add_argument(
        "--har_best_csv",
        default="result/_summary/v25_har_hhar_multiview_existing/har_hhar_multiview_task_summary.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = load_pse_audit(args.pse_csv)
    best = dict(REMOTE_BEST)
    best.update(load_har_best(args.har_best_csv))

    rows = []
    for key, best_row in sorted(best.items()):
        audit_row = audit.get(key)
        if not audit_row:
            continue
        row = {
            "dataset": key[0],
            "task": key[1],
            "baseline": best_row["baseline"],
            "best": best_row["best"],
            "best_gain": best_row["best"] - best_row["baseline"],
            "best_view": best_row["best_view"],
            "best_strength": best_row["best_strength"],
            "best_strength_rank": strength_rank(best_row["best_strength"]),
            "structure_view": best_row["structure_view"],
            "segment": best_row["segment"],
            "intra": best_row["intra"],
            "trend": best_row["trend"],
            "segment_inter": best_row["segment_inter"],
            "boundary": best_row["boundary"],
            "dynamics": best_row["dynamics"],
        }
        for name in [
            "source_reliability",
            "source_compactness",
            "source_separability",
            "target_margin_ratio",
            "target_assignment_entropy",
            "temporal_mismatch_cv",
            "temporal_mismatch_top20_ratio",
            "uniform_k3_adjusted_compression",
            "uniform_k5_adjusted_compression",
            "uniform_k8_adjusted_compression",
            "dynamics_delta_cosine_mean",
            "suggested_view",
            "suggested_strength_factor",
        ]:
            row[name] = audit_row.get(name, "")
        rows.append(row)

    metrics = [
        "source_reliability",
        "target_margin_ratio",
        "temporal_mismatch_cv",
        "uniform_k5_adjusted_compression",
        "uniform_k8_adjusted_compression",
        "dynamics_delta_cosine_mean",
    ]
    correlations = {
        metric: pearson(
            [to_float(row.get(metric)) for row in rows],
            [to_float(row.get("best_strength_rank")) for row in rows],
        )
        for metric in metrics
    }
    by_view = group_means(rows, "best_view", metrics + ["suggested_strength_factor"])
    by_strength = group_means(rows, "best_strength", metrics + ["suggested_strength_factor"])

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(out_dir / "v260_metric_best_alignment.csv", rows, fieldnames)
    write_md(out_dir / "v260_metric_best_alignment.md", rows, correlations, by_view, by_strength)
    print(f"Wrote {len(rows)} aligned rows to {out_dir}")


if __name__ == "__main__":
    main()
