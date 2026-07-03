#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SMOOTH_CONFIGS = {"smooth_k3", "v276_smooth_k3_w1"}
PLAIN_CONFIGS = {"plain"}


ANCHORABILITY_RULES = [
    ("cluster_compactness", "lower"),
    ("silhouette_score_sampled", "higher"),
    ("assignment_entropy_mean", "lower"),
    ("max_assignment_prob_mean", "higher"),
    ("temporal_persistence", "higher"),
    ("class_conditional_anchor_entropy", "lower"),
    ("class_conditional_anchor_purity", "higher"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize task-level relation among smooth DA gain, anchorability, and anchor correspondence gain."
    )
    parser.add_argument(
        "--anchor_dir",
        default="logs/v301_anchor_correspondence_diagnostic_20260701_100650",
        help="v3.0.1 anchor diagnostic log directory.",
    )
    parser.add_argument(
        "--da_rows",
        default="logs/v281_a_group_full_20260622_121128/full12/raw_strength_rows.tsv",
        help="raw_strength_rows.tsv containing plain and smooth_k3 DA F1.",
    )
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    return float(text)


def mean(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def canonical_config(config):
    if config in SMOOTH_CONFIGS:
        return "smooth_k3"
    if config in PLAIN_CONFIGS:
        return "plain"
    return config


def paired_delta(rows, value_field):
    by_seed = defaultdict(dict)
    for row in rows:
        cfg = canonical_config(row.get("checkpoint_config") or row.get("config"))
        if cfg in {"plain", "smooth_k3"}:
            by_seed[str(row["seed"])][cfg] = row
    deltas = []
    for seed, pair in by_seed.items():
        if "plain" not in pair or "smooth_k3" not in pair:
            continue
        plain = safe_float(pair["plain"].get(value_field))
        smooth = safe_float(pair["smooth_k3"].get(value_field))
        if plain is None or smooth is None:
            continue
        deltas.append(smooth - plain)
    return mean(deltas), len(deltas)


def direction(value):
    value = safe_float(value)
    if value is None:
        return "missing"
    if value > 1e-9:
        return "up"
    if value < -1e-9:
        return "down"
    return "flat"


def relation_label(da_delta, anchorability_score, anchor_gain_delta):
    dirs = [direction(da_delta), direction(anchorability_score), direction(anchor_gain_delta)]
    if "missing" in dirs:
        return "missing"
    if dirs[0] == dirs[1] == dirs[2] and dirs[0] != "flat":
        return "same_direction"
    if dirs[0] == "up" and dirs[1] == "up" and dirs[2] == "up":
        return "same_positive"
    return "not_same"


def load_da_rows(path):
    rows = []
    for row in read_tsv(path):
        cfg = canonical_config(row.get("config", ""))
        if cfg not in {"plain", "smooth_k3"}:
            continue
        if row.get("status", "ok") not in {"", "ok"}:
            continue
        rows.append({**row, "checkpoint_config": cfg})
    return rows


def summarize(anchor_dir, da_rows_path, output_dir):
    anchor_dir = Path(anchor_dir)
    output_dir = Path(output_dir) if output_dir else anchor_dir
    anchorability = read_tsv(anchor_dir / "anchorability_summary.tsv")
    nonshift = read_tsv(anchor_dir / "non_shiftness_summary.tsv")
    da_rows = load_da_rows(da_rows_path)

    tasks = sorted(
        set(row["task"] for row in anchorability)
        | set(row["task"] for row in nonshift)
        | set(row["task"] for row in da_rows)
    )
    out_rows = []
    for task in tasks:
        task_da = [row for row in da_rows if row["task"] == task]
        task_anchor = [row for row in anchorability if row["task"] == task]
        task_non = [row for row in nonshift if row["task"] == task]

        da_delta, da_n = paired_delta(task_da, "da_f1")
        source_target_delta, _ = paired_delta(task_da, "source_on_target_f1")
        anchor_gain_delta, corr_n = paired_delta(task_non, "anchor_gain_vs_best_global_shift")
        after_shift_gain_delta, _ = paired_delta(task_non, "anchor_after_shift_gain_vs_best_global_shift")

        metric_deltas = {}
        favorable = 0
        unfavorable = 0
        valid_metric_count = 0
        for metric, good_direction in ANCHORABILITY_RULES:
            delta, _ = paired_delta(task_anchor, metric)
            metric_deltas[f"{metric}_delta"] = delta
            if delta is None:
                continue
            valid_metric_count += 1
            improved = delta < 0 if good_direction == "lower" else delta > 0
            if abs(delta) <= 1e-9:
                continue
            if improved:
                favorable += 1
            else:
                unfavorable += 1
        anchorability_score = favorable - unfavorable
        row = {
            "task": task,
            "paired_seed_count_da": da_n,
            "paired_seed_count_anchor": corr_n,
            "smooth_minus_plain_da_f1": da_delta,
            "smooth_minus_plain_source_on_target_f1": source_target_delta,
            "anchorability_favorable_metrics": favorable,
            "anchorability_unfavorable_metrics": unfavorable,
            "anchorability_valid_metrics": valid_metric_count,
            "anchorability_score": anchorability_score,
            "smooth_minus_plain_anchor_gain_vs_best_global": anchor_gain_delta,
            "smooth_minus_plain_anchor_after_shift_gain_vs_best_global": after_shift_gain_delta,
            "da_direction": direction(da_delta),
            "anchorability_direction": direction(anchorability_score),
            "anchor_gain_direction": direction(anchor_gain_delta),
            "three_way_relation": relation_label(da_delta, anchorability_score, anchor_gain_delta),
            **metric_deltas,
        }
        out_rows.append(row)

    fields = [
        "task",
        "paired_seed_count_da",
        "paired_seed_count_anchor",
        "smooth_minus_plain_da_f1",
        "smooth_minus_plain_source_on_target_f1",
        "anchorability_favorable_metrics",
        "anchorability_unfavorable_metrics",
        "anchorability_valid_metrics",
        "anchorability_score",
        "smooth_minus_plain_anchor_gain_vs_best_global",
        "smooth_minus_plain_anchor_after_shift_gain_vs_best_global",
        "da_direction",
        "anchorability_direction",
        "anchor_gain_direction",
        "three_way_relation",
    ] + [f"{metric}_delta" for metric, _ in ANCHORABILITY_RULES]
    write_tsv(output_dir / "smooth_plain_anchor_relation_summary.tsv", out_rows, fields)
    write_markdown(output_dir / "smooth_plain_anchor_relation_summary.md", out_rows)
    return out_rows


def write_markdown(path, rows):
    lines = []
    lines.append("# Smooth vs Plain: DA, Anchorability, Anchor Correspondence Relation\n\n")
    lines.append("口径：所有变化均为 `smooth_k3 - plain`。DA 使用 `da_f1`；锚点对应收益使用 `anchor_gain_vs_best_global_shift`；可锚定性用 7 个方向明确的指标投票。\n\n")
    lines.append("## 1. Task-level Summary\n\n")
    lines.append("| task | ΔDA F1 | Δsource-on-target | anchorability score | Δanchor gain vs global | Δanchor-after-shift gain | directions | relation |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|\n")
    for row in rows:
        directions = f"{row['da_direction']}/{row['anchorability_direction']}/{row['anchor_gain_direction']}"
        lines.append(
            f"| {row['task'].replace('_to_', '->')} | {fmt(row['smooth_minus_plain_da_f1'])} | "
            f"{fmt(row['smooth_minus_plain_source_on_target_f1'])} | {fmt(row['anchorability_score'])} | "
            f"{fmt(row['smooth_minus_plain_anchor_gain_vs_best_global'])} | "
            f"{fmt(row['smooth_minus_plain_anchor_after_shift_gain_vs_best_global'])} | "
            f"{directions} | {row['three_way_relation']} |\n"
        )
    lines.append("\n## 2. Anchorability Metric Deltas\n\n")
    lines.append("| task | compactness Δ | silhouette Δ | entropy Δ | max prob Δ | persistence Δ | class entropy Δ | purity Δ |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        lines.append(
            f"| {row['task'].replace('_to_', '->')} | {fmt(row['cluster_compactness_delta'])} | "
            f"{fmt(row['silhouette_score_sampled_delta'])} | {fmt(row['assignment_entropy_mean_delta'])} | "
            f"{fmt(row['max_assignment_prob_mean_delta'])} | {fmt(row['temporal_persistence_delta'])} | "
            f"{fmt(row['class_conditional_anchor_entropy_delta'])} | "
            f"{fmt(row['class_conditional_anchor_purity_delta'])} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    summarize(args.anchor_dir, args.da_rows, args.output_dir)


if __name__ == "__main__":
    main()
