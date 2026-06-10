#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


PREDICTORS = [
    "plain_target_accuracy",
    "plain_target_confidence",
    "plain_target_entropy",
    "plain_target_effrank",
]

EFFECTS = {
    "plain": "plain_effect",
    "raw": "raw_grad_effect",
    "reshaper": "reshaper_direct_effect",
    "combo": "reshaper_combo_total_effect",
}


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8f}"
    return str(value)


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_den = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_den = sum((y - y_mean) ** 2 for _, y in pairs)
    if x_den <= 0.0 or y_den <= 0.0:
        return None, len(pairs)
    return num / math.sqrt(x_den * y_den), len(pairs)


def build_plain_metric_map(metric_rows):
    out = {}
    for row in metric_rows:
        if row.get("status") != "ok" or row.get("config") != "plain":
            continue
        key = (row["task"], row["seed"])
        out[key] = {
            "plain_target_accuracy": safe_float(row.get("target_accuracy")),
            "plain_target_confidence": safe_float(row.get("target_confidence")),
            "plain_target_entropy": safe_float(row.get("target_entropy")),
            "plain_target_effrank": safe_float(row.get("target_effrank")),
            "plain_source_fisher": safe_float(row.get("source_fisher")),
            "plain_source_effrank": safe_float(row.get("source_effrank")),
        }
    return out


def best_worst(effects):
    items = [(name, value) for name, value in effects.items() if value is not None]
    if not items:
        return None, None, None, None
    best_name, best_value = max(items, key=lambda item: item[1])
    worst_name, worst_value = min(items, key=lambda item: item[1])
    return best_name, best_value, worst_name, worst_value


def build_seed_rows(metric_rows, contrast_rows):
    plain_metrics = build_plain_metric_map(metric_rows)
    rows = []
    for contrast in contrast_rows:
        key = (contrast["task"], contrast["seed"])
        metrics = plain_metrics.get(key)
        if metrics is None:
            continue
        effects = {
            "plain": 0.0,
            "raw": safe_float(contrast.get("raw_grad_effect")),
            "reshaper": safe_float(contrast.get("reshaper_direct_effect")),
            "combo": safe_float(contrast.get("reshaper_combo_total_effect")),
        }
        best_name, best_value, worst_name, worst_value = best_worst(effects)
        row = {
            "task": contrast["task"],
            "seed": contrast["seed"],
            **metrics,
            "raw_effect": effects["raw"],
            "reshaper_effect": effects["reshaper"],
            "combo_effect": effects["combo"],
            "best_mechanism": best_name,
            "best_effect": best_value,
            "worst_mechanism": worst_name,
            "worst_effect": worst_value,
            "mechanism_gap": None if best_value is None or worst_value is None else best_value - worst_value,
            "raw_minus_reshaper": None
            if effects["raw"] is None or effects["reshaper"] is None
            else effects["raw"] - effects["reshaper"],
            "combo_minus_best_single": None
            if effects["combo"] is None or effects["raw"] is None or effects["reshaper"] is None
            else effects["combo"] - max(0.0, effects["raw"], effects["reshaper"]),
        }
        rows.append(row)
    return rows


def summarize_task_rows(seed_rows):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row["task"]].append(row)
    out = []
    numeric_fields = [
        *PREDICTORS,
        "plain_source_fisher",
        "plain_source_effrank",
        "raw_effect",
        "reshaper_effect",
        "combo_effect",
        "best_effect",
        "worst_effect",
        "mechanism_gap",
        "raw_minus_reshaper",
        "combo_minus_best_single",
    ]
    for task, rows in sorted(grouped.items()):
        row = {"task": task, "n": len(rows)}
        for field in numeric_fields:
            vals = [safe_float(item.get(field)) for item in rows]
            row[f"{field}_mean"] = mean(vals)
            row[f"{field}_std"] = stdev(vals)
        counts = defaultdict(int)
        for item in rows:
            counts[item["best_mechanism"]] += 1
        row["best_mechanism_counts"] = ",".join(f"{name}:{counts[name]}" for name in sorted(counts))
        row["best_mechanism_mode"] = max(counts.items(), key=lambda item: item[1])[0] if counts else ""
        out.append(row)
    return out


def summarize_best_groups(seed_rows):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row["best_mechanism"]].append(row)
    out = []
    for name, rows in sorted(grouped.items()):
        row = {"best_mechanism": name, "n": len(rows)}
        for field in [*PREDICTORS, "mechanism_gap", "best_effect"]:
            vals = [safe_float(item.get(field)) for item in rows]
            row[f"{field}_mean"] = mean(vals)
            row[f"{field}_std"] = stdev(vals)
        out.append(row)
    return out


def build_correlations(seed_rows, task_rows):
    specs = [
        ("seed", seed_rows, ""),
        ("task_mean", task_rows, "_mean"),
    ]
    outcome_fields = [
        "mechanism_gap",
        "best_effect",
        "raw_minus_reshaper",
        "combo_minus_best_single",
    ]
    out = []
    for level, rows, suffix in specs:
        for predictor in PREDICTORS:
            pred_field = predictor if level == "seed" else f"{predictor}_mean"
            xs = [safe_float(row.get(pred_field)) for row in rows]
            for outcome in outcome_fields:
                out_field = outcome if level == "seed" else f"{outcome}_mean"
                ys = [safe_float(row.get(out_field)) for row in rows]
                corr, n = pearson(xs, ys)
                out.append(
                    {
                        "level": level,
                        "predictor": predictor,
                        "outcome": outcome,
                        "pearson": corr,
                        "r2": None if corr is None else corr * corr,
                        "n": n,
                    }
                )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize whether plain pre-DA target state predicts v2.4.3b mechanism preference."
    )
    parser.add_argument("--geometry_metrics", required=True)
    parser.add_argument("--counterfactual_contrasts", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    metric_rows = read_tsv(args.geometry_metrics)
    contrast_rows = read_tsv(args.counterfactual_contrasts)
    seed_rows = build_seed_rows(metric_rows, contrast_rows)
    task_rows = summarize_task_rows(seed_rows)
    group_rows = summarize_best_groups(seed_rows)
    corr_rows = build_correlations(seed_rows, task_rows)

    output_dir = Path(args.output_dir)
    seed_fields = [
        "task",
        "seed",
        *PREDICTORS,
        "plain_source_fisher",
        "plain_source_effrank",
        "raw_effect",
        "reshaper_effect",
        "combo_effect",
        "best_mechanism",
        "best_effect",
        "worst_mechanism",
        "worst_effect",
        "mechanism_gap",
        "raw_minus_reshaper",
        "combo_minus_best_single",
    ]
    write_tsv(output_dir / "pre_da_seed_predictors.tsv", seed_rows, seed_fields)

    task_fields = ["task", "n"]
    for field in [
        *PREDICTORS,
        "plain_source_fisher",
        "plain_source_effrank",
        "raw_effect",
        "reshaper_effect",
        "combo_effect",
        "best_effect",
        "worst_effect",
        "mechanism_gap",
        "raw_minus_reshaper",
        "combo_minus_best_single",
    ]:
        task_fields.extend([f"{field}_mean", f"{field}_std"])
    task_fields.extend(["best_mechanism_counts", "best_mechanism_mode"])
    write_tsv(output_dir / "pre_da_task_predictors.tsv", task_rows, task_fields)

    group_fields = ["best_mechanism", "n"]
    for field in [*PREDICTORS, "mechanism_gap", "best_effect"]:
        group_fields.extend([f"{field}_mean", f"{field}_std"])
    write_tsv(output_dir / "pre_da_best_mechanism_groups.tsv", group_rows, group_fields)
    write_tsv(output_dir / "pre_da_predictor_correlations.tsv", corr_rows, ["level", "predictor", "outcome", "pearson", "r2", "n"])

    print("Wrote:", output_dir / "pre_da_seed_predictors.tsv")
    print("Wrote:", output_dir / "pre_da_task_predictors.tsv")
    print("Wrote:", output_dir / "pre_da_best_mechanism_groups.tsv")
    print("Wrote:", output_dir / "pre_da_predictor_correlations.tsv")


if __name__ == "__main__":
    main()
