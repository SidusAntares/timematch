#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path


WEIGHTS = ["0.250000", "0.500000", "1.000000", "2.000000"]


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def read_tsv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(fmt(row.get(field)) for field in fields) + "\n")


def mean(values):
    vals = [value for value in values if value is not None]
    return None if not vals else sum(vals) / len(vals)


def stdev(values):
    vals = [value for value in values if value is not None]
    if len(vals) < 2:
        return None
    mu = mean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / (len(vals) - 1))


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def latest_response_map_dir():
    roots = sorted(
        (
            path
            for path in Path("logs").glob("v243b_raw_strength_response_map_*")
            if (path / "strength_effect_curve.tsv").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not roots:
        raise SystemExit("No logs/v243b_raw_strength_response_map_* directory with strength_effect_curve.tsv found")
    return roots[0]


def classify_response(delta_values, positive_count, readiness_mean):
    curve_mean = mean(delta_values) or 0.0
    high_mean = mean(delta_values[2:]) or 0.0
    light_mean = mean(delta_values[:2]) or 0.0
    if curve_mean >= 0.04 and positive_count >= 9:
        return "strong_positive"
    if curve_mean >= 0.015 and positive_count >= 8:
        return "moderate_positive"
    if light_mean > 0.005 and high_mean <= 0.005:
        return "light_positive_high_unstable"
    if curve_mean <= -0.005 and readiness_mean <= 0.0:
        return "negative_or_flat"
    return "mixed_or_weak"


def readiness_note(last_all_values):
    value = mean(last_all_values) or 0.0
    if value >= 0.02:
        return "readiness_improves"
    if value <= -0.01:
        return "readiness_degrades"
    return "readiness_mixed"


def build_task_rows(root, rows, curve_rows, delta_rows):
    plain_rows = [row for row in rows if row["config"] == "plain" and row["status"] == "ok"]
    curve_by_task_weight = {
        (row["task"], f"{to_float(row['compact_weight']):.6f}"): row for row in curve_rows
    }
    delta_by_task = {}
    for row in delta_rows:
        delta_by_task.setdefault(row["task"], []).append(row)

    task_rows = []
    for task in sorted({row["task"] for row in rows}):
        task_plain = [row for row in plain_rows if row["task"] == task]
        deltas = []
        last_all = []
        best_all = []
        initial_all = []
        pos_count = 0
        out = {
            "task": task,
            "plain_source_on_target_mean": mean(to_float(row["source_on_target_f1"]) for row in task_plain),
            "plain_da_f1_mean": mean(to_float(row["da_f1"]) for row in task_plain),
            "plain_initial_all_f1_mean": mean(to_float(row["initial_all_f1"]) for row in task_plain),
            "plain_initial_masked_f1_mean": mean(to_float(row["initial_masked_f1"]) for row in task_plain),
            "plain_initial_coverage_mean": mean(to_float(row["initial_coverage"]) for row in task_plain),
            "plain_last_all_f1_mean": mean(to_float(row["last_all_f1"]) for row in task_plain),
            "plain_last_masked_f1_mean": mean(to_float(row["last_masked_f1"]) for row in task_plain),
            "plain_last_coverage_mean": mean(to_float(row["last_coverage"]) for row in task_plain),
        }
        for weight in WEIGHTS:
            row = curve_by_task_weight.get((task, weight), {})
            da = to_float(row.get("delta_da_f1_mean"))
            la = to_float(row.get("delta_last_all_f1_mean"))
            ba = to_float(row.get("delta_best_epoch_all_f1_mean"))
            ia = to_float(row.get("delta_initial_all_f1_mean"))
            pos = int(to_float(row.get("delta_da_f1_pos")) or 0)
            deltas.append(da)
            last_all.append(la)
            best_all.append(ba)
            initial_all.append(ia)
            pos_count += pos
            tag = weight.replace(".", "p").rstrip("0").rstrip("p")
            out[f"w{tag}_delta_da"] = da
            out[f"w{tag}_delta_last_all"] = la
            out[f"w{tag}_delta_best_all"] = ba
            out[f"w{tag}_positive"] = pos

        task_delta_rows = delta_by_task.get(task, [])
        da_values = [to_float(row.get("delta_da_f1")) for row in task_delta_rows]
        out.update(
            {
                "curve_delta_da_mean": mean(deltas),
                "curve_delta_da_std_across_weights": stdev(deltas),
                "positive_seed_weight_count": pos_count,
                "light_weight_delta_mean": mean(deltas[:2]),
                "high_weight_delta_mean": mean(deltas[2:]),
                "w1_minus_w05_delta_da": None
                if deltas[1] is None or deltas[2] is None
                else deltas[2] - deltas[1],
                "w2_minus_w1_delta_da": None
                if deltas[2] is None or deltas[3] is None
                else deltas[3] - deltas[2],
                "curve_delta_last_all_mean": mean(last_all),
                "curve_delta_best_all_mean": mean(best_all),
                "curve_delta_initial_all_mean": mean(initial_all),
                "corr_delta_da_last_all": corr(
                    [to_float(row.get("delta_last_all_f1")) for row in task_delta_rows],
                    da_values,
                ),
                "corr_delta_da_best_all": corr(
                    [to_float(row.get("delta_best_epoch_all_f1")) for row in task_delta_rows],
                    da_values,
                ),
                "corr_delta_da_initial_all": corr(
                    [to_float(row.get("delta_initial_all_f1")) for row in task_delta_rows],
                    da_values,
                ),
            }
        )
        out["response_family"] = classify_response(
            deltas,
            pos_count,
            out["curve_delta_last_all_mean"] or 0.0,
        )
        out["readiness_pattern"] = readiness_note(last_all)
        task_rows.append(out)
    return task_rows


def build_group_rows(task_rows):
    groups = {}
    for row in task_rows:
        groups.setdefault(row["response_family"], []).append(row)
    fields = [
        "plain_source_on_target_mean",
        "plain_da_f1_mean",
        "plain_initial_all_f1_mean",
        "plain_initial_masked_f1_mean",
        "plain_initial_coverage_mean",
        "curve_delta_da_mean",
        "curve_delta_last_all_mean",
        "curve_delta_best_all_mean",
        "light_weight_delta_mean",
        "high_weight_delta_mean",
        "positive_seed_weight_count",
    ]
    out = []
    for family, rows in sorted(groups.items()):
        item = {"response_family": family, "task_count": len(rows), "tasks": ",".join(row["task"] for row in rows)}
        for field in fields:
            item[f"{field}_mean"] = mean(row.get(field) for row in rows)
        out.append(item)
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else latest_response_map_dir()
    rows_path = root / "raw_strength_rows.tsv"
    curve_path = root / "strength_effect_curve.tsv"
    delta_path = root / "pseudolabel_divergence.tsv"
    if not rows_path.exists() or not curve_path.exists() or not delta_path.exists():
        raise SystemExit(f"Missing required summary files in {root}")

    rows = read_tsv(rows_path)
    curve_rows = read_tsv(curve_path)
    delta_rows = read_tsv(delta_path)
    task_rows = build_task_rows(root, rows, curve_rows, delta_rows)
    group_rows = build_group_rows(task_rows)

    task_fields = [
        "task",
        "response_family",
        "readiness_pattern",
        "plain_source_on_target_mean",
        "plain_da_f1_mean",
        "plain_initial_all_f1_mean",
        "plain_initial_masked_f1_mean",
        "plain_initial_coverage_mean",
        "plain_last_all_f1_mean",
        "plain_last_masked_f1_mean",
        "plain_last_coverage_mean",
        "w0p25_delta_da",
        "w0p5_delta_da",
        "w1_delta_da",
        "w2_delta_da",
        "w0p25_positive",
        "w0p5_positive",
        "w1_positive",
        "w2_positive",
        "w0p25_delta_last_all",
        "w0p5_delta_last_all",
        "w1_delta_last_all",
        "w2_delta_last_all",
        "curve_delta_da_mean",
        "curve_delta_da_std_across_weights",
        "positive_seed_weight_count",
        "light_weight_delta_mean",
        "high_weight_delta_mean",
        "w1_minus_w05_delta_da",
        "w2_minus_w1_delta_da",
        "curve_delta_last_all_mean",
        "curve_delta_best_all_mean",
        "curve_delta_initial_all_mean",
        "corr_delta_da_last_all",
        "corr_delta_da_best_all",
        "corr_delta_da_initial_all",
    ]
    group_fields = [
        "response_family",
        "task_count",
        "tasks",
        "plain_source_on_target_mean_mean",
        "plain_da_f1_mean_mean",
        "plain_initial_all_f1_mean_mean",
        "plain_initial_masked_f1_mean_mean",
        "plain_initial_coverage_mean_mean",
        "curve_delta_da_mean_mean",
        "curve_delta_last_all_mean_mean",
        "curve_delta_best_all_mean_mean",
        "light_weight_delta_mean_mean",
        "high_weight_delta_mean_mean",
        "positive_seed_weight_count_mean",
    ]
    write_tsv(root / "strength_response_type_attribution.tsv", task_rows, task_fields)
    write_tsv(root / "strength_response_group_summary.tsv", group_rows, group_fields)
    print("Wrote:", root / "strength_response_type_attribution.tsv")
    print("Wrote:", root / "strength_response_group_summary.tsv")


if __name__ == "__main__":
    main()
