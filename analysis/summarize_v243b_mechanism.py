#!/usr/bin/env python3
import csv
import re
import statistics as stats
import sys
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    name_match = NAME_RE.match(path.name)
    task = name_match.group(1) if name_match else path.stem
    seed = int(name_match.group(2)) if name_match else -1
    config = name_match.group(3) if name_match else "unknown"
    status = "ok"
    if "Traceback" in text or "error:" in text.lower():
        status = "error"
    if len(tests) < 3:
        status = "incomplete"
    source_self = source_target = da = None
    if len(tests) >= 3:
        source_self = float(tests[-3][2])
        source_target = float(tests[-2][2])
        da = float(tests[-1][2])
    return {
        "task": task,
        "seed": seed,
        "config": config,
        "source_self_f1": source_self,
        "source_on_target_f1": source_target,
        "da_f1": da,
        "da_gain": None if da is None or source_target is None else da - source_target,
        "status": status,
        "log": path.name,
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(field)) for field in fields) + "\n")


def mean(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def stdev(values):
    values = [v for v in values if v is not None]
    return None if len(values) < 2 else stats.stdev(values)


def cohens_d(values_a, values_b):
    values_a = [v for v in values_a if v is not None]
    values_b = [v for v in values_b if v is not None]
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    var_a = stats.variance(values_a)
    var_b = stats.variance(values_b)
    pooled = ((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / (
        len(values_a) + len(values_b) - 2
    )
    if pooled <= 0:
        return None
    return (mean(values_a) - mean(values_b)) / (pooled ** 0.5)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_mechanism.py LOG_DIR")
    root = Path(sys.argv[1])
    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]

    fields = [
        "task",
        "seed",
        "config",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "status",
        "log",
    ]
    write_tsv(root / "summary.tsv", rows, fields)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["task"], row["config"]), []).append(row)

    config_rows = []
    for (task, config), group in sorted(grouped.items()):
        da_values = [row["da_f1"] for row in group]
        sot_values = [row["source_on_target_f1"] for row in group]
        gain_values = [row["da_gain"] for row in group]
        config_rows.append(
            {
                "task": task,
                "config": config,
                "n": len(group),
                "da_mean": mean(da_values),
                "da_std": stdev(da_values),
                "source_on_target_mean": mean(sot_values),
                "source_on_target_std": stdev(sot_values),
                "da_gain_mean": mean(gain_values),
                "da_gain_std": stdev(gain_values),
                "ok_count": sum(1 for row in group if row["status"] == "ok"),
            }
        )
    config_fields = [
        "task",
        "config",
        "n",
        "ok_count",
        "da_mean",
        "da_std",
        "source_on_target_mean",
        "source_on_target_std",
        "da_gain_mean",
        "da_gain_std",
    ]
    write_tsv(root / "config_summary.tsv", config_rows, config_fields)

    contrasts = [
        ("full_doy", "nostruct_global", "full_vs_nostruct"),
        ("full_doy", "full_uniform", "doy_vs_uniform"),
        ("full_doy", "doy_intra_trend", "transition_boundary_added"),
        ("full_doy", "doy_no_trend_renorm", "trend_needed"),
        ("full_doy", "doy_no_segment_inter_renorm", "segment_inter_needed"),
        ("full_doy", "doy_no_boundary_mod", "boundary_mod_needed"),
        ("full_doy", "full_doy_daoff", "da_stage_structure_needed"),
        ("full_doy", "no_reshaper", "reshaper_stack_needed"),
    ]
    by_task_config = {}
    for row in rows:
        by_task_config.setdefault((row["task"], row["config"]), []).append(row)

    contrast_rows = []
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        for left, right, name in contrasts:
            left_rows = by_task_config.get((task, left), [])
            right_rows = by_task_config.get((task, right), [])
            left_values = [row["da_f1"] for row in left_rows]
            right_values = [row["da_f1"] for row in right_rows]
            contrast_rows.append(
                {
                    "task": task,
                    "contrast": name,
                    "left": left,
                    "right": right,
                    "left_mean": mean(left_values),
                    "right_mean": mean(right_values),
                    "delta": None
                    if mean(left_values) is None or mean(right_values) is None
                    else mean(left_values) - mean(right_values),
                    "cohens_d": cohens_d(left_values, right_values),
                    "left_n": len(left_rows),
                    "right_n": len(right_rows),
                }
            )
    contrast_fields = [
        "task",
        "contrast",
        "left",
        "right",
        "left_n",
        "right_n",
        "left_mean",
        "right_mean",
        "delta",
        "cohens_d",
    ]
    write_tsv(root / "contrast_summary.tsv", contrast_rows, contrast_fields)

    print("Wrote:", root / "summary.tsv")
    print("Wrote:", root / "config_summary.tsv")
    print("Wrote:", root / "contrast_summary.tsv")
    for row in contrast_rows:
        if row["contrast"] in {"doy_vs_uniform", "segment_inter_needed", "boundary_mod_needed"}:
            print(
                f"{row['task']} {row['contrast']}: "
                f"delta={fmt(row['delta'])}, d={fmt(row['cohens_d'])}"
            )


if __name__ == "__main__":
    main()
