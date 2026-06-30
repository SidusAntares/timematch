#!/usr/bin/env python3
import csv
import json
import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
SOURCE_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_source\.log$")
DA_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")

CONFIG_ORDER = [
    "original_timematch",
    "no_shift",
    "fixed_initial_shift",
    "oracle_scalar_shift_diagnostic",
    "topk_shift_ensemble_diagnostic",
    "oracle_classwise_shift_diagnostic",
]


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def fmt_md(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def stdev(values):
    values = [value for value in values if value is not None]
    return None if len(values) < 2 else stats.stdev(values)


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_md_table(handle, rows, fields, headers=None, limit=None):
    rows = rows[:limit] if limit else rows
    headers = headers or fields
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("|" + "|".join("---" for _ in headers) + "|\n")
    for row in rows:
        handle.write("| " + " | ".join(fmt_md(row.get(field)) for field in fields) + " |\n")
    handle.write("\n")


def parse_log_tests(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    status = "ok"
    if "Traceback" in text or "error:" in text.lower():
        status = "error"
    return tests, status, text


def parse_source_logs(root):
    source_rows = {}
    failures = []
    for path in sorted(root.glob("gpu*_source.log")):
        match = SOURCE_RE.match(path.name)
        if not match:
            continue
        task, seed = match.group(1), int(match.group(2))
        tests, status, _ = parse_log_tests(path)
        source_self = source_on_target = None
        if len(tests) >= 2:
            source_self = safe_float(tests[-2][2])
            source_on_target = safe_float(tests[-1][2])
        else:
            status = "incomplete"
        source_rows[(task, seed)] = {
            "task": task,
            "seed": seed,
            "source_self_f1": source_self,
            "source_on_target_f1": source_on_target,
            "source_log": path.name,
            "status": status,
        }
        if status != "ok":
            failures.append({"stage": "source", "task": task, "seed": seed, "config": "source", "status": status, "log": path.name})
    return source_rows, failures


def parse_da_logs(root, source_rows):
    rows = []
    failures = []
    for path in sorted(root.glob("gpu*_*.log")):
        if path.name.endswith("_source.log"):
            continue
        match = DA_RE.match(path.name)
        if not match:
            continue
        task, seed, config = match.group(1), int(match.group(2)), match.group(3)
        if config not in CONFIG_ORDER:
            continue
        tests, status, _ = parse_log_tests(path)
        da_f1 = safe_float(tests[-1][2]) if tests else None
        if not tests:
            status = "incomplete"
        source = source_rows.get((task, seed), {})
        row = {
            "task": task,
            "seed": seed,
            "config": config,
            "source_self_f1": source.get("source_self_f1"),
            "source_on_target_f1": source.get("source_on_target_f1"),
            "da_f1": da_f1,
            "da_gain": delta(da_f1, source.get("source_on_target_f1")),
            "shift_policy": config,
            "log": path.name,
            "status": status,
        }
        rows.append(row)
        if status != "ok":
            failures.append({"stage": "da", "task": task, "seed": seed, "config": config, "status": status, "log": path.name})
    return rows, failures


def merge_offline(root, filename, output_name, fields=None):
    rows = []
    for path in sorted((root / "offline").glob(f"*/{filename}")):
        rows.extend(read_tsv(path))
    if rows:
        fields = fields or list(rows[0].keys())
    else:
        fields = fields or []
    if fields:
        write_tsv(root / output_name, rows, fields)
    return rows, fields


def merge_trajectory(root):
    rows = []
    for path in sorted((root / "trajectory").glob("*.tsv")):
        rows.extend(read_tsv(path))
    fields = list(rows[0].keys()) if rows else []
    if fields:
        write_tsv(root / "timematch_shift_trajectory.tsv", rows, fields)
    return rows


def summarize_by_task_config(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups[(row["task"], row["config"])].append(row)
    original = {}
    for (task, config), items in groups.items():
        if config == "original_timematch":
            original[task] = mean([safe_float(item.get("da_f1")) for item in items])

    summary = []
    for key in sorted(groups, key=lambda k: (k[0], CONFIG_ORDER.index(k[1]) if k[1] in CONFIG_ORDER else 999)):
        task, config = key
        items = groups[key]
        da_values = [safe_float(item.get("da_f1")) for item in items]
        source_values = [safe_float(item.get("source_on_target_f1")) for item in items]
        gain_values = [safe_float(item.get("da_gain")) for item in items]
        base = original.get(task)
        deltas = [delta(safe_float(item.get("da_f1")), base) for item in items]
        positive = sum(1 for value in deltas if value is not None and value > 0)
        negative = sum(1 for value in deltas if value is not None and value < 0)
        summary.append(
            {
                "task": task,
                "config": config,
                "n": len(items),
                "source_on_target_mean": mean(source_values),
                "da_f1_mean": mean(da_values),
                "da_gain_mean": mean(gain_values),
                "da_f1_std": stdev(da_values),
                "delta_vs_original": delta(mean(da_values), base),
                "positive_seeds_vs_original": positive,
                "negative_seeds_vs_original": negative,
            }
        )
    return summary


def summarize_by_config(task_summary):
    groups = defaultdict(list)
    for row in task_summary:
        groups[row["config"]].append(row)
    config_rows = []
    for config in CONFIG_ORDER:
        items = groups.get(config, [])
        if not items:
            continue
        deltas = [safe_float(item.get("delta_vs_original")) for item in items]
        config_rows.append(
            {
                "config": config,
                "n": sum(int(item.get("n") or 0) for item in items),
                "da_f1_mean": mean([safe_float(item.get("da_f1_mean")) for item in items]),
                "da_gain_mean": mean([safe_float(item.get("da_gain_mean")) for item in items]),
                "delta_vs_original": mean(deltas),
                "positive_tasks_vs_original": sum(1 for value in deltas if value is not None and value > 0),
                "negative_tasks_vs_original": sum(1 for value in deltas if value is not None and value < 0),
                "positive_seeds_vs_original": sum(int(item.get("positive_seeds_vs_original") or 0) for item in items),
                "negative_seeds_vs_original": sum(int(item.get("negative_seeds_vs_original") or 0) for item in items),
            }
        )
    return config_rows


def latest_trajectory_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("task"), row.get("seed"), row.get("config"))].append(row)
    finals = {}
    for key, items in groups.items():
        items = sorted(items, key=lambda item: safe_int(item.get("epoch")) or -1)
        shifts = [safe_float(item.get("estimated_shift_t_to_s")) for item in items if item.get("estimated_shift_t_to_s") != ""]
        margins = [safe_float(item.get("am_top1_top2_margin")) for item in items]
        coverage = [safe_float(item.get("teacher_pseudo_coverage")) for item in items]
        entropy = [safe_float(item.get("teacher_pseudo_class_entropy")) for item in items]
        if shifts:
            shift_changes = sum(1 for a, b in zip(shifts, shifts[1:]) if a != b)
            shift_drift = shifts[-1] - shifts[0]
            shift_std = stdev(shifts)
        else:
            shift_changes = 0
            shift_drift = None
            shift_std = None
        finals[key] = {
            "initial_estimated_shift": shifts[0] if shifts else None,
            "final_estimated_shift": shifts[-1] if shifts else None,
            "shift_total_drift": shift_drift,
            "shift_num_changes": shift_changes,
            "shift_std_over_epochs": shift_std,
            "mean_am_margin": mean(margins),
            "mean_pseudo_coverage": mean(coverage),
            "mean_pseudo_class_entropy": mean(entropy),
        }
    return finals


def add_trajectory_to_results(rows, trajectory):
    for row in rows:
        key = (row["task"], str(row["seed"]), row["config"])
        alt_key = (row["task"], row["seed"], row["config"])
        detail = trajectory.get(key) or trajectory.get(alt_key) or {}
        row.update(detail)
        row["initial_shift"] = detail.get("initial_estimated_shift")
        row["final_shift"] = detail.get("final_estimated_shift")
        row["oracle_scalar_shift_if_available"] = ""
    return rows


def write_summary_md(root, curve_rows, oracle_rows, class_dispersion_rows, trajectory_rows, ablation_summary, config_summary, failures):
    md = root / "v300_global_shift_assumption_diagnostic_summary.md"
    oracle_delta = next((row for row in config_summary if row["config"] == "oracle_scalar_shift_diagnostic"), {})
    no_shift_delta = next((row for row in config_summary if row["config"] == "no_shift"), {})
    fixed_delta = next((row for row in config_summary if row["config"] == "fixed_initial_shift"), {})
    topk_delta = next((row for row in config_summary if row["config"] == "topk_shift_ensemble_diagnostic"), {})

    abs_shift_errors = [safe_float(row.get("abs_shift_error")) for row in oracle_rows]
    class_dispersion = [safe_float(row.get("weighted_class_shift_std")) for row in class_dispersion_rows]
    shift_changes = [safe_float(row.get("shift_num_changes")) for row in trajectory_rows]

    with md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v3.0.0 Global Shift Assumption Diagnostic Summary\n\n")
        handle.write("## 1. Experiment Setup\n\n")
        handle.write(
            f"| source foundation | tasks | seeds/config runs | failed |\n"
            f"|---|---:|---:|---:|\n"
            f"| smooth_k3 λ=1.0, source 100 epochs | "
            f"{len(set(row.get('task') for row in ablation_summary))} | "
            f"{sum(int(row.get('n') or 0) for row in ablation_summary)} | {len(failures)} |\n\n"
        )

        handle.write("## 2. Source-only Shift Score Curves\n\n")
        handle.write(
            f"| curve rows | mean abs estimated-vs-oracle shift error |\n"
            f"|---:|---:|\n"
            f"| {len(curve_rows)} | {fmt_md(mean(abs_shift_errors))} |\n\n"
        )

        handle.write("## 3. Offline Oracle Scalar Shift\n\n")
        write_md_table(
            handle,
            oracle_rows,
            [
                "task",
                "seed",
                "estimated_shift_am",
                "oracle_scalar_shift_by_f1",
                "estimated_shift_f1",
                "oracle_scalar_f1",
                "shift_estimation_gap",
                "abs_shift_error",
            ],
            limit=30,
        )

        handle.write("## 4. Class-wise Oracle Shift\n\n")
        write_md_table(
            handle,
            class_dispersion_rows,
            [
                "task",
                "seed",
                "n_classes_valid",
                "global_oracle_shift",
                "estimated_am_shift",
                "weighted_class_shift_std",
                "mean_abs_class_shift_minus_global",
                "mean_abs_class_shift_minus_estimated",
            ],
            limit=30,
        )

        handle.write("## 5. TimeMatch Shift Trajectory\n\n")
        handle.write(
            f"| trajectory rows | mean shift changes | mean shift std |\n"
            f"|---:|---:|---:|\n"
            f"| {len(trajectory_rows)} | {fmt_md(mean(shift_changes))} | "
            f"{fmt_md(mean([safe_float(row.get('shift_std_over_epochs')) for row in trajectory_rows]))} |\n\n"
        )

        handle.write("## 6. Shift Ablation Results\n\n")
        write_md_table(
            handle,
            config_summary,
            [
                "config",
                "n",
                "da_f1_mean",
                "da_gain_mean",
                "delta_vs_original",
                "positive_tasks_vs_original",
                "negative_tasks_vs_original",
                "positive_seeds_vs_original",
                "negative_seeds_vs_original",
            ],
        )

        handle.write("## 7. Bottleneck Evidence\n\n")
        oracle_delta_value = safe_float(oracle_delta.get("delta_vs_original"))
        topk_delta_value = safe_float(topk_delta.get("delta_vs_original"))
        fixed_delta_value = safe_float(fixed_delta.get("delta_vs_original"))
        no_shift_delta_value = safe_float(no_shift_delta.get("delta_vs_original"))
        mean_abs_error = mean(abs_shift_errors)
        mean_class_std = mean(class_dispersion)
        mean_changes = mean(shift_changes)
        evidence_rows = [
            {
                "question": "shift estimator bottleneck",
                "value": oracle_delta_value,
                "support": "yes" if oracle_delta_value is not None and oracle_delta_value >= 0.005 and (mean_abs_error or 0) > 0 else "no/weak",
            },
            {
                "question": "global scalar expressivity bottleneck",
                "value": mean_class_std,
                "support": "check_classwise_dispersion",
            },
            {
                "question": "hard argmax bottleneck",
                "value": topk_delta_value,
                "support": "yes" if topk_delta_value is not None and topk_delta_value >= 0.005 else "no/weak",
            },
            {
                "question": "re-estimation bottleneck",
                "value": fixed_delta_value,
                "support": "yes" if fixed_delta_value is not None and fixed_delta_value >= 0.005 and (mean_changes or 0) > 0 else "no/weak",
            },
            {
                "question": "shift module not primary bottleneck",
                "value": no_shift_delta_value,
                "support": "check_if_no_shift_lower_and_oracles_not_higher",
            },
        ]
        write_md_table(handle, evidence_rows, ["question", "value", "support"])

        handle.write("## 8. Minimal Observations\n\n")
        handle.write("- This summary reports diagnostic statistics only.\n")
        handle.write("- Oracle scalar/classwise shifts use target labels only for offline diagnosis.\n")
        handle.write("- oracle_classwise_shift_diagnostic is recorded as offline_only unless full DA rows exist.\n")
    return md


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v300_global_shift_assumption_diagnostic.py LOG_DIR")
    root = Path(sys.argv[1])

    curve_rows, curve_fields = merge_offline(root, "shift_score_curves_source_only.tsv", "shift_score_curves_source_only.tsv")
    oracle_rows, oracle_fields = merge_offline(root, "oracle_scalar_shift_summary.tsv", "oracle_scalar_shift_summary.tsv")
    class_rows, class_fields = merge_offline(root, "classwise_oracle_shift.tsv", "classwise_oracle_shift.tsv")
    class_dispersion_rows, class_dispersion_fields = merge_offline(
        root,
        "classwise_shift_dispersion_summary.tsv",
        "classwise_shift_dispersion_summary.tsv",
    )
    trajectory_rows = merge_trajectory(root)

    source_rows, source_failures = parse_source_logs(root)
    da_rows, da_failures = parse_da_logs(root, source_rows)
    trajectory_summary = latest_trajectory_summary(trajectory_rows)
    final_rows = add_trajectory_to_results(da_rows, trajectory_summary)

    oracle_by_task_seed = {
        (row.get("task"), safe_int(row.get("seed"))): row.get("oracle_scalar_shift_by_f1")
        for row in oracle_rows
    }
    for row in final_rows:
        row["oracle_scalar_shift_if_available"] = oracle_by_task_seed.get((row["task"], safe_int(row["seed"])), "")

    final_fields = [
        "task",
        "seed",
        "config",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "initial_shift",
        "final_shift",
        "oracle_scalar_shift_if_available",
        "shift_policy",
        "log",
        "status",
    ]
    write_tsv(root / "shift_ablation_results.tsv", final_rows, final_fields)
    write_tsv(root / "timematch_final_results.tsv", final_rows, final_fields)

    task_summary = summarize_by_task_config(final_rows)
    task_fields = [
        "task",
        "config",
        "n",
        "source_on_target_mean",
        "da_f1_mean",
        "da_gain_mean",
        "da_f1_std",
        "delta_vs_original",
        "positive_seeds_vs_original",
        "negative_seeds_vs_original",
    ]
    write_tsv(root / "shift_ablation_summary.tsv", task_summary, task_fields)

    config_summary = summarize_by_config(task_summary)
    config_fields = [
        "config",
        "n",
        "da_f1_mean",
        "da_gain_mean",
        "delta_vs_original",
        "positive_tasks_vs_original",
        "negative_tasks_vs_original",
        "positive_seeds_vs_original",
        "negative_seeds_vs_original",
    ]
    write_tsv(root / "shift_ablation_config_summary.tsv", config_summary, config_fields)

    failures = source_failures + da_failures
    write_tsv(root / "failed_runs.tsv", failures, ["stage", "task", "seed", "config", "status", "log"])
    md = write_summary_md(
        root,
        curve_rows,
        oracle_rows,
        class_dispersion_rows,
        trajectory_rows,
        task_summary,
        config_summary,
        failures,
    )

    print("Wrote:", root / "shift_score_curves_source_only.tsv")
    print("Wrote:", root / "oracle_scalar_shift_summary.tsv")
    print("Wrote:", root / "classwise_oracle_shift.tsv")
    print("Wrote:", root / "classwise_shift_dispersion_summary.tsv")
    print("Wrote:", root / "timematch_shift_trajectory.tsv")
    print("Wrote:", root / "timematch_final_results.tsv")
    print("Wrote:", root / "shift_ablation_results.tsv")
    print("Wrote:", root / "shift_ablation_summary.tsv")
    print("Wrote:", root / "shift_ablation_config_summary.tsv")
    print("Wrote:", md)
    print("Wrote:", root / "failed_runs.tsv")


if __name__ == "__main__":
    main()
