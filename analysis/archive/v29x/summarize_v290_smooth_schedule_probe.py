#!/usr/bin/env python3
import csv
import re
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")


CONFIG_ORDER = [
    "smooth_const",
    "decay_70_05",
    "decay_70_03",
    "decay_50_03",
    "warmup_30",
    "cosine_decay_03",
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


def parse_kv_line(line, prefix):
    values = {}
    for part in line[len(prefix):].strip().split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def parse_task(task):
    source, target = task.split("_to_")
    return source, target


def write_tsv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
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


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    name_match = NAME_RE.match(path.name)
    task = name_match.group(1) if name_match else path.stem
    seed = int(name_match.group(2)) if name_match else -1
    config = name_match.group(3) if name_match else "unknown"
    source, target = parse_task(task) if "_to_" in task else ("", "")

    tests = TEST_RE.findall(text)
    source_rows = [
        parse_kv_line(line, "SOURCE_EPOCH_SUMMARY|")
        for line in text.splitlines()
        if line.startswith("SOURCE_EPOCH_SUMMARY|")
    ]
    lambda_curve_rows = [
        parse_kv_line(line, "SOURCE_LAMBDA_CURVE|")
        for line in text.splitlines()
        if line.startswith("SOURCE_LAMBDA_CURVE|")
    ]
    final_source = source_rows[-1] if source_rows else {}
    lambda_curve = lambda_curve_rows[-1] if lambda_curve_rows else {}

    status = "ok"
    if "Traceback" in text or "error:" in text.lower():
        status = "error"
    if len(tests) < 3:
        status = "incomplete"

    source_self_f1 = source_on_target_f1 = da_f1 = None
    if len(tests) >= 3:
        source_self_f1 = safe_float(tests[-3][2])
        source_on_target_f1 = safe_float(tests[-2][2])
        da_f1 = safe_float(tests[-1][2])

    return {
        "task": task,
        "source": source,
        "target": target,
        "seed": seed,
        "config": config,
        "schedule_type": lambda_curve.get("schedule") or final_source.get("schedule"),
        "lambda_base": safe_float(lambda_curve.get("base")),
        "lambda_final": safe_float(lambda_curve.get("final")),
        "decay_start_epoch": safe_int(lambda_curve.get("decay_start_epoch")),
        "warmup_epochs": safe_int(lambda_curve.get("warmup_epochs")),
        "max_epoch": safe_int(lambda_curve.get("max_epoch")),
        "source_self_f1": source_self_f1,
        "source_on_target_f1": source_on_target_f1,
        "da_f1": da_f1,
        "da_gain": delta(da_f1, source_on_target_f1),
        "final_source_structure_lambda": safe_float(lambda_curve.get("final_value") or final_source.get("source_structure_lambda")),
        "mean_source_structure_lambda": safe_float(lambda_curve.get("mean")),
        "min_source_structure_lambda": safe_float(lambda_curve.get("min")),
        "max_source_structure_lambda": safe_float(lambda_curve.get("max")),
        "source_loss": safe_float(final_source.get("loss")),
        "source_cls_loss": safe_float(final_source.get("cls")),
        "source_compact_loss": safe_float(final_source.get("compact")),
        "source_compact_raw_loss": safe_float(final_source.get("compact_raw")),
        "status": status,
        "log": path.name,
    }


def parse_lambda_curve_files(root):
    rows = []
    for path in sorted(root.glob("gpu*_*.log")):
        parsed = parse_log(path)
        if parsed["config"] == "unknown":
            continue
        for epoch, value in lambda_values_from_schedule(parsed):
            rows.append(
                {
                    "config": parsed["config"],
                    "epoch": epoch,
                    "lambda_value": value,
                    "schedule_type": parsed.get("schedule_type"),
                    "lambda_base": parsed.get("lambda_base"),
                    "lambda_final": parsed.get("lambda_final"),
                    "decay_start_epoch": parsed.get("decay_start_epoch"),
                    "warmup_epochs": parsed.get("warmup_epochs"),
                    "max_epoch": parsed.get("max_epoch"),
                }
            )
    seen = set()
    unique = []
    for row in rows:
        key = (row["config"], row["epoch"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return sorted(unique, key=lambda row: (CONFIG_ORDER.index(row["config"]) if row["config"] in CONFIG_ORDER else 999, row["epoch"]))


def lambda_values_from_schedule(row):
    schedule = row.get("schedule_type") or "constant"
    base = row.get("lambda_base")
    final = row.get("lambda_final")
    max_epoch = row.get("max_epoch")
    if base is None or final is None or max_epoch is None:
        return []
    values = []
    for epoch in range(1, int(max_epoch) + 1):
        if schedule == "constant":
            value = base
        elif schedule == "linear_decay":
            decay_start = int(row.get("decay_start_epoch") or 1)
            if epoch <= decay_start:
                value = base
            else:
                progress = min(1.0, max(0.0, (epoch - decay_start) / float(max(1, max_epoch - decay_start))))
                value = base + progress * (final - base)
        elif schedule == "warmup_then_constant":
            value = 0.0 if epoch <= int(row.get("warmup_epochs") or 0) else base
        elif schedule == "cosine_decay":
            import math

            progress = min(1.0, max(0.0, epoch / float(max_epoch)))
            value = final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))
        else:
            value = base
        values.append((epoch, value))
    return values


def summarize(rows, group_fields, value_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item["n"] = len(group)
        for field in value_fields:
            item[f"{field}_mean"] = mean([row.get(field) for row in group])
            item[f"{field}_std"] = stdev([row.get(field) for row in group])
        out.append(item)
    return out


def build_config_summary(ok_rows):
    rows = summarize(
        ok_rows,
        ["config"],
        [
            "source_self_f1",
            "source_on_target_f1",
            "da_f1",
            "da_gain",
            "mean_source_structure_lambda",
            "final_source_structure_lambda",
        ],
    )
    order = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    return sorted(rows, key=lambda row: order.get(row["config"], 999))


def build_task_summary(ok_rows):
    rows = summarize(
        ok_rows,
        ["task", "source", "target", "config"],
        ["source_self_f1", "source_on_target_f1", "da_f1", "da_gain"],
    )
    order = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    return sorted(rows, key=lambda row: (row["task"], order.get(row["config"], 999)))


def build_relative(ok_rows):
    base = {
        (row["task"], row["seed"]): row
        for row in ok_rows
        if row.get("config") == "smooth_const"
    }
    grouped = defaultdict(list)
    for row in ok_rows:
        if row.get("config") == "smooth_const":
            continue
        smooth = base.get((row["task"], row["seed"]))
        if smooth is None:
            continue
        item = {
            "task": row["task"],
            "source": row["source"],
            "target": row["target"],
            "seed": row["seed"],
            "config": row["config"],
            "smooth_const_da": smooth.get("da_f1"),
            "config_da": row.get("da_f1"),
            "delta_da_vs_smooth_const": delta(row.get("da_f1"), smooth.get("da_f1")),
            "smooth_const_gain": smooth.get("da_gain"),
            "config_gain": row.get("da_gain"),
            "delta_gain_vs_smooth_const": delta(row.get("da_gain"), smooth.get("da_gain")),
        }
        grouped[(row["task"], row["source"], row["target"], row["config"])].append(item)

    rows = []
    order = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    for key, group in sorted(grouped.items(), key=lambda item: (item[0][0], order.get(item[0][3], 999))):
        task, source, target, config = key
        rows.append(
            {
                "task": task,
                "source": source,
                "target": target,
                "config": config,
                "smooth_const_da": mean([row["smooth_const_da"] for row in group]),
                "config_da": mean([row["config_da"] for row in group]),
                "delta_da_vs_smooth_const": mean([row["delta_da_vs_smooth_const"] for row in group]),
                "smooth_const_gain": mean([row["smooth_const_gain"] for row in group]),
                "config_gain": mean([row["config_gain"] for row in group]),
                "delta_gain_vs_smooth_const": mean([row["delta_gain_vs_smooth_const"] for row in group]),
                "positive_seeds": sum(1 for row in group if (row["delta_da_vs_smooth_const"] or 0.0) > 0.0),
                "negative_seeds": sum(1 for row in group if (row["delta_da_vs_smooth_const"] or 0.0) < 0.0),
            }
        )
    return rows


def build_failed(rows):
    return [row for row in rows if row.get("status") != "ok"]


def positive_pattern(relative_rows):
    grouped = defaultdict(list)
    for row in relative_rows:
        grouped[row["config"]].append(row)
    rows = []
    order = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    for config, group in sorted(grouped.items(), key=lambda item: order.get(item[0], 999)):
        rows.append(
            {
                "config": config,
                "positive_tasks": sum(1 for row in group if (row.get("delta_da_vs_smooth_const") or 0.0) > 0.0),
                "negative_tasks": sum(1 for row in group if (row.get("delta_da_vs_smooth_const") or 0.0) < 0.0),
                "positive_seeds": sum(int(row.get("positive_seeds") or 0) for row in group),
                "negative_seeds": sum(int(row.get("negative_seeds") or 0) for row in group),
                "mean_delta_da": mean([row.get("delta_da_vs_smooth_const") for row in group]),
            }
        )
    return rows


def write_summary(root, rows, config_summary, task_summary, relative_rows, lambda_rows, failed):
    positive_rows = positive_pattern(relative_rows)
    lambda_summary = summarize(
        lambda_rows,
        ["config", "schedule_type"],
        ["lambda_value"],
    )
    with (root / "v290_smooth_schedule_probe_summary.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v2.9.0 Smooth Schedule Probe Summary\n\n")
        handle.write("## 1. Experiment Setup\n\n")
        setup = {
            "tasks": len({row["task"] for row in rows}),
            "seeds": len({row["seed"] for row in rows}),
            "configs": len({row["config"] for row in rows}),
            "runs": len(rows),
            "ok": len([row for row in rows if row.get("status") == "ok"]),
            "failed": len(failed),
        }
        write_md_table(handle, [setup], ["tasks", "seeds", "configs", "runs", "ok", "failed"])

        handle.write("## 2. Lambda Schedules\n\n")
        write_md_table(
            handle,
            lambda_summary,
            ["config", "schedule_type", "n", "lambda_value_mean", "lambda_value_std"],
            ["config", "schedule", "epochs", "mean lambda", "lambda std"],
        )

        handle.write("## 3. Config-level Results\n\n")
        write_md_table(
            handle,
            config_summary,
            [
                "config",
                "n",
                "source_self_f1_mean",
                "source_on_target_f1_mean",
                "da_f1_mean",
                "da_gain_mean",
                "da_f1_std",
                "da_gain_std",
                "mean_source_structure_lambda_mean",
                "final_source_structure_lambda_mean",
            ],
            [
                "config",
                "n",
                "source self",
                "source-on-target",
                "DA F1",
                "DA gain",
                "DA std",
                "gain std",
                "mean lambda",
                "final lambda",
            ],
        )

        handle.write("## 4. Task-level Results\n\n")
        write_md_table(
            handle,
            task_summary,
            ["task", "config", "n", "source_on_target_f1_mean", "da_f1_mean", "da_gain_mean", "da_f1_std"],
            ["task", "config", "n", "source-on-target", "DA F1", "DA gain", "DA std"],
        )

        handle.write("## 5. Relative to smooth_const\n\n")
        write_md_table(
            handle,
            relative_rows,
            [
                "task",
                "config",
                "smooth_const_da",
                "config_da",
                "delta_da_vs_smooth_const",
                "positive_seeds",
                "negative_seeds",
            ],
            ["task", "config", "smooth DA", "config DA", "delta DA", "positive seeds", "negative seeds"],
        )

        handle.write("## 6. Positive / Negative Task Pattern\n\n")
        write_md_table(
            handle,
            positive_rows,
            ["config", "mean_delta_da", "positive_tasks", "negative_tasks", "positive_seeds", "negative_seeds"],
        )

        handle.write("## 7. Key Diagnostic Questions\n\n")
        best = max(config_summary, key=lambda row: row.get("da_f1_mean") or float("-inf"), default={})
        smooth = next((row for row in config_summary if row.get("config") == "smooth_const"), {})
        best_delta = delta(best.get("da_f1_mean"), smooth.get("da_f1_mean"))
        handle.write(f"- best_config_by_probe_avg: {best.get('config')} ({fmt_md(best.get('da_f1_mean'))}).\n")
        handle.write(f"- best_delta_vs_smooth_const: {fmt_md(best_delta)}.\n")
        for config in [row["config"] for row in positive_rows]:
            item = next(row for row in positive_rows if row["config"] == config)
            handle.write(
                f"- {config}: positive_tasks={item['positive_tasks']}, "
                f"negative_tasks={item['negative_tasks']}, mean_delta_da={fmt_md(item['mean_delta_da'])}.\n"
            )
        handle.write("\n## 8. Minimal Observations\n\n")
        handle.write("- This file reports probe statistics only.\n")
        handle.write("- No selector, routing, UMSC, elastic radius change, or DA-stage structure loss is used.\n")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    rows = [parse_log(path) for path in sorted(root.glob("gpu*_*.log"))]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lambda_rows = parse_lambda_curve_files(root)
    config_summary = build_config_summary(ok_rows)
    task_summary = build_task_summary(ok_rows)
    relative_rows = build_relative(ok_rows)
    failed = build_failed(rows)

    raw_fields = [
        "task",
        "source",
        "target",
        "seed",
        "config",
        "schedule_type",
        "lambda_base",
        "lambda_final",
        "decay_start_epoch",
        "warmup_epochs",
        "max_epoch",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "final_source_structure_lambda",
        "mean_source_structure_lambda",
        "min_source_structure_lambda",
        "max_source_structure_lambda",
        "source_loss",
        "source_cls_loss",
        "source_compact_loss",
        "source_compact_raw_loss",
        "status",
        "log",
    ]
    write_tsv(root / "raw_strength_rows.tsv", rows, raw_fields)
    write_tsv(
        root / "config_summary.tsv",
        config_summary,
        [
            "config",
            "n",
            "source_self_f1_mean",
            "source_self_f1_std",
            "source_on_target_f1_mean",
            "source_on_target_f1_std",
            "da_f1_mean",
            "da_f1_std",
            "da_gain_mean",
            "da_gain_std",
            "mean_source_structure_lambda_mean",
            "final_source_structure_lambda_mean",
        ],
    )
    write_tsv(
        root / "task_summary.tsv",
        task_summary,
        [
            "task",
            "source",
            "target",
            "config",
            "n",
            "source_self_f1_mean",
            "source_self_f1_std",
            "source_on_target_f1_mean",
            "source_on_target_f1_std",
            "da_f1_mean",
            "da_f1_std",
            "da_gain_mean",
            "da_gain_std",
        ],
    )
    write_tsv(
        root / "relative_to_smooth_const.tsv",
        relative_rows,
        [
            "task",
            "source",
            "target",
            "config",
            "smooth_const_da",
            "config_da",
            "delta_da_vs_smooth_const",
            "smooth_const_gain",
            "config_gain",
            "delta_gain_vs_smooth_const",
            "positive_seeds",
            "negative_seeds",
        ],
    )
    write_tsv(
        root / "schedule_lambda_curves.tsv",
        lambda_rows,
        [
            "config",
            "epoch",
            "lambda_value",
            "schedule_type",
            "lambda_base",
            "lambda_final",
            "decay_start_epoch",
            "warmup_epochs",
            "max_epoch",
        ],
    )
    write_tsv(root / "failed_runs.tsv", failed, raw_fields)
    write_summary(root, rows, config_summary, task_summary, relative_rows, lambda_rows, failed)
    print(f"WROTE|root={root}|runs={len(rows)}|ok={len(ok_rows)}|failed={len(failed)}")


if __name__ == "__main__":
    main()
