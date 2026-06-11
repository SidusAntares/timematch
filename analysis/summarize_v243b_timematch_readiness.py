#!/usr/bin/env python3
import re
import statistics as stats
import sys
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")


def parse_kv_line(line, prefix):
    out = {}
    text = line[len(prefix):]
    for part in text.strip().split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key] = value
    return out


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_int(value):
    if value is None or value == "":
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


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(fmt(row.get(field)) for field in fields) + "\n")


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def stdev(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return stats.stdev(values)


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    pseudo_rows = [
        parse_kv_line(line, "TIMEMATCH_PSEUDO_SUMMARY|")
        for line in text.splitlines()
        if line.startswith("TIMEMATCH_PSEUDO_SUMMARY|")
    ]
    epoch_rows = [
        parse_kv_line(line, "TIMEMATCH_EPOCH_SUMMARY|")
        for line in text.splitlines()
        if line.startswith("TIMEMATCH_EPOCH_SUMMARY|")
    ]
    name_match = NAME_RE.match(path.name)
    task = name_match.group(1) if name_match else path.stem
    seed = int(name_match.group(2)) if name_match else -1
    config = name_match.group(3) if name_match else "unknown"

    status = "ok"
    if "Traceback" in text or "error:" in text.lower():
        status = "error"
    if not tests:
        status = "incomplete"

    source_self_f1 = None
    source_on_target_f1 = None
    da_f1 = None
    if len(tests) >= 3:
        source_self_f1 = safe_float(tests[-3][2])
        source_on_target_f1 = safe_float(tests[-2][2])
        da_f1 = safe_float(tests[-1][2])
    elif len(tests) >= 2:
        source_on_target_f1 = safe_float(tests[-2][2])
        da_f1 = safe_float(tests[-1][2])

    initial = next((row for row in pseudo_rows if row.get("stage") == "initial"), {})
    train_rows = [row for row in pseudo_rows if row.get("stage") == "train_epoch"]
    epoch1 = train_rows[0] if train_rows else {}
    last = train_rows[-1] if train_rows else {}
    best_masked = max((safe_float(row.get("masked_f1")) or 0.0 for row in train_rows), default=None)
    best_all = max((safe_float(row.get("all_f1")) or 0.0 for row in train_rows), default=None)

    return {
        "task": task,
        "seed": seed,
        "config": config,
        "source_self_f1": source_self_f1,
        "source_on_target_f1": source_on_target_f1,
        "da_f1": da_f1,
        "initial_shift": safe_int(initial.get("shift")),
        "initial_all_f1": safe_float(initial.get("all_f1")),
        "initial_all_acc": safe_float(initial.get("all_acc")),
        "initial_masked_f1": safe_float(initial.get("masked_f1")),
        "initial_masked_acc": safe_float(initial.get("masked_acc")),
        "initial_coverage": safe_float(initial.get("coverage")),
        "initial_mean_conf": safe_float(initial.get("mean_conf")),
        "epoch1_shift": safe_int(epoch1.get("shift")),
        "epoch1_all_f1": safe_float(epoch1.get("all_f1")),
        "epoch1_masked_f1": safe_float(epoch1.get("masked_f1")),
        "epoch1_coverage": safe_float(epoch1.get("coverage")),
        "epoch1_mean_conf": safe_float(epoch1.get("mean_conf")),
        "last_shift": safe_int(last.get("shift")),
        "last_all_f1": safe_float(last.get("all_f1")),
        "last_masked_f1": safe_float(last.get("masked_f1")),
        "last_coverage": safe_float(last.get("coverage")),
        "last_mean_conf": safe_float(last.get("mean_conf")),
        "best_epoch_all_f1": best_all,
        "best_epoch_masked_f1": best_masked,
        "epoch_summary_count": len(epoch_rows),
        "pseudo_summary_count": len(pseudo_rows),
        "status": status,
        "log": path.name,
    }


def diff(left, right):
    if left is None or right is None:
        return None
    return left - right


def build_contrasts(rows, base_config, shaped_config):
    by_key = {
        (row["task"], str(row["seed"]), row["config"]): row
        for row in rows
        if row["status"] == "ok"
    }
    task_seeds = sorted({(task, seed) for task, seed, _ in by_key})
    fields = [
        "source_on_target_f1",
        "da_f1",
        "initial_all_f1",
        "initial_masked_f1",
        "initial_coverage",
        "initial_mean_conf",
        "epoch1_all_f1",
        "epoch1_masked_f1",
        "epoch1_coverage",
        "last_all_f1",
        "last_masked_f1",
        "last_coverage",
        "best_epoch_all_f1",
        "best_epoch_masked_f1",
    ]
    out = []
    for task, seed in task_seeds:
        base = by_key.get((task, seed, base_config))
        shaped = by_key.get((task, seed, shaped_config))
        if base is None or shaped is None:
            continue
        row = {
            "task": task,
            "seed": seed,
            "base_config": base_config,
            "shaped_config": shaped_config,
            "base_da_f1": base["da_f1"],
            "shaped_da_f1": shaped["da_f1"],
            "delta_da_f1": diff(shaped["da_f1"], base["da_f1"]),
            "base_initial_shift": base["initial_shift"],
            "shaped_initial_shift": shaped["initial_shift"],
            "base_last_shift": base["last_shift"],
            "shaped_last_shift": shaped["last_shift"],
        }
        for field in fields:
            row[f"base_{field}"] = base[field]
            row[f"shaped_{field}"] = shaped[field]
            row[f"delta_{field}"] = diff(shaped[field], base[field])
        out.append(row)
    return out


def summarize_contrasts(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    summary = []
    fields = [
        "delta_da_f1",
        "delta_source_on_target_f1",
        "delta_initial_all_f1",
        "delta_initial_masked_f1",
        "delta_initial_coverage",
        "delta_epoch1_all_f1",
        "delta_epoch1_masked_f1",
        "delta_last_all_f1",
        "delta_last_masked_f1",
        "delta_best_epoch_all_f1",
        "delta_best_epoch_masked_f1",
    ]
    for task, group in sorted(grouped.items()):
        out = {"task": task, "n": len(group)}
        for field in fields:
            vals = [row.get(field) for row in group]
            out[f"{field}_mean"] = mean(vals)
            out[f"{field}_std"] = stdev(vals)
            out[f"{field}_pos"] = sum(value is not None and value > 0 for value in vals)
        summary.append(out)
    return summary


def main():
    if len(sys.argv) not in {2, 4}:
        raise SystemExit(
            "Usage: summarize_v243b_timematch_readiness.py LOG_DIR [BASE_CONFIG SHAPED_CONFIG]"
        )
    root = Path(sys.argv[1])
    base_config = sys.argv[2] if len(sys.argv) == 4 else "plain"
    shaped_config = sys.argv[3] if len(sys.argv) == 4 else "raw_global_w1_source_only"

    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]
    detail_fields = [
        "task",
        "seed",
        "config",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "initial_shift",
        "initial_all_f1",
        "initial_all_acc",
        "initial_masked_f1",
        "initial_masked_acc",
        "initial_coverage",
        "initial_mean_conf",
        "epoch1_shift",
        "epoch1_all_f1",
        "epoch1_masked_f1",
        "epoch1_coverage",
        "epoch1_mean_conf",
        "last_shift",
        "last_all_f1",
        "last_masked_f1",
        "last_coverage",
        "last_mean_conf",
        "best_epoch_all_f1",
        "best_epoch_masked_f1",
        "epoch_summary_count",
        "pseudo_summary_count",
        "status",
        "log",
    ]
    write_tsv(root / "timematch_readiness.tsv", rows, detail_fields)

    contrasts = build_contrasts(rows, base_config, shaped_config)
    contrast_fields = list(contrasts[0].keys()) if contrasts else []
    write_tsv(root / "timematch_readiness_contrasts.tsv", contrasts, contrast_fields)

    summary = summarize_contrasts(contrasts)
    summary_fields = list(summary[0].keys()) if summary else []
    write_tsv(root / "timematch_readiness_summary.tsv", summary, summary_fields)

    print("Wrote:", root / "timematch_readiness.tsv")
    print("Wrote:", root / "timematch_readiness_contrasts.tsv")
    print("Wrote:", root / "timematch_readiness_summary.tsv")


if __name__ == "__main__":
    main()
