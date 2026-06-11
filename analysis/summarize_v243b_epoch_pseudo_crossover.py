#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")
TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")


def parse_kv_line(line, prefix):
    out = {}
    for part in line[len(prefix):].strip().split("|"):
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
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    name_match = NAME_RE.match(path.name)
    task = name_match.group(1) if name_match else path.stem
    seed = int(name_match.group(2)) if name_match else -1
    config = name_match.group(3) if name_match else "unknown"
    tests = TEST_RE.findall(text)
    da_f1 = safe_float(tests[-1][2]) if tests else None

    rows = []
    for line in text.splitlines():
        if not line.startswith("TIMEMATCH_PSEUDO_SUMMARY|"):
            continue
        values = parse_kv_line(line, "TIMEMATCH_PSEUDO_SUMMARY|")
        if values.get("stage") != "train_epoch":
            continue
        rows.append(
            {
                "task": task,
                "seed": seed,
                "config": config,
                "epoch": safe_int(values.get("epoch")),
                "shift": safe_int(values.get("shift")),
                "masked_f1": safe_float(values.get("masked_f1")),
                "all_f1": safe_float(values.get("all_f1")),
                "coverage": safe_float(values.get("coverage")),
                "mean_conf": safe_float(values.get("mean_conf")),
                "da_f1": da_f1,
                "log": path.name,
            }
        )
    return rows


def read_curves(log_dir, task_filter):
    rows = []
    for path in sorted(Path(log_dir).glob("*.log")):
        parsed = parse_log(path)
        if task_filter:
            parsed = [row for row in parsed if row["task"] == task_filter]
        rows.extend(parsed)
    return rows


def build_contrasts(rows, base_config, shaped_config):
    by_key = {
        (row["task"], str(row["seed"]), row["config"], row["epoch"]): row
        for row in rows
        if row["epoch"] is not None
    }
    keys = sorted(
        {
            (task, seed, epoch)
            for task, seed, config, epoch in by_key
            if config == base_config
        },
        key=lambda item: (item[0], int(item[1]), item[2]),
    )
    contrasts = []
    for task, seed, epoch in keys:
        base = by_key.get((task, seed, base_config, epoch))
        shaped = by_key.get((task, seed, shaped_config, epoch))
        if base is None or shaped is None:
            continue
        contrasts.append(
            {
                "task": task,
                "seed": seed,
                "epoch": epoch,
                "base_config": base_config,
                "shaped_config": shaped_config,
                "base_masked_f1": base["masked_f1"],
                "shaped_masked_f1": shaped["masked_f1"],
                "delta_masked_f1": delta(shaped["masked_f1"], base["masked_f1"]),
                "base_all_f1": base["all_f1"],
                "shaped_all_f1": shaped["all_f1"],
                "delta_all_f1": delta(shaped["all_f1"], base["all_f1"]),
                "base_coverage": base["coverage"],
                "shaped_coverage": shaped["coverage"],
                "delta_coverage": delta(shaped["coverage"], base["coverage"]),
                "base_shift": base["shift"],
                "shaped_shift": shaped["shift"],
                "base_da_f1": base["da_f1"],
                "shaped_da_f1": shaped["da_f1"],
                "delta_da_f1": delta(shaped["da_f1"], base["da_f1"]),
            }
        )
    return contrasts


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def first_epoch(rows, field, predicate):
    for row in sorted(rows, key=lambda item: item["epoch"]):
        value = row.get(field)
        if value is not None and predicate(value):
            return row["epoch"]
    return None


def summarize_crossovers(contrasts, early_epoch):
    grouped = {}
    for row in contrasts:
        grouped.setdefault((row["task"], row["seed"]), []).append(row)
    out = []
    for (task, seed), rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        last = sorted(rows, key=lambda item: item["epoch"])[-1]
        first_masked_positive = first_epoch(rows, "delta_masked_f1", lambda value: value > 0.0)
        first_all_positive = first_epoch(rows, "delta_all_f1", lambda value: value > 0.0)
        first_coverage_positive = first_epoch(rows, "delta_coverage", lambda value: value > 0.0)
        if first_masked_positive is None:
            masked_recovery_label = "never"
        elif first_masked_positive <= early_epoch:
            masked_recovery_label = "early"
        else:
            masked_recovery_label = "late"

        out.append(
            {
                "task": task,
                "seed": seed,
                "delta_da_f1": last["delta_da_f1"],
                "first_masked_positive_epoch": first_masked_positive,
                "first_all_positive_epoch": first_all_positive,
                "first_coverage_positive_epoch": first_coverage_positive,
                "masked_recovery_label": masked_recovery_label,
                "last_delta_masked_f1": last["delta_masked_f1"],
                "last_delta_all_f1": last["delta_all_f1"],
                "last_delta_coverage": last["delta_coverage"],
                "best_delta_masked_f1": max(
                    (row["delta_masked_f1"] for row in rows if row["delta_masked_f1"] is not None),
                    default=None,
                ),
                "best_delta_all_f1": max(
                    (row["delta_all_f1"] for row in rows if row["delta_all_f1"] is not None),
                    default=None,
                ),
                "late_recovery_supported": int(
                    (last["delta_da_f1"] or 0.0) > 0.0
                    and first_masked_positive is not None
                    and first_masked_positive > early_epoch
                ),
                "late_recovery_refuted": int(
                    (last["delta_da_f1"] or 0.0) > 0.0
                    and first_masked_positive is None
                ),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize epoch-wise pseudo-label crossover for v2.4.3b readiness logs."
    )
    parser.add_argument("log_dir")
    parser.add_argument("--task", default="FR1_to_AT1")
    parser.add_argument("--base_config", default="plain")
    parser.add_argument("--shaped_config", default="raw_global_w1_source_only")
    parser.add_argument("--early_epoch", type=int, default=3)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    curves = read_curves(log_dir, args.task)
    contrasts = build_contrasts(curves, args.base_config, args.shaped_config)
    summary = summarize_crossovers(contrasts, args.early_epoch)

    curve_fields = [
        "task",
        "seed",
        "config",
        "epoch",
        "shift",
        "masked_f1",
        "all_f1",
        "coverage",
        "mean_conf",
        "da_f1",
        "log",
    ]
    contrast_fields = [
        "task",
        "seed",
        "epoch",
        "base_config",
        "shaped_config",
        "base_masked_f1",
        "shaped_masked_f1",
        "delta_masked_f1",
        "base_all_f1",
        "shaped_all_f1",
        "delta_all_f1",
        "base_coverage",
        "shaped_coverage",
        "delta_coverage",
        "base_shift",
        "shaped_shift",
        "base_da_f1",
        "shaped_da_f1",
        "delta_da_f1",
    ]
    summary_fields = [
        "task",
        "seed",
        "delta_da_f1",
        "first_masked_positive_epoch",
        "first_all_positive_epoch",
        "first_coverage_positive_epoch",
        "masked_recovery_label",
        "last_delta_masked_f1",
        "last_delta_all_f1",
        "last_delta_coverage",
        "best_delta_masked_f1",
        "best_delta_all_f1",
        "late_recovery_supported",
        "late_recovery_refuted",
    ]
    suffix = args.task if args.task else "all"
    write_tsv(log_dir / f"epoch_pseudo_curves_{suffix}.tsv", curves, curve_fields)
    write_tsv(log_dir / f"epoch_pseudo_contrasts_{suffix}.tsv", contrasts, contrast_fields)
    write_tsv(log_dir / f"epoch_pseudo_crossover_{suffix}.tsv", summary, summary_fields)
    print("Wrote:", log_dir / f"epoch_pseudo_curves_{suffix}.tsv")
    print("Wrote:", log_dir / f"epoch_pseudo_contrasts_{suffix}.tsv")
    print("Wrote:", log_dir / f"epoch_pseudo_crossover_{suffix}.tsv")


if __name__ == "__main__":
    main()
