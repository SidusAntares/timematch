#!/usr/bin/env python3
import re
import statistics as stats
import sys
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")
WEIGHT_RE = re.compile(r"_w([0-9p]+)_")


def parse_kv_line(line, prefix):
    values = {}
    for part in line[len(prefix):].strip().split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


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


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(fmt(row.get(field)) for field in fields) + "\n")


def mean(items):
    items = [item for item in items if item is not None]
    return None if not items else sum(items) / len(items)


def stdev(items):
    items = [item for item in items if item is not None]
    return None if len(items) < 2 else stats.stdev(items)


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def compact_weight_from_config(config):
    if config == "plain":
        return 0.0
    match = WEIGHT_RE.search(config)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    source_rows = [
        parse_kv_line(line, "SOURCE_EPOCH_SUMMARY|")
        for line in text.splitlines()
        if line.startswith("SOURCE_EPOCH_SUMMARY|")
    ]
    pseudo_rows = [
        parse_kv_line(line, "TIMEMATCH_PSEUDO_SUMMARY|")
        for line in text.splitlines()
        if line.startswith("TIMEMATCH_PSEUDO_SUMMARY|")
    ]

    name_match = NAME_RE.match(path.name)
    task = name_match.group(1) if name_match else path.stem
    seed = int(name_match.group(2)) if name_match else -1
    config = name_match.group(3) if name_match else "unknown"

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

    final_source = source_rows[-1] if source_rows else {}
    initial = next((row for row in pseudo_rows if row.get("stage") == "initial"), {})
    train_rows = [row for row in pseudo_rows if row.get("stage") == "train_epoch"]
    epoch1 = train_rows[0] if train_rows else {}
    last = train_rows[-1] if train_rows else {}

    best_all = max((safe_float(row.get("all_f1")) or 0.0 for row in train_rows), default=None)
    best_masked = max((safe_float(row.get("masked_f1")) or 0.0 for row in train_rows), default=None)
    best_coverage = max((safe_float(row.get("coverage")) or 0.0 for row in train_rows), default=None)

    return {
        "task": task,
        "seed": seed,
        "config": config,
        "compact_weight": compact_weight_from_config(config),
        "source_self_f1": source_self_f1,
        "source_on_target_f1": source_on_target_f1,
        "da_f1": da_f1,
        "da_gain": delta(da_f1, source_on_target_f1),
        "source_loss": safe_float(final_source.get("loss")),
        "source_cls_loss": safe_float(final_source.get("cls")),
        "source_compact_loss": safe_float(final_source.get("compact")),
        "source_compact_raw_loss": safe_float(final_source.get("compact_raw")),
        "source_spatial_delta": safe_float(final_source.get("spatial_delta")),
        "source_temporal_delta": safe_float(final_source.get("temporal_delta")),
        "initial_shift": safe_int(initial.get("shift")),
        "initial_all_f1": safe_float(initial.get("all_f1")),
        "initial_masked_f1": safe_float(initial.get("masked_f1")),
        "initial_coverage": safe_float(initial.get("coverage")),
        "initial_mean_conf": safe_float(initial.get("mean_conf")),
        "epoch1_all_f1": safe_float(epoch1.get("all_f1")),
        "epoch1_masked_f1": safe_float(epoch1.get("masked_f1")),
        "epoch1_coverage": safe_float(epoch1.get("coverage")),
        "last_shift": safe_int(last.get("shift")),
        "last_all_f1": safe_float(last.get("all_f1")),
        "last_masked_f1": safe_float(last.get("masked_f1")),
        "last_coverage": safe_float(last.get("coverage")),
        "last_mean_conf": safe_float(last.get("mean_conf")),
        "best_epoch_all_f1": best_all,
        "best_epoch_masked_f1": best_masked,
        "best_epoch_coverage": best_coverage,
        "pseudo_summary_count": len(pseudo_rows),
        "status": status,
        "log": path.name,
    }


def build_deltas(rows):
    base = {
        (row["task"], row["seed"]): row
        for row in rows
        if row["status"] == "ok" and row["config"] == "plain"
    }
    fields = [
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "source_compact_loss",
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
        "last_mean_conf",
        "best_epoch_all_f1",
        "best_epoch_masked_f1",
        "best_epoch_coverage",
    ]
    out = []
    for row in rows:
        if row["status"] != "ok" or row["config"] == "plain":
            continue
        plain = base.get((row["task"], row["seed"]))
        if plain is None:
            continue
        contrast = {
            "task": row["task"],
            "seed": row["seed"],
            "config": row["config"],
            "compact_weight": row["compact_weight"],
            "plain_da_f1": plain["da_f1"],
            "config_da_f1": row["da_f1"],
            "delta_da_f1": delta(row["da_f1"], plain["da_f1"]),
        }
        for field in fields:
            contrast[f"plain_{field}"] = plain.get(field)
            contrast[f"config_{field}"] = row.get(field)
            contrast[f"delta_{field}"] = delta(row.get(field), plain.get(field))
        out.append(contrast)
    return out


def summarize(rows, group_fields, value_fields):
    grouped = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    out_rows = []
    for key, group in sorted(grouped.items()):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n"] = len(group)
        for field in value_fields:
            vals = [row.get(field) for row in group]
            vals = [value for value in vals if value is not None]
            out[f"{field}_mean"] = mean(vals)
            out[f"{field}_std"] = stdev(vals)
            out[f"{field}_pos"] = sum(value > 0 for value in vals)
        out_rows.append(out)
    return out_rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_raw_strength_intervention.py LOG_DIR")
    root = Path(sys.argv[1])
    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]

    detail_fields = [
        "task",
        "seed",
        "config",
        "compact_weight",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "source_loss",
        "source_cls_loss",
        "source_compact_loss",
        "source_compact_raw_loss",
        "source_spatial_delta",
        "source_temporal_delta",
        "initial_shift",
        "initial_all_f1",
        "initial_masked_f1",
        "initial_coverage",
        "initial_mean_conf",
        "epoch1_all_f1",
        "epoch1_masked_f1",
        "epoch1_coverage",
        "last_shift",
        "last_all_f1",
        "last_masked_f1",
        "last_coverage",
        "last_mean_conf",
        "best_epoch_all_f1",
        "best_epoch_masked_f1",
        "best_epoch_coverage",
        "pseudo_summary_count",
        "status",
        "log",
    ]
    write_tsv(root / "raw_strength_rows.tsv", rows, detail_fields)

    deltas = build_deltas(rows)
    delta_fields = list(deltas[0].keys()) if deltas else []
    write_tsv(root / "raw_strength_delta_vs_plain.tsv", deltas, delta_fields)

    summary_fields = [
        "delta_da_f1",
        "delta_source_on_target_f1",
        "delta_da_gain",
        "delta_initial_all_f1",
        "delta_initial_masked_f1",
        "delta_initial_coverage",
        "delta_epoch1_all_f1",
        "delta_epoch1_masked_f1",
        "delta_epoch1_coverage",
        "delta_last_all_f1",
        "delta_last_masked_f1",
        "delta_last_coverage",
        "delta_best_epoch_all_f1",
        "delta_best_epoch_masked_f1",
        "delta_best_epoch_coverage",
    ]
    write_tsv(
        root / "raw_strength_summary_by_task.tsv",
        summarize(deltas, ["task", "config", "compact_weight"], summary_fields),
        ["task", "config", "compact_weight", "n"]
        + [item for field in summary_fields for item in (f"{field}_mean", f"{field}_std", f"{field}_pos")],
    )
    write_tsv(
        root / "raw_strength_summary_overall.tsv",
        summarize(deltas, ["config", "compact_weight"], summary_fields),
        ["config", "compact_weight", "n"]
        + [item for field in summary_fields for item in (f"{field}_mean", f"{field}_std", f"{field}_pos")],
    )

    curve_fields = [
        "delta_da_f1",
        "delta_source_on_target_f1",
        "delta_da_gain",
        "delta_initial_all_f1",
        "delta_initial_masked_f1",
        "delta_initial_coverage",
        "delta_initial_mean_conf",
        "delta_last_all_f1",
        "delta_last_masked_f1",
        "delta_last_coverage",
        "delta_last_mean_conf",
        "delta_best_epoch_all_f1",
        "delta_best_epoch_masked_f1",
        "delta_best_epoch_coverage",
    ]
    write_tsv(
        root / "strength_effect_curve.tsv",
        summarize(deltas, ["task", "compact_weight", "config"], curve_fields),
        ["task", "compact_weight", "config", "n"]
        + [item for field in curve_fields for item in (f"{field}_mean", f"{field}_std", f"{field}_pos")],
    )

    pseudo_fields = [
        "task",
        "seed",
        "config",
        "compact_weight",
        "delta_da_f1",
        "delta_source_on_target_f1",
        "delta_da_gain",
        "delta_initial_all_f1",
        "delta_initial_masked_f1",
        "delta_initial_coverage",
        "delta_initial_mean_conf",
        "delta_epoch1_all_f1",
        "delta_epoch1_masked_f1",
        "delta_epoch1_coverage",
        "delta_last_all_f1",
        "delta_last_masked_f1",
        "delta_last_coverage",
        "delta_last_mean_conf",
        "delta_best_epoch_all_f1",
        "delta_best_epoch_masked_f1",
        "delta_best_epoch_coverage",
    ]
    write_tsv(root / "pseudolabel_divergence.tsv", deltas, pseudo_fields)

    print("Wrote:", root / "raw_strength_rows.tsv")
    print("Wrote:", root / "raw_strength_delta_vs_plain.tsv")
    print("Wrote:", root / "raw_strength_summary_by_task.tsv")
    print("Wrote:", root / "raw_strength_summary_overall.tsv")
    print("Wrote:", root / "strength_effect_curve.tsv")
    print("Wrote:", root / "pseudolabel_divergence.tsv")


if __name__ == "__main__":
    main()
