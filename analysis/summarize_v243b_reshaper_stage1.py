#!/usr/bin/env python3
import re
import statistics as stats
import sys
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")


def parse_kv_line(line, prefix):
    values = {}
    for part in line[len(prefix):].strip().split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def config_family(config):
    if config.startswith("frozen_s003"):
        return "frozen_s003"
    if config.startswith("frozen_s010"):
        return "frozen_s010"
    return config


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    source_summaries = [
        parse_kv_line(line, "SOURCE_EPOCH_SUMMARY|")
        for line in text.splitlines()
        if line.startswith("SOURCE_EPOCH_SUMMARY|")
    ]
    final_source = source_summaries[-1] if source_summaries else {}
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
        "family": config_family(config),
        "source_self_f1": source_self,
        "source_on_target_f1": source_target,
        "da_f1": da,
        "da_gain": None if da is None or source_target is None else da - source_target,
        "source_loss": final_source.get("loss"),
        "source_cls_loss": final_source.get("cls"),
        "source_spatial_delta": final_source.get("spatial_delta"),
        "source_temporal_delta": final_source.get("temporal_delta"),
        "status": status,
        "log": path.name,
    }


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


def values(rows, field):
    out = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        out.append(float(value))
    return out


def mean(items):
    return None if not items else sum(items) / len(items)


def stdev(items):
    return None if len(items) < 2 else stats.stdev(items)


def summarize(rows, group_fields):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for key, group in sorted(grouped.items()):
        out = {field: value for field, value in zip(group_fields, key)}
        out.update(
            {
                "n": len(group),
                "ok_count": sum(1 for row in group if row["status"] == "ok"),
                "da_mean": mean(values(group, "da_f1")),
                "da_std": stdev(values(group, "da_f1")),
                "source_on_target_mean": mean(values(group, "source_on_target_f1")),
                "source_on_target_std": stdev(values(group, "source_on_target_f1")),
                "da_gain_mean": mean(values(group, "da_gain")),
                "spatial_delta_mean": mean(values(group, "source_spatial_delta")),
                "spatial_delta_std": stdev(values(group, "source_spatial_delta")),
                "temporal_delta_mean": mean(values(group, "source_temporal_delta")),
                "temporal_delta_std": stdev(values(group, "source_temporal_delta")),
            }
        )
        summary_rows.append(out)
    return summary_rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_reshaper_stage1.py LOG_DIR")
    root = Path(sys.argv[1])
    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]

    fields = [
        "task",
        "seed",
        "config",
        "family",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "source_loss",
        "source_cls_loss",
        "source_spatial_delta",
        "source_temporal_delta",
        "status",
        "log",
    ]
    write_tsv(root / "summary.tsv", rows, fields)

    summary_fields = [
        "task",
        "config",
        "n",
        "ok_count",
        "da_mean",
        "da_std",
        "source_on_target_mean",
        "source_on_target_std",
        "da_gain_mean",
        "spatial_delta_mean",
        "spatial_delta_std",
        "temporal_delta_mean",
        "temporal_delta_std",
    ]
    write_tsv(root / "config_summary.tsv", summarize(rows, ["task", "config"]), summary_fields)

    family_fields = [
        "task",
        "family",
        "n",
        "ok_count",
        "da_mean",
        "da_std",
        "source_on_target_mean",
        "source_on_target_std",
        "da_gain_mean",
        "spatial_delta_mean",
        "spatial_delta_std",
        "temporal_delta_mean",
        "temporal_delta_std",
    ]
    write_tsv(root / "family_summary.tsv", summarize(rows, ["task", "family"]), family_fields)

    print("Wrote:", root / "summary.tsv")
    print("Wrote:", root / "config_summary.tsv")
    print("Wrote:", root / "family_summary.tsv")


if __name__ == "__main__":
    main()
