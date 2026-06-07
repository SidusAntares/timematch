#!/usr/bin/env python3
import re
import statistics as stats
import sys
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")
WEIGHT_RE = re.compile(r"_w([0-9p]+)")


def compact_weight_from_config(config):
    match = WEIGHT_RE.search(config)
    if not match:
        return 0.0
    return float(match.group(1).replace("p", "."))


def parse_kv_line(line, prefix):
    values = {}
    for part in line[len(prefix):].strip().split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def config_meta(config):
    target = "auto"
    detached = False
    compact = 0.0
    mechanism = config

    if "raw_global" in config:
        target = "raw"
        compact = compact_weight_from_config(config)
    if "reshaped_global" in config:
        target = "reshaped"
        compact = compact_weight_from_config(config)
    if "both_global" in config:
        target = "both"
        compact = compact_weight_from_config(config)
    if "detached" in config:
        detached = True
    if config == "plain":
        mechanism = "none"
    elif config.startswith("raw_global"):
        mechanism = "raw_compact"
    elif config.startswith("strength0"):
        mechanism = "strength0_dualpath"
    elif config.startswith("frozen_s003"):
        mechanism = "frozen_s003"
    elif config.startswith("trainable_s003"):
        mechanism = "trainable_s003"

    if "raw_global" in config and not config.startswith("raw_global"):
        mechanism = mechanism + "_plus_raw"
    if "reshaped_global" in config:
        mechanism = mechanism + "_plus_reshaped"
    if "both_global" in config:
        mechanism = mechanism + "_plus_both"
    if detached:
        mechanism = mechanism + "_detached"

    return {
        "mechanism": mechanism,
        "structure_target": target,
        "detached": detached,
        "compact_weight": compact,
    }


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
    meta = config_meta(config)

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
        "mechanism": meta["mechanism"],
        "structure_target": final_source.get("target", meta["structure_target"]),
        "detached": final_source.get("detached", str(meta["detached"])),
        "compact_weight": meta["compact_weight"],
        "source_self_f1": source_self,
        "source_on_target_f1": source_target,
        "da_f1": da,
        "da_gain": None if da is None or source_target is None else da - source_target,
        "delta_vs_plain": None,
        "source_loss": final_source.get("loss"),
        "source_cls_loss": final_source.get("cls"),
        "source_compact_loss": final_source.get("compact"),
        "source_compact_raw_loss": final_source.get("compact_raw"),
        "source_compact_reshaped_loss": final_source.get("compact_reshaped"),
        "source_spatial_delta": final_source.get("spatial_delta"),
        "source_temporal_delta": final_source.get("temporal_delta"),
        "status": status,
        "log": path.name,
    }


def enrich_deltas(rows):
    plain = {
        (row["task"], row["seed"]): row["da_f1"]
        for row in rows
        if row["status"] == "ok" and row["config"] == "plain" and row["da_f1"] is not None
    }
    for row in rows:
        if row["da_f1"] is None:
            continue
        plain_value = plain.get((row["task"], row["seed"]))
        if plain_value is not None:
            row["delta_vs_plain"] = row["da_f1"] - plain_value


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
                "delta_vs_plain_mean": mean(values(group, "delta_vs_plain")),
                "delta_vs_plain_std": stdev(values(group, "delta_vs_plain")),
                "delta_vs_plain_pos": sum(value > 0 for value in values(group, "delta_vs_plain")),
                "compact_loss_mean": mean(values(group, "source_compact_loss")),
                "compact_raw_loss_mean": mean(values(group, "source_compact_raw_loss")),
                "compact_reshaped_loss_mean": mean(values(group, "source_compact_reshaped_loss")),
                "spatial_delta_mean": mean(values(group, "source_spatial_delta")),
                "temporal_delta_mean": mean(values(group, "source_temporal_delta")),
            }
        )
        summary_rows.append(out)
    return summary_rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_structure_target_stage2b.py LOG_DIR")
    root = Path(sys.argv[1])
    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]
    enrich_deltas(rows)

    fields = [
        "task",
        "seed",
        "config",
        "mechanism",
        "structure_target",
        "detached",
        "compact_weight",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "delta_vs_plain",
        "source_loss",
        "source_cls_loss",
        "source_compact_loss",
        "source_compact_raw_loss",
        "source_compact_reshaped_loss",
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
        "delta_vs_plain_mean",
        "delta_vs_plain_std",
        "delta_vs_plain_pos",
        "compact_loss_mean",
        "compact_raw_loss_mean",
        "compact_reshaped_loss_mean",
        "spatial_delta_mean",
        "temporal_delta_mean",
    ]
    write_tsv(root / "config_summary.tsv", summarize(rows, ["task", "config"]), summary_fields)
    mechanism_fields = [
        "task",
        "mechanism",
        "structure_target",
        "detached",
        "n",
        "ok_count",
        "da_mean",
        "da_std",
        "source_on_target_mean",
        "source_on_target_std",
        "da_gain_mean",
        "delta_vs_plain_mean",
        "delta_vs_plain_std",
        "delta_vs_plain_pos",
        "compact_loss_mean",
        "compact_raw_loss_mean",
        "compact_reshaped_loss_mean",
        "spatial_delta_mean",
        "temporal_delta_mean",
    ]
    write_tsv(
        root / "mechanism_target_summary.tsv",
        summarize(rows, ["task", "mechanism", "structure_target", "detached"]),
        mechanism_fields,
    )

    print("Wrote:", root / "summary.tsv")
    print("Wrote:", root / "config_summary.tsv")
    print("Wrote:", root / "mechanism_target_summary.tsv")


if __name__ == "__main__":
    main()
