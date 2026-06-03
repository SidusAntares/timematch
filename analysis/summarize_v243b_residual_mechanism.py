#!/usr/bin/env python3
import re
import statistics as stats
import sys
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(.+)\.log$")
SOURCE_SUMMARY_RE = re.compile(
    r"SOURCE_EPOCH_SUMMARY\|epoch=(\d+)\|loss=([0-9.eE+-]+)\|cls=([0-9.eE+-]+)\|"
    r"compact=([0-9.eE+-]+)\|reshaper=([0-9.eE+-]+)\|dualcls=([0-9.eE+-]+)\|dualrel=([0-9.eE+-]+)"
)


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    source_summaries = SOURCE_SUMMARY_RE.findall(text)
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

    final_source = None
    if source_summaries:
        epoch, loss, cls, compact, reshaper, dualcls, dualrel = source_summaries[-1]
        final_source = {
            "source_epoch": int(epoch),
            "source_loss": float(loss),
            "source_cls_loss": float(cls),
            "source_compact_loss": float(compact),
            "source_reshaper_loss": float(reshaper),
            "source_dualcls_loss": float(dualcls),
            "source_dualrel_loss": float(dualrel),
        }
    else:
        final_source = {
            "source_epoch": None,
            "source_loss": None,
            "source_cls_loss": None,
            "source_compact_loss": None,
            "source_reshaper_loss": None,
            "source_dualcls_loss": None,
            "source_dualrel_loss": None,
        }

    row = {
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
    row.update(final_source)
    return row


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
    return [row[field] for row in rows if row.get(field) is not None]


def mean(items):
    return None if not items else sum(items) / len(items)


def stdev(items):
    return None if len(items) < 2 else stats.stdev(items)


def grouped_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["task"], row["config"]), []).append(row)
    return grouped


def mean_for(grouped, task, config, field):
    return mean(values(grouped.get((task, config), []), field))


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def factorial_effect_rows(rows):
    grouped = grouped_rows(rows)
    tasks = sorted({row["task"] for row in rows})
    effect_rows = []
    for task in tasks:
        plain = mean_for(grouped, task, "plain", "da_f1")
        reshaper = mean_for(grouped, task, "reshaper_only", "da_f1")
        global_compact = mean_for(grouped, task, "global_compact", "da_f1")
        reshaper_global = mean_for(grouped, task, "reshaper_global_compact", "da_f1")
        local_compact = mean_for(grouped, task, "local_compact", "da_f1")
        reshaper_local = mean_for(grouped, task, "reshaper_local_compact", "da_f1")

        interaction_global = None
        if None not in (plain, reshaper, global_compact, reshaper_global):
            interaction_global = reshaper_global - reshaper - global_compact + plain

        effect_rows.extend(
            [
                {
                    "task": task,
                    "effect": "reshaper_main_without_compact",
                    "left": "reshaper_only",
                    "right": "plain",
                    "delta": delta(reshaper, plain),
                },
                {
                    "task": task,
                    "effect": "global_compact_main_without_reshaper",
                    "left": "global_compact",
                    "right": "plain",
                    "delta": delta(global_compact, plain),
                },
                {
                    "task": task,
                    "effect": "reshaper_global_compact_vs_reshaper_only",
                    "left": "reshaper_global_compact",
                    "right": "reshaper_only",
                    "delta": delta(reshaper_global, reshaper),
                },
                {
                    "task": task,
                    "effect": "global_interaction",
                    "left": "reshaper_global_compact",
                    "right": "reshaper + global_compact - plain",
                    "delta": interaction_global,
                },
                {
                    "task": task,
                    "effect": "local_vs_global_without_reshaper",
                    "left": "local_compact",
                    "right": "global_compact",
                    "delta": delta(local_compact, global_compact),
                },
                {
                    "task": task,
                    "effect": "local_vs_global_with_reshaper",
                    "left": "reshaper_local_compact",
                    "right": "reshaper_global_compact",
                    "delta": delta(reshaper_local, reshaper_global),
                },
            ]
        )
    return effect_rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_residual_mechanism.py LOG_DIR")
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
        "source_epoch",
        "source_loss",
        "source_cls_loss",
        "source_compact_loss",
        "source_reshaper_loss",
        "source_dualcls_loss",
        "source_dualrel_loss",
        "status",
        "log",
    ]
    write_tsv(root / "summary.tsv", rows, fields)

    grouped = grouped_rows(rows)
    config_rows = []
    for (task, config), group in sorted(grouped.items()):
        config_rows.append(
            {
                "task": task,
                "config": config,
                "n": len(group),
                "ok_count": sum(1 for row in group if row["status"] == "ok"),
                "da_mean": mean(values(group, "da_f1")),
                "da_std": stdev(values(group, "da_f1")),
                "source_on_target_mean": mean(values(group, "source_on_target_f1")),
                "source_on_target_std": stdev(values(group, "source_on_target_f1")),
                "da_gain_mean": mean(values(group, "da_gain")),
                "source_compact_loss_mean": mean(values(group, "source_compact_loss")),
                "source_reshaper_loss_mean": mean(values(group, "source_reshaper_loss")),
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
        "source_compact_loss_mean",
        "source_reshaper_loss_mean",
    ]
    write_tsv(root / "config_summary.tsv", config_rows, config_fields)

    effect_rows = factorial_effect_rows(rows)
    effect_fields = ["task", "effect", "left", "right", "delta"]
    write_tsv(root / "factorial_effects.tsv", effect_rows, effect_fields)

    print("Wrote:", root / "summary.tsv")
    print("Wrote:", root / "config_summary.tsv")
    print("Wrote:", root / "factorial_effects.tsv")
    for row in effect_rows:
        if row["effect"] in {
            "reshaper_main_without_compact",
            "global_compact_main_without_reshaper",
            "local_vs_global_with_reshaper",
        }:
            print(f"{row['task']} {row['effect']}: delta={fmt(row['delta'])}")


if __name__ == "__main__":
    main()
