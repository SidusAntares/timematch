#!/usr/bin/env python3
import re
import sys
from collections import defaultdict
from pathlib import Path


TEST_RE = re.compile(r"Test result for ([^:]+): accuracy=([0-9.]+), f1=([0-9.]+)")
NAME_RE = re.compile(r"gpu\d+_(.+)_seed(\d+)_(on|off)_")


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    name_match = NAME_RE.search(path.name)
    task = name_match.group(1) if name_match else path.stem
    seed = int(name_match.group(2)) if name_match else -1
    setting = name_match.group(3) if name_match else "unknown"
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
        "setting": setting,
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


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(field)) for field in fields) + "\n")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_paired_lift.py LOG_DIR")
    root = Path(sys.argv[1])
    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]

    summary_fields = [
        "task",
        "seed",
        "setting",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "status",
        "log",
    ]
    write_tsv(root / "summary.tsv", rows, summary_fields)

    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[(row["task"], row["seed"])][row["setting"]] = row

    pair_rows = []
    for (task, seed), pair in sorted(by_pair.items()):
        off = pair.get("off", {})
        on = pair.get("on", {})
        pair_rows.append(
            {
                "task": task,
                "seed": seed,
                "off_da_f1": off.get("da_f1"),
                "on_da_f1": on.get("da_f1"),
                "delta_da_f1": None
                if off.get("da_f1") is None or on.get("da_f1") is None
                else on["da_f1"] - off["da_f1"],
                "off_source_on_target_f1": off.get("source_on_target_f1"),
                "on_source_on_target_f1": on.get("source_on_target_f1"),
                "delta_source_on_target_f1": None
                if off.get("source_on_target_f1") is None
                or on.get("source_on_target_f1") is None
                else on["source_on_target_f1"] - off["source_on_target_f1"],
                "off_da_gain": off.get("da_gain"),
                "on_da_gain": on.get("da_gain"),
                "delta_da_gain": None
                if off.get("da_gain") is None or on.get("da_gain") is None
                else on["da_gain"] - off["da_gain"],
            }
        )

    pair_fields = [
        "task",
        "seed",
        "off_da_f1",
        "on_da_f1",
        "delta_da_f1",
        "off_source_on_target_f1",
        "on_source_on_target_f1",
        "delta_source_on_target_f1",
        "off_da_gain",
        "on_da_gain",
        "delta_da_gain",
    ]
    write_tsv(root / "paired_delta.tsv", pair_rows, pair_fields)

    print("Wrote:", root / "summary.tsv")
    print("Wrote:", root / "paired_delta.tsv")
    for task in sorted({row["task"] for row in pair_rows}):
        deltas = [row["delta_da_f1"] for row in pair_rows if row["task"] == task]
        positives = sum(1 for value in deltas if value is not None and value > 0)
        print(
            f"{task}: delta_mean={fmt(mean(deltas))}, "
            f"delta_std={fmt(std(deltas))}, positive={positives}/{len(deltas)}"
        )


if __name__ == "__main__":
    main()
