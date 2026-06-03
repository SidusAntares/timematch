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


def to_float(value):
    if value in (None, ""):
        return None
    return float(value)


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

    grad_rows = []
    cos_rows = []
    for line in text.splitlines():
        if line.startswith("SOURCE_GRAD_DIAG|"):
            row = parse_kv_line(line, "SOURCE_GRAD_DIAG|")
            row.update({"task": task, "seed": seed, "config": config, "log": path.name})
            grad_rows.append(row)
        elif line.startswith("SOURCE_GRAD_COS|"):
            row = parse_kv_line(line, "SOURCE_GRAD_COS|")
            row.update({"task": task, "seed": seed, "config": config, "log": path.name})
            cos_rows.append(row)

    summary = {
        "task": task,
        "seed": seed,
        "config": config,
        "source_self_f1": source_self,
        "source_on_target_f1": source_target,
        "da_f1": da,
        "da_gain": None if da is None or source_target is None else da - source_target,
        "grad_diag_count": len(grad_rows),
        "grad_cos_count": len(cos_rows),
        "status": status,
        "log": path.name,
    }
    return summary, grad_rows, cos_rows


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


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_grad_diagnostic.py LOG_DIR")
    root = Path(sys.argv[1])
    summaries = []
    grad_rows = []
    cos_rows = []
    for path in sorted(root.glob("*.log")):
        summary, grad, cos = parse_log(path)
        summaries.append(summary)
        grad_rows.extend(grad)
        cos_rows.extend(cos)

    summary_fields = [
        "task",
        "seed",
        "config",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "grad_diag_count",
        "grad_cos_count",
        "status",
        "log",
    ]
    write_tsv(root / "summary.tsv", summaries, summary_fields)

    grad_fields = [
        "task",
        "seed",
        "config",
        "global_step",
        "epoch",
        "batch_step",
        "loss",
        "group",
        "norm",
        "active",
        "log",
    ]
    write_tsv(root / "grad_norms.tsv", grad_rows, grad_fields)

    cos_fields = [
        "task",
        "seed",
        "config",
        "global_step",
        "epoch",
        "batch_step",
        "left",
        "right",
        "group",
        "left_norm",
        "right_norm",
        "ratio",
        "cosine",
        "valid",
        "log",
    ]
    write_tsv(root / "grad_cosines.tsv", cos_rows, cos_fields)

    grouped = {}
    for row in summaries:
        grouped.setdefault((row["task"], row["config"]), []).append(row)
    config_rows = []
    for (task, config), rows in sorted(grouped.items()):
        config_rows.append(
            {
                "task": task,
                "config": config,
                "n": len(rows),
                "ok_count": sum(1 for row in rows if row["status"] == "ok"),
                "da_mean": mean(values(rows, "da_f1")),
                "da_std": stdev(values(rows, "da_f1")),
                "source_on_target_mean": mean(values(rows, "source_on_target_f1")),
                "source_on_target_std": stdev(values(rows, "source_on_target_f1")),
                "da_gain_mean": mean(values(rows, "da_gain")),
                "grad_diag_count": sum(int(row["grad_diag_count"]) for row in rows),
                "grad_cos_count": sum(int(row["grad_cos_count"]) for row in rows),
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
        "grad_diag_count",
        "grad_cos_count",
    ]
    write_tsv(root / "config_summary.tsv", config_rows, config_fields)

    print("Wrote:", root / "summary.tsv")
    print("Wrote:", root / "config_summary.tsv")
    print("Wrote:", root / "grad_norms.tsv")
    print("Wrote:", root / "grad_cosines.tsv")


if __name__ == "__main__":
    main()
