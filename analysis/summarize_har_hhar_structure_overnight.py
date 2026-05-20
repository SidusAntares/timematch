import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


TEST_RE = re.compile(r"Test result for .*?: accuracy=([0-9.]+), f1=([0-9.]+)")
ERROR_PATTERNS = ("Traceback", "error:", "RuntimeError", "ValueError", "StopIteration", "Unsupported")


def parse_name(path):
    name = path.name
    if "_source_" in name:
        prefix = name.split("_source_", 1)[0]
        phase = "source"
    elif "_timematch_" in name:
        prefix = name.split("_timematch_", 1)[0]
        phase = "timematch"
    else:
        return None

    if prefix.startswith("hhar_sa_"):
        dataset = "HHAR_SA"
        rest = prefix[len("hhar_sa_") :]
    elif prefix.startswith("har_"):
        dataset = "HAR"
        rest = prefix[len("har_") :]
    else:
        return None

    if "_to_" not in rest:
        return None
    source, tail = rest.split("_to_", 1)
    if "_" not in tail:
        return None
    target, variant = tail.split("_", 1)
    return {
        "dataset": dataset,
        "task": f"{source}->{target}",
        "variant": variant,
        "phase": phase,
    }


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tests = TEST_RE.findall(text)
    if not tests:
        return None
    accuracy, f1 = tests[-1]
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "has_error": any(pattern in text for pattern in ERROR_PATTERNS),
    }


def collect_records(log_root):
    records = []
    for path in sorted(log_root.glob("*.log")):
        meta = parse_name(path)
        if meta is None:
            continue
        metrics = parse_log(path)
        if metrics is None:
            continue
        records.append({**meta, **metrics, "log": str(path)})
    return records


def best_nonbaseline(records):
    variants = [record for record in records if record["variant"] != "baseline"]
    if not variants:
        return None
    return max(variants, key=lambda record: record["f1"])


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["task"], record["phase"])].append(record)

    task_keys = sorted({(record["dataset"], record["task"]) for record in records})
    rows = []
    for dataset, task in task_keys:
        source_records = grouped.get((dataset, task, "source"), [])
        da_records = grouped.get((dataset, task, "timematch"), [])

        source_base = next((record for record in source_records if record["variant"] == "baseline"), None)
        da_base = next((record for record in da_records if record["variant"] == "baseline"), None)
        source_best = best_nonbaseline(source_records)
        da_best = best_nonbaseline(da_records)

        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "source_baseline": value_or_none(source_base),
                "source_best_variant": variant_or_none(source_best),
                "source_best": value_or_none(source_best),
                "source_delta": delta_or_none(source_best, source_base),
                "da_baseline": value_or_none(da_base),
                "da_best_variant": variant_or_none(da_best),
                "da_best": value_or_none(da_best),
                "da_delta": delta_or_none(da_best, da_base),
                "baseline_da_minus_source": delta_or_none(da_base, source_base),
                "source_variants_tested": len([r for r in source_records if r["variant"] != "baseline"]),
                "da_variants_tested": len([r for r in da_records if r["variant"] != "baseline"]),
            }
        )
    return rows


def value_or_none(record):
    return None if record is None else record["f1"]


def variant_or_none(record):
    return "NA" if record is None else record["variant"]


def delta_or_none(left, right):
    if left is None or right is None:
        return None
    return left["f1"] - right["f1"]


def fmt(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_csv(path, rows):
    fieldnames = [
        "dataset",
        "task",
        "source_baseline",
        "source_best_variant",
        "source_best",
        "source_delta",
        "da_baseline",
        "da_best_variant",
        "da_best",
        "da_delta",
        "baseline_da_minus_source",
        "source_variants_tested",
        "da_variants_tested",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows, dataset=None):
    subset = [row for row in rows if dataset is None or row["dataset"] == dataset]
    if not subset:
        return None

    def avg(key):
        values = [row[key] for row in subset if row[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "n": len(subset),
        "source_gain": sum(1 for row in subset if row["source_delta"] is not None and row["source_delta"] > 0),
        "da_gain": sum(1 for row in subset if row["da_delta"] is not None and row["da_delta"] > 0),
        "source_baseline": avg("source_baseline"),
        "source_best": avg("source_best"),
        "source_delta": avg("source_delta"),
        "da_baseline": avg("da_baseline"),
        "da_best": avg("da_best"),
        "da_delta": avg("da_delta"),
        "baseline_da_minus_source": avg("baseline_da_minus_source"),
    }


def markdown_table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def write_markdown(path, rows, log_root, error_count):
    datasets = sorted({row["dataset"] for row in rows})
    agg_rows = []
    for dataset in datasets:
        stats = aggregate(rows, dataset)
        agg_rows.append(
            [
                dataset,
                str(stats["n"]),
                f'{stats["source_gain"]}/{stats["n"]}',
                fmt(stats["source_baseline"]),
                fmt(stats["source_best"]),
                fmt(stats["source_delta"]),
                f'{stats["da_gain"]}/{stats["n"]}',
                fmt(stats["da_baseline"]),
                fmt(stats["da_best"]),
                fmt(stats["da_delta"]),
            ]
        )
    all_stats = aggregate(rows)
    agg_rows.append(
        [
            "ALL",
            str(all_stats["n"]),
            f'{all_stats["source_gain"]}/{all_stats["n"]}',
            fmt(all_stats["source_baseline"]),
            fmt(all_stats["source_best"]),
            fmt(all_stats["source_delta"]),
            f'{all_stats["da_gain"]}/{all_stats["n"]}',
            fmt(all_stats["da_baseline"]),
            fmt(all_stats["da_best"]),
            fmt(all_stats["da_delta"]),
        ]
    )

    task_rows = []
    for row in rows:
        task_rows.append(
            [
                row["dataset"],
                row["task"],
                fmt(row["source_baseline"]),
                row["source_best_variant"],
                fmt(row["source_best"]),
                fmt(row["source_delta"]),
                fmt(row["da_baseline"]),
                row["da_best_variant"],
                fmt(row["da_best"]),
                fmt(row["da_delta"]),
                fmt(row["baseline_da_minus_source"]),
            ]
        )

    text = []
    text.append("# HAR/HHAR Structure Overnight Summary")
    text.append("")
    text.append(f"- Log root: `{log_root}`")
    text.append(f"- Parsed tasks: {len(rows)}")
    text.append(f"- Logs with explicit error patterns: {error_count}")
    text.append("")
    text.append("## Aggregate")
    text.append("")
    text.append(
        markdown_table(
            [
                "dataset",
                "tasks",
                "source gains",
                "source base",
                "source best",
                "source delta",
                "DA gains",
                "DA base",
                "DA best",
                "DA delta",
            ],
            agg_rows,
        )
    )
    text.append("")
    text.append("## Task-Level Results")
    text.append("")
    text.append(
        markdown_table(
            [
                "dataset",
                "task",
                "source base",
                "source best view",
                "source best",
                "source delta",
                "DA base",
                "DA best view",
                "DA best",
                "DA delta",
                "DA-source base",
            ],
            task_rows,
        )
    )
    text.append("")
    text.append("## Reading")
    text.append("")
    text.append(
        "- `source delta` measures whether source-side structure training improves the source-trained checkpoint before TimeMatch."
    )
    text.append(
        "- `DA delta` measures whether the same structure-trained source checkpoint improves the final TimeMatch result."
    )
    text.append(
        "- `DA-source base` shows whether vanilla TimeMatch helps or hurts compared with vanilla source-only on the same task."
    )
    text.append(
        "- A positive `source delta` but negative `DA delta` suggests the structure view helps representation but the DA stage may be misaligned for that task."
    )
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def count_error_logs(log_root):
    count = 0
    for path in log_root.glob("*.log"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in ERROR_PATTERNS):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    log_root = args.log_root
    if args.output_dir is None:
        output_dir = log_root
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(log_root)
    rows = summarize(records)
    error_count = count_error_logs(log_root)

    csv_path = output_dir / "har_hhar_structure_source_da_summary.csv"
    md_path = output_dir / "har_hhar_structure_source_da_summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, log_root, error_count)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
