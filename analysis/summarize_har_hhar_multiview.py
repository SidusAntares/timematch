import argparse
import csv
import glob
from collections import Counter, defaultdict
from pathlib import Path

from summarize_har_hhar_structure_overnight import collect_records, fmt


def expand_roots(patterns):
    roots = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        roots.extend(matches if matches else [Path(pattern)])
    return [root for root in roots if root.exists()]


def split_records_by_root(roots):
    root_records = []
    for root in roots:
        records = collect_records(root)
        if records:
            root_records.append((root, records))
    return root_records


def best_variant(records, dataset, task, phase):
    phase_records = [
        record
        for record in records
        if record["dataset"] == dataset and record["task"] == task and record["phase"] == phase
    ]
    baseline = next((record for record in phase_records if record["variant"] == "baseline"), None)
    variants = [record for record in phase_records if record["variant"] != "baseline"]
    best = max(variants, key=lambda record: record["f1"]) if variants else None
    return baseline, best, variants


def build_rows(root_records):
    rows = []
    for root, records in root_records:
        tasks = sorted({(record["dataset"], record["task"]) for record in records})
        for dataset, task in tasks:
            source_base, source_best, source_variants = best_variant(records, dataset, task, "source")
            da_base, da_best, da_variants = best_variant(records, dataset, task, "timematch")
            rows.append(
                {
                    "log_root": str(root),
                    "dataset": dataset,
                    "task": task,
                    "source_baseline": value(source_base),
                    "source_best_view": variant(source_best),
                    "source_best": value(source_best),
                    "source_delta": delta(source_best, source_base),
                    "source_views_tested": len(source_variants),
                    "da_baseline": value(da_base),
                    "da_best_view": variant(da_best),
                    "da_best": value(da_best),
                    "da_delta": delta(da_best, da_base),
                    "da_views_tested": len(da_variants),
                }
            )
    return rows


def value(record):
    return None if record is None else record["f1"]


def variant(record):
    return "NA" if record is None else record["variant"]


def delta(left, right):
    if left is None or right is None:
        return None
    return left["f1"] - right["f1"]


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"])].append(row)

    task_rows = []
    for (dataset, task), group in sorted(groups.items()):
        source_views = Counter(row["source_best_view"] for row in group if row["source_best_view"] != "NA")
        da_views = Counter(row["da_best_view"] for row in group if row["da_best_view"] != "NA")
        task_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "runs": len(group),
                "source_gain_runs": sum(1 for row in group if positive(row["source_delta"])),
                "source_baseline": mean(row["source_baseline"] for row in group),
                "source_best": mean(row["source_best"] for row in group),
                "source_delta": mean(row["source_delta"] for row in group),
                "source_best_views": view_counts(source_views),
                "da_gain_runs": sum(1 for row in group if positive(row["da_delta"])),
                "da_baseline": mean(row["da_baseline"] for row in group),
                "da_best": mean(row["da_best"] for row in group),
                "da_delta": mean(row["da_delta"] for row in group),
                "da_best_views": view_counts(da_views),
            }
        )
    return task_rows


def positive(value):
    return value is not None and value > 0


def view_counts(counter):
    if not counter:
        return "NA"
    return ", ".join(f"{name}:{count}" for name, count in counter.most_common())


def dataset_aggregate(task_rows, dataset=None):
    subset = [row for row in task_rows if dataset is None or row["dataset"] == dataset]
    if not subset:
        return None
    runs = sum(row["runs"] for row in subset)
    return {
        "dataset": dataset or "ALL",
        "tasks": len(subset),
        "runs": runs,
        "source_gain_runs": sum(row["source_gain_runs"] for row in subset),
        "source_baseline": mean(row["source_baseline"] for row in subset),
        "source_best": mean(row["source_best"] for row in subset),
        "source_delta": mean(row["source_delta"] for row in subset),
        "da_gain_runs": sum(row["da_gain_runs"] for row in subset),
        "da_baseline": mean(row["da_baseline"] for row in subset),
        "da_best": mean(row["da_best"] for row in subset),
        "da_delta": mean(row["da_delta"] for row in subset),
    }


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def write_markdown(path, task_rows, roots):
    datasets = sorted({row["dataset"] for row in task_rows})
    aggregate_body = []
    for dataset in datasets + [None]:
        stats = dataset_aggregate(task_rows, dataset)
        aggregate_body.append(
            [
                stats["dataset"],
                str(stats["tasks"]),
                str(stats["runs"]),
                f'{stats["source_gain_runs"]}/{stats["runs"]}',
                fmt(stats["source_baseline"]),
                fmt(stats["source_best"]),
                fmt(stats["source_delta"]),
                f'{stats["da_gain_runs"]}/{stats["runs"]}',
                fmt(stats["da_baseline"]),
                fmt(stats["da_best"]),
                fmt(stats["da_delta"]),
            ]
        )

    task_body = []
    for row in task_rows:
        task_body.append(
            [
                row["dataset"],
                row["task"],
                str(row["runs"]),
                f'{row["source_gain_runs"]}/{row["runs"]}',
                fmt(row["source_baseline"]),
                fmt(row["source_best"]),
                fmt(row["source_delta"]),
                row["source_best_views"],
                f'{row["da_gain_runs"]}/{row["runs"]}',
                fmt(row["da_baseline"]),
                fmt(row["da_best"]),
                fmt(row["da_delta"]),
                row["da_best_views"],
            ]
        )

    text = [
        "# v2.5 HAR/HHAR Multi-View Summary",
        "",
        "## Inputs",
        "",
        *[f"- `{root}`" for root in roots],
        "",
        "## Aggregate",
        "",
        markdown_table(
            [
                "dataset",
                "tasks",
                "runs",
                "source gains",
                "source base",
                "source best",
                "source delta",
                "DA gains",
                "DA base",
                "DA best",
                "DA delta",
            ],
            aggregate_body,
        ),
        "",
        "## Task View Preference",
        "",
        markdown_table(
            [
                "dataset",
                "task",
                "runs",
                "source gains",
                "source base",
                "source best",
                "source delta",
                "source best views",
                "DA gains",
                "DA base",
                "DA best",
                "DA delta",
                "DA best views",
            ],
            task_body,
        ),
        "",
        "## Reading",
        "",
        "- `best views` records which structure view wins within each log root. If several views appear across roots, the task has unstable or seed-sensitive structure preference.",
        "- The goal of this table is not to claim a single fixed parameter is universal, but to test whether a multi-view source-structure bank contains useful views across generic time-series UDA tasks.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_roots", nargs="+")
    parser.add_argument("--output_dir", type=Path, default=Path("result/_summary/har_hhar_multiview"))
    args = parser.parse_args()

    roots = expand_roots(args.log_roots)
    root_records = split_records_by_root(roots)
    rows = build_rows(root_records)
    task_rows = aggregate_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "har_hhar_multiview_runs.csv", rows)
    write_csv(args.output_dir / "har_hhar_multiview_task_summary.csv", task_rows)
    write_markdown(args.output_dir / "har_hhar_multiview_summary.md", task_rows, [root for root, _ in root_records])

    print(f"Wrote {args.output_dir / 'har_hhar_multiview_runs.csv'}")
    print(f"Wrote {args.output_dir / 'har_hhar_multiview_task_summary.csv'}")
    print(f"Wrote {args.output_dir / 'har_hhar_multiview_summary.md'}")


if __name__ == "__main__":
    main()
