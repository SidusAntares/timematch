"""Summarize compact v3.2.1 local-shift TSV logs."""

import argparse
import csv
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local-shift TSV logs.")
    parser.add_argument("tsv")
    args = parser.parse_args()

    rows = []
    with open(args.tsv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows.extend(reader)

    counts = defaultdict(int)
    for row in rows:
        counts[row.get("task", "unknown")] += 1
    for task, count in sorted(counts.items()):
        print(f"{task}\t{count}")


if __name__ == "__main__":
    main()
