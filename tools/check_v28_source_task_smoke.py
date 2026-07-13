import argparse
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description="Check source task-name smoke equivalence.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    rows = read_rows(args.input)
    by_config = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)
    fields = [
        "config",
        "field",
        "left_task",
        "right_task",
        "left_value",
        "right_value",
        "equivalent",
    ]
    comparisons = []
    failures = []
    numeric_fields = {
        "first_step_classification_loss",
        "first_step_structure_loss",
        "first_step_total_loss",
    }
    compared_fields = [
        "initial_state_hash",
        "first_batch_hash",
        "first_step_classification_loss",
        "first_step_structure_loss",
        "first_step_total_loss",
        "final_state_dict_hash",
    ]
    for config, group in sorted(by_config.items()):
        if len(group) != 2:
            failures.append(f"{config}: expected 2 rows, got {len(group)}")
            continue
        left, right = sorted(group, key=lambda row: row["task"])
        for field in compared_fields:
            if field in numeric_fields:
                equivalent = abs(float(left[field]) - float(right[field])) <= args.tolerance
            else:
                equivalent = bool(left[field]) and left[field] == right[field]
            comparison = {
                "config": config,
                "field": field,
                "left_task": left["task"],
                "right_task": right["task"],
                "left_value": left[field],
                "right_value": right[field],
                "equivalent": equivalent,
            }
            comparisons.append(comparison)
            if not equivalent:
                failures.append(f"{config}/{field}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"SOURCE_SMOKE_CHECK|comparisons={len(comparisons)}|failures={len(failures)}|output={output}")
    if failures:
        print("SOURCE_SMOKE_DIVERGENCE|" + ",".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
