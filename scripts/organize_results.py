import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result"
SUMMARY_ROOT = RESULT_ROOT / "_summary"
FR1_DK1_ROOT = RESULT_ROOT / "FR1" / "DK1"
ARCHIVE_ROOT = FR1_DK1_ROOT / "_archive_duplicates"


@dataclass
class ResultRow:
    path: Path
    source: str
    target: str
    source_macro_f1: Optional[float]
    timematch_macro_f1: Optional[float]
    timematch_pra_macro_f1: Optional[float]
    source_accuracy: Optional[float]
    timematch_accuracy: Optional[float]
    timematch_pra_accuracy: Optional[float]
    source_experiment: str
    timematch_experiment: str
    timematch_pra_experiment: str
    pra_point_trade_off: Optional[float]
    pra_trade_off: Optional[float]
    pra_warmup_epochs: Optional[int]
    pra_min_samples_per_class: Optional[int]
    pra_pseudo_threshold: Optional[float]
    pra_bank_momentum: Optional[float]

    @property
    def is_pra(self) -> bool:
        return bool(self.timematch_pra_experiment)


def parse_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(float(value))


def read_results() -> List[ResultRow]:
    rows: List[ResultRow] = []
    for path in RESULT_ROOT.rglob("results.csv"):
        if path.is_relative_to(SUMMARY_ROOT):
            continue
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                source = raw.get("source")
                target = raw.get("target")
                if not source or not target:
                    continue
                rows.append(
                    ResultRow(
                        path=path,
                        source=source,
                        target=target,
                        source_macro_f1=parse_float(raw.get("source_macro_f1") or raw.get("source_f1")),
                        timematch_macro_f1=parse_float(raw.get("timematch_macro_f1")),
                        timematch_pra_macro_f1=parse_float(raw.get("timematch_pra_macro_f1")),
                        source_accuracy=parse_float(raw.get("source_accuracy")),
                        timematch_accuracy=parse_float(raw.get("timematch_accuracy")),
                        timematch_pra_accuracy=parse_float(raw.get("timematch_pra_accuracy")),
                        source_experiment=raw.get("source_experiment", ""),
                        timematch_experiment=raw.get("timematch_experiment", ""),
                        timematch_pra_experiment=raw.get("timematch_pra_experiment", ""),
                        pra_point_trade_off=parse_float(raw.get("pra_point_trade_off")),
                        pra_trade_off=parse_float(raw.get("pra_trade_off")),
                        pra_warmup_epochs=parse_int(raw.get("pra_warmup_epochs")),
                        pra_min_samples_per_class=parse_int(raw.get("pra_min_samples_per_class")),
                        pra_pseudo_threshold=parse_float(raw.get("pra_pseudo_threshold")),
                        pra_bank_momentum=parse_float(raw.get("pra_bank_momentum")),
                    )
                )
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_baseline_table(rows: List[ResultRow]) -> List[Dict[str, object]]:
    baseline_rows = []
    seen_pairs = set()
    for row in sorted(rows, key=lambda item: (item.source, item.target, str(item.path))):
        if row.is_pra or row.source_macro_f1 is None or row.timematch_macro_f1 is None:
            continue
        key = (row.source, row.target)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        baseline_rows.append(
            {
                "source": row.source,
                "target": row.target,
                "source_macro_f1": f"{row.source_macro_f1:.4f}",
                "timematch_macro_f1": f"{row.timematch_macro_f1:.4f}",
                "timematch_minus_source_macro_f1": f"{row.timematch_macro_f1 - row.source_macro_f1:+.4f}",
                "source_accuracy": "" if row.source_accuracy is None else f"{row.source_accuracy:.4f}",
                "timematch_accuracy": "" if row.timematch_accuracy is None else f"{row.timematch_accuracy:.4f}",
                "result_file": str(row.path.relative_to(ROOT)),
            }
        )
    return baseline_rows


def build_pra_table(rows: List[ResultRow]) -> List[Dict[str, object]]:
    pra_rows = []
    latest_by_experiment: Dict[str, ResultRow] = {}
    for row in rows:
        if not row.is_pra:
            continue
        if (row.source, row.target) != ("france/30TXT/2017", "denmark/32VNH/2017"):
            continue
        current = latest_by_experiment.get(row.timematch_pra_experiment)
        if current is None or row.path.parent.name > current.path.parent.name:
            latest_by_experiment[row.timematch_pra_experiment] = row

    for experiment, row in sorted(latest_by_experiment.items(), key=lambda item: item[1].path.parent.name):
        pra_rows.append(
            {
                "config_tag": row.path.parent.name,
                "timematch_pra_experiment": experiment,
                "source": row.source,
                "target": row.target,
                "source_macro_f1": f"{row.source_macro_f1:.4f}",
                "timematch_macro_f1": f"{row.timematch_macro_f1:.4f}",
                "timematch_pra_macro_f1": f"{row.timematch_pra_macro_f1:.4f}",
                "pra_minus_timematch_macro_f1": f"{row.timematch_pra_macro_f1 - row.timematch_macro_f1:+.4f}",
                "pra_minus_source_macro_f1": f"{row.timematch_pra_macro_f1 - row.source_macro_f1:+.4f}",
                "source_accuracy": "" if row.source_accuracy is None else f"{row.source_accuracy:.4f}",
                "timematch_accuracy": "" if row.timematch_accuracy is None else f"{row.timematch_accuracy:.4f}",
                "timematch_pra_accuracy": "" if row.timematch_pra_accuracy is None else f"{row.timematch_pra_accuracy:.4f}",
                "pra_point_trade_off": "" if row.pra_point_trade_off is None else row.pra_point_trade_off,
                "pra_trade_off": "" if row.pra_trade_off is None else row.pra_trade_off,
                "pra_warmup_epochs": "" if row.pra_warmup_epochs is None else row.pra_warmup_epochs,
                "pra_min_samples_per_class": "" if row.pra_min_samples_per_class is None else row.pra_min_samples_per_class,
                "pra_pseudo_threshold": "" if row.pra_pseudo_threshold is None else row.pra_pseudo_threshold,
                "pra_bank_momentum": "" if row.pra_bank_momentum is None else row.pra_bank_momentum,
                "result_file": str(row.path.relative_to(ROOT)),
            }
        )
    return pra_rows


def archive_duplicate_fr1_dk1_results(rows: List[ResultRow]) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = []
    keep_dirs = {
        "20260423_150356",
        "log_recovered_20260423",
    }

    latest_by_experiment: Dict[str, Path] = {}
    for row in rows:
        if not row.is_pra or row.path.parent.parent != FR1_DK1_ROOT:
            continue
        current = latest_by_experiment.get(row.timematch_pra_experiment)
        if current is None or row.path.parent.name > current.name:
            latest_by_experiment[row.timematch_pra_experiment] = row.path.parent

    keep_dirs.update(path.name for path in latest_by_experiment.values())

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for child in sorted(FR1_DK1_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.name == ARCHIVE_ROOT.name:
            continue
        if child.name in keep_dirs:
            actions.append({"action": "kept", "path": str(child.relative_to(ROOT))})
            continue

        destination = ARCHIVE_ROOT / child.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(child), str(destination))
        actions.append(
            {
                "action": "archived",
                "path": str(child.relative_to(ROOT)),
                "archived_to": str(destination.relative_to(ROOT)),
            }
        )
    return actions


def write_next_steps(baseline_rows: List[Dict[str, object]], pra_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Result Organization Notes",
        "",
        "## Current scope",
        "- PRA is only tested on `FR1 -> DK1`.",
        "- All other transfer pairs should still be treated as TimeMatch baseline results.",
        "- Unless PRA first exceeds the `FR1 -> DK1` baseline, there is no strong reason to expand PRA to other domain adaptation pairs yet.",
        "",
        "## Baseline overview",
    ]

    for row in baseline_rows:
        lines.append(
            f"- `{row['source']} -> {row['target']}`: "
            f"source `{row['source_macro_f1']}`, TimeMatch `{row['timematch_macro_f1']}`, "
            f"delta `{row['timematch_minus_source_macro_f1']}`"
        )

    lines.extend(
        [
            "",
            "## FR1 -> DK1 PRA overview",
        ]
    )

    for row in pra_rows:
        lines.append(
            f"- `{row['config_tag']}`: PRA `{row['timematch_pra_macro_f1']}`, "
            f"vs TimeMatch `{row['pra_minus_timematch_macro_f1']}`, "
            f"vs source `{row['pra_minus_source_macro_f1']}`"
        )

    lines.extend(
        [
            "",
            "## Recommended next step",
            "- Stop expanding PRA to other source-target pairs for now.",
            "- Keep `FR1 -> DK1` as the single PRA validation track until PRA clearly beats the TimeMatch baseline `0.5608` macro-F1.",
            "- The next useful experiments should focus on prototype quality and scheduling, not on adding more transfer pairs.",
            "- Suggested order:",
            "  1. tune point alignment weight downward from `0.005`",
            "  2. tune bank momentum upward from `0.95`",
            "  3. if needed, retune pseudo-threshold around `0.90-0.93`",
        ]
    )

    notes_path = SUMMARY_ROOT / "next_steps.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_results()
    baseline_rows = build_baseline_table(rows)
    pra_rows = build_pra_table(rows)
    cleanup_actions = archive_duplicate_fr1_dk1_results(rows)

    write_csv(
        SUMMARY_ROOT / "baseline_overview.csv",
        [
            "source",
            "target",
            "source_macro_f1",
            "timematch_macro_f1",
            "timematch_minus_source_macro_f1",
            "source_accuracy",
            "timematch_accuracy",
            "result_file",
        ],
        baseline_rows,
    )
    write_csv(
        SUMMARY_ROOT / "fr1_dk1_pra_overview.csv",
        [
            "config_tag",
            "timematch_pra_experiment",
            "source",
            "target",
            "source_macro_f1",
            "timematch_macro_f1",
            "timematch_pra_macro_f1",
            "pra_minus_timematch_macro_f1",
            "pra_minus_source_macro_f1",
            "source_accuracy",
            "timematch_accuracy",
            "timematch_pra_accuracy",
            "pra_point_trade_off",
            "pra_trade_off",
            "pra_warmup_epochs",
            "pra_min_samples_per_class",
            "pra_pseudo_threshold",
            "pra_bank_momentum",
            "result_file",
        ],
        pra_rows,
    )
    write_csv(
        SUMMARY_ROOT / "cleanup_manifest.csv",
        ["action", "path", "archived_to"],
        cleanup_actions,
    )
    write_next_steps(baseline_rows, pra_rows)

    print(f"Baseline summary: {SUMMARY_ROOT / 'baseline_overview.csv'}")
    print(f"FR1->DK1 PRA summary: {SUMMARY_ROOT / 'fr1_dk1_pra_overview.csv'}")
    print(f"Cleanup manifest: {SUMMARY_ROOT / 'cleanup_manifest.csv'}")
    print(f"Next steps note: {SUMMARY_ROOT / 'next_steps.md'}")


if __name__ == "__main__":
    main()
