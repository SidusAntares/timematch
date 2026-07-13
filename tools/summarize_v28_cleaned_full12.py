#!/usr/bin/env python3
"""Summarize v2.8 cleaned-code full12 reproduction logs."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional


VAL_RE = re.compile(
    r"Validation result:\s*loss=([-+0-9.eE]+),\s*acc=([-+0-9.eE]+),\s*f1=([-+0-9.eE]+)"
)
TEST_RE = re.compile(
    r"Test result for [^:]+:\s*accuracy=([-+0-9.eE]+),\s*f1=([-+0-9.eE]+)"
)

CONFIGS = ["base", "raw_global", "smooth_k3", "elastic_r2"]

SUMMARY_FIELDS = [
    "task",
    "source",
    "target",
    "seed",
    "config",
    "source_kind",
    "source_checkpoint",
    "status",
    "source_on_target_macro_f1",
    "da_val_macro_f1",
    "da_test_macro_f1",
    "da_gain",
    "global_shift",
    "pseudo_confidence",
    "pseudo_ratio",
    "target_loss",
    "runtime_s",
    "log_path",
]


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fields})


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def _std(values: List[float]) -> Optional[float]:
    return stdev(values) if len(values) >= 2 else None


def _last_test_f1(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = TEST_RE.findall(text)
    if not matches:
        return None
    return _safe_float(matches[-1][1])


def _best_val_f1(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    values = [_safe_float(match[2]) for match in VAL_RE.findall(text)]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _last_diag(path: Path) -> Dict[str, str]:
    rows = _read_tsv(path)
    return rows[-1] if rows else {}


def _normalize_status(status: str) -> str:
    status = str(status).strip()
    return "ok" if status == "0" else (status or "unknown")


def _source_kind(config: str) -> str:
    return "plain_source" if config == "base" else "source_structure"


def build_summary(log_root: Path) -> List[Dict[str, object]]:
    rows = _read_tsv(log_root / "da_job_status.tsv")
    summary: List[Dict[str, object]] = []
    for row in rows:
        config = row.get("config", "")
        source_eval_log = Path(row.get("source_eval_log", ""))
        da_log = Path(row.get("da_log", ""))
        diag_log = Path(row.get("diag_log", ""))
        source_on_target = _last_test_f1(source_eval_log)
        da_test = _last_test_f1(da_log)
        da_val = _best_val_f1(da_log)
        diag = _last_diag(diag_log)
        da_gain = None
        if source_on_target is not None and da_test is not None:
            da_gain = da_test - source_on_target
        task = row.get("task", "")
        summary.append(
            {
                "task": task,
                "source": row.get("source", ""),
                "target": row.get("target", ""),
                "seed": row.get("seed", ""),
                "config": config,
                "source_kind": _source_kind(config),
                "source_checkpoint": row.get("source_checkpoint", ""),
                "status": _normalize_status(row.get("status", "")),
                "source_on_target_macro_f1": source_on_target,
                "da_val_macro_f1": da_val,
                "da_test_macro_f1": da_test,
                "da_gain": da_gain,
                "global_shift": _safe_float(diag.get("estimated_shift_t_to_s")),
                "pseudo_confidence": _safe_float(diag.get("teacher_pseudo_confidence_mean")),
                "pseudo_ratio": _safe_float(diag.get("teacher_pseudo_coverage")),
                "target_loss": _safe_float(diag.get("target_loss")),
                "runtime_s": _safe_float(row.get("runtime_s")),
                "log_path": str(da_log),
            }
        )
    return summary


def aggregate_by_config(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for config in sorted({str(row.get("config", "")) for row in rows}):
        group = [row for row in rows if row.get("config") == config]
        source_values = [_safe_float(row.get("source_on_target_macro_f1")) for row in group]
        test_values = [_safe_float(row.get("da_test_macro_f1")) for row in group]
        gain_values = [_safe_float(row.get("da_gain")) for row in group]
        runtime_values = [_safe_float(row.get("runtime_s")) for row in group]
        source_values = [value for value in source_values if value is not None]
        test_values = [value for value in test_values if value is not None]
        gain_values = [value for value in gain_values if value is not None]
        runtime_values = [value for value in runtime_values if value is not None]
        out.append(
            {
                "config": config,
                "mean_source_on_target_f1": _mean(source_values),
                "std_source_on_target_f1": _std(source_values),
                "mean_test_f1": _mean(test_values),
                "std_test_f1": _std(test_values),
                "mean_da_gain": _mean(gain_values),
                "std_da_gain": _std(gain_values),
                "mean_runtime_s": _mean(runtime_values),
                "num_jobs": len(group),
                "num_success": sum(1 for row in group if row.get("status") == "ok"),
            }
        )
    return out


def aggregate_by_task(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    keys = sorted({(str(row.get("task", "")), str(row.get("config", ""))) for row in rows})
    for task, config in keys:
        group = [row for row in rows if row.get("task") == task and row.get("config") == config]
        test_values = [_safe_float(row.get("da_test_macro_f1")) for row in group]
        gain_values = [_safe_float(row.get("da_gain")) for row in group]
        test_values = [value for value in test_values if value is not None]
        gain_values = [value for value in gain_values if value is not None]
        out.append(
            {
                "task": task,
                "config": config,
                "mean_test_f1": _mean(test_values),
                "std_test_f1": _std(test_values),
                "mean_da_gain": _mean(gain_values),
                "num_seeds": len({str(row.get("seed", "")) for row in group}),
            }
        )
    return out


def delta_vs_base(by_task: List[Dict[str, object]]) -> List[Dict[str, object]]:
    base_by_task = {
        str(row.get("task")): _safe_float(row.get("mean_test_f1"))
        for row in by_task
        if row.get("config") == "base"
    }
    out: List[Dict[str, object]] = []
    for row in by_task:
        task = str(row.get("task", ""))
        config = str(row.get("config", ""))
        if config == "base":
            continue
        base_value = base_by_task.get(task)
        config_value = _safe_float(row.get("mean_test_f1"))
        out.append(
            {
                "task": task,
                "config": config,
                "base_mean_test_f1": base_value,
                "config_mean_test_f1": config_value,
                "delta_vs_base": config_value - base_value if config_value is not None and base_value is not None else None,
                "num_seeds": row.get("num_seeds"),
            }
        )
    return out


def per_seed_pivot(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        key = (str(row.get("task", "")), str(row.get("seed", "")))
        item = grouped.setdefault(key, {"task": key[0], "seed": key[1]})
        item[str(row.get("config", ""))] = row.get("da_test_macro_f1")
    return [grouped[key] for key in sorted(grouped)]


def seed_improvement_count(pivot: List[Dict[str, object]], config: str) -> int:
    count = 0
    for row in pivot:
        base = _safe_float(row.get("base"))
        value = _safe_float(row.get(config))
        if base is not None and value is not None and value > base:
            count += 1
    return count


def task_improvement_count(delta_rows: List[Dict[str, object]], config: str) -> int:
    return sum(
        1
        for row in delta_rows
        if row.get("config") == config and (_safe_float(row.get("delta_vs_base")) or 0.0) > 0.0
    )


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _markdown_table(rows: List[Dict[str, object]], fields: List[str]) -> str:
    if not rows:
        return "\n无记录。\n"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    log_root: Path,
    summary: List[Dict[str, object]],
    by_config: List[Dict[str, object]],
    by_task: List[Dict[str, object]],
    delta_rows: List[Dict[str, object]],
    pivot_rows: List[Dict[str, object]],
) -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = _read_tsv(log_root / "source_checkpoint_inventory.tsv")
    job_status = _read_tsv(log_root / "job_status.tsv")
    failed_jobs = [row for row in job_status if row.get("status") not in {"0", "ok", "reused_probe", "reused_full12"}]
    config_fields = [
        "config",
        "mean_source_on_target_f1",
        "std_source_on_target_f1",
        "mean_test_f1",
        "std_test_f1",
        "mean_da_gain",
        "std_da_gain",
        "mean_runtime_s",
        "num_jobs",
        "num_success",
    ]
    task_fields = ["task", "config", "mean_test_f1", "std_test_f1", "mean_da_gain", "num_seeds"]
    delta_fields = ["task", "config", "base_mean_test_f1", "config_mean_test_f1", "delta_vs_base", "num_seeds"]
    full_fields = [
        "task",
        "seed",
        "config",
        "status",
        "source_on_target_macro_f1",
        "da_val_macro_f1",
        "da_test_macro_f1",
        "da_gain",
        "global_shift",
        "pseudo_confidence",
        "pseudo_ratio",
        "target_loss",
        "runtime_s",
    ]
    improvement_rows = []
    for config in CONFIGS:
        if config == "base":
            continue
        improvement_rows.append(
            {
                "config": config,
                "tasks_improved_vs_base": task_improvement_count(delta_rows, config),
                "seeds_improved_vs_base": seed_improvement_count(pivot_rows, config),
            }
        )

    lines = [
        "# v2.8 Cleaned-Code Full12 Reproduction Report",
        "",
        "## 1. Setup",
        "",
        f"- run directory: `{log_root}`",
        f"- code commit: `{_git_commit(root)}`",
        "- tasks: 12 ordered remote-sensing domain pairs",
        "- seeds: 1, 2, 3",
        "- configs: base, raw_global, smooth_k3, elastic_r2",
        "- DA: epochs=20, steps_per_epoch=500, with_shift_aug=False, closed_set=True",
        "- DA method: normal TimeMatch only",
        "",
        "## 2. Source Checkpoint Inventory",
        "",
        f"- inventory rows: {len(inventory)}",
        f"- reused from probe: {sum(1 for row in inventory if row.get('reused_from_probe') == 'True')}",
        f"- source training needed: {sum(1 for row in inventory if row.get('source_training_needed') == 'True')}",
        f"- failed/missing source checkpoints: {sum(1 for row in inventory if row.get('checkpoint_exists') != 'True')}",
        "",
        "## 3. Job Status",
        "",
        f"- all job status rows: {len(job_status)}",
        f"- failed jobs: {len(failed_jobs)}",
        "",
        "## 4. Config-Level Summary",
        "",
        _markdown_table(by_config, config_fields),
        "## 5. Task-Level Summary",
        "",
        _markdown_table(by_task, task_fields),
        "## 6. Delta vs Base",
        "",
        _markdown_table(delta_rows, delta_fields),
        "## 7. Improvement Counts",
        "",
        _markdown_table(improvement_rows, ["config", "tasks_improved_vs_base", "seeds_improved_vs_base"]),
        "## 8. Full Result Table",
        "",
        _markdown_table(summary, full_fields),
    ]
    if failed_jobs:
        lines.extend(
            [
                "## 9. Failed Jobs",
                "",
                _markdown_table(failed_jobs, ["phase", "source", "target", "config", "seed", "status", "runtime_s", "log_path"]),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    summary = build_summary(log_root)
    by_config = aggregate_by_config(summary)
    by_task = aggregate_by_task(summary)
    delta_rows = delta_vs_base(by_task)
    pivot_rows = per_seed_pivot(summary)

    _write_tsv(Path(args.output), summary, SUMMARY_FIELDS)
    _write_tsv(
        log_root / "summary_by_config.tsv",
        by_config,
        [
            "config",
            "mean_source_on_target_f1",
            "std_source_on_target_f1",
            "mean_test_f1",
            "std_test_f1",
            "mean_da_gain",
            "std_da_gain",
            "mean_runtime_s",
            "num_jobs",
            "num_success",
        ],
    )
    _write_tsv(log_root / "summary_by_task.tsv", by_task, ["task", "config", "mean_test_f1", "std_test_f1", "mean_da_gain", "num_seeds"])
    _write_tsv(log_root / "delta_vs_base_by_task.tsv", delta_rows, ["task", "config", "base_mean_test_f1", "config_mean_test_f1", "delta_vs_base", "num_seeds"])
    _write_tsv(log_root / "per_seed_pivot.tsv", pivot_rows, ["task", "seed", "base", "raw_global", "smooth_k3", "elastic_r2"])

    if args.report:
        write_report(Path(args.report), log_root, summary, by_config, by_task, delta_rows, pivot_rows)

    print(f"Wrote {args.output}")
    print(f"Wrote {log_root / 'summary_by_config.tsv'}")
    print(f"Wrote {log_root / 'summary_by_task.tsv'}")
    print(f"Wrote {log_root / 'delta_vs_base_by_task.tsv'}")
    print(f"Wrote {log_root / 'per_seed_pivot.tsv'}")
    if args.report:
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
