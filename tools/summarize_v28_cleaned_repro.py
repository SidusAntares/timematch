#!/usr/bin/env python3
"""Summarize cleaned-code v2.8 reproduction logs.

The launcher writes one DA status row per job plus compact TimeMatch diagnostic
TSVs. This script keeps the summary purely observational: it parses logs and
writes per-job and aggregate TSV files without making method decisions.
"""

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


def _source_kind(config: str) -> str:
    return "plain_source" if config == "base" else "source_structure"


def _normalize_status(status: str) -> str:
    status = str(status).strip()
    if status == "0":
        return "ok"
    return status or "unknown"


def build_summary(log_root: Path) -> List[Dict[str, object]]:
    status_path = log_root / "da_job_status.tsv"
    rows = _read_tsv(status_path)
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
        summary.append(
            {
                "task": row.get("task", ""),
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


def _aggregate(rows: List[Dict[str, object]], keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        group_key = tuple(row.get(key, "") for key in keys)
        groups.setdefault(group_key, []).append(row)

    metrics = [
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
    out: List[Dict[str, object]] = []
    for group_key, group in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["n"] = len(group)
        item["ok_n"] = sum(1 for row in group if row.get("status") == "ok")
        for metric in metrics:
            values = [_safe_float(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            item[metric] = mean(values) if values else None
        out.append(item)
    return out


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return stdev(values)


def _aggregate_for_config(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("config", "")), []).append(row)
    out: List[Dict[str, object]] = []
    for config, group in sorted(groups.items()):
        test_values = [_safe_float(row.get("da_test_macro_f1")) for row in group]
        gain_values = [_safe_float(row.get("da_gain")) for row in group]
        test_values = [value for value in test_values if value is not None]
        gain_values = [value for value in gain_values if value is not None]
        out.append(
            {
                "config": config,
                "mean_test_f1": mean(test_values) if test_values else None,
                "std_test_f1": _std(test_values),
                "mean_da_gain": mean(gain_values) if gain_values else None,
                "std_da_gain": _std(gain_values),
                "num_jobs": len(group),
                "num_success": sum(1 for row in group if row.get("status") == "ok"),
            }
        )
    return out


def _aggregate_for_task(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("task", "")), str(row.get("config", ""))), []).append(row)
    out: List[Dict[str, object]] = []
    for (task, config), group in sorted(groups.items()):
        test_values = [_safe_float(row.get("da_test_macro_f1")) for row in group]
        gain_values = [_safe_float(row.get("da_gain")) for row in group]
        test_values = [value for value in test_values if value is not None]
        gain_values = [value for value in gain_values if value is not None]
        out.append(
            {
                "task": task,
                "config": config,
                "mean_test_f1": mean(test_values) if test_values else None,
                "std_test_f1": _std(test_values),
                "mean_da_gain": mean(gain_values) if gain_values else None,
                "num_seeds": len({str(row.get("seed", "")) for row in group}),
            }
        )
    return out


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


def _markdown_table(rows: List[Dict[str, object]], fields: List[str], limit: Optional[int] = None) -> str:
    selected = rows if limit is None else rows[:limit]
    if not selected:
        return "\n无记录。\n"
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in selected:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _comparison_vs_base(by_config: List[Dict[str, object]]) -> List[Dict[str, object]]:
    base = next((row for row in by_config if row.get("config") == "base"), None)
    base_test = _safe_float(base.get("mean_test_f1")) if base else None
    base_gain = _safe_float(base.get("mean_da_gain")) if base else None
    out: List[Dict[str, object]] = []
    for row in by_config:
        test_value = _safe_float(row.get("mean_test_f1"))
        gain_value = _safe_float(row.get("mean_da_gain"))
        out.append(
            {
                "config": row.get("config"),
                "mean_test_f1": test_value,
                "delta_test_f1_vs_base": test_value - base_test if test_value is not None and base_test is not None else None,
                "mean_da_gain": gain_value,
                "delta_da_gain_vs_base": gain_value - base_gain if gain_value is not None and base_gain is not None else None,
            }
        )
    return out


def write_report(
    path: Path,
    log_root: Path,
    summary: List[Dict[str, object]],
    by_config: List[Dict[str, object]],
    by_task: List[Dict[str, object]],
) -> None:
    failed = [row for row in summary if row.get("status") != "ok"]
    inventory = _read_tsv(log_root / "source_checkpoint_inventory.tsv")
    root = Path(__file__).resolve().parents[1]
    config_fields = ["config", "mean_test_f1", "std_test_f1", "mean_da_gain", "std_da_gain", "num_jobs", "num_success"]
    task_fields = ["task", "config", "mean_test_f1", "std_test_f1", "mean_da_gain", "num_seeds"]
    inventory_fields = [
        "task",
        "seed",
        "config",
        "source_region",
        "checkpoint_exists",
        "source_training_needed",
        "source_train_status",
        "checkpoint_path",
    ]
    comparison_fields = ["config", "mean_test_f1", "delta_test_f1_vs_base", "mean_da_gain", "delta_da_gain_vs_base"]
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
    text = [
        "# v2.8 Cleaned-Code Reproduction Report",
        "",
        "## 实验口径",
        "",
        "- 入口：cleaned-code TimeMatch baseline 与 source-side structure checkpoints",
        "- 配置：base、raw_global、smooth_k3、elastic_r2",
        "- DA：epochs=20，steps_per_epoch=500，with_shift_aug=False",
        "- 汇总来源：source-on-target eval log、TimeMatch train log、timematch_diag.tsv",
        f"- code_commit: `{_git_commit(root)}`",
        "",
        "## 运行状态",
        "",
        f"- log_root: `{log_root}`",
        f"- jobs: {len(summary)}",
        f"- failed_or_missing: {len(failed)}",
        "",
        "## Source Checkpoint Inventory",
        "",
        _markdown_table(inventory, inventory_fields),
        "## 配置均值",
        "",
        _markdown_table(by_config, config_fields),
        "## 相对 Base 对比",
        "",
        _markdown_table(_comparison_vs_base(by_config), comparison_fields),
        "## 任务-配置均值",
        "",
        _markdown_table(by_task, task_fields),
        "## 完整结果表",
        "",
        _markdown_table(summary, full_fields),
    ]
    if failed:
        text.extend(
            [
                "## 失败任务",
                "",
                _markdown_table(
                    failed,
                    ["task", "seed", "config", "status", "source_checkpoint", "log_path"],
                ),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    output = Path(args.output)
    summary = build_summary(log_root)
    _write_tsv(output, summary, SUMMARY_FIELDS)

    by_config = _aggregate_for_config(summary)
    by_task = _aggregate_for_task(summary)
    config_fields = ["config", "mean_test_f1", "std_test_f1", "mean_da_gain", "std_da_gain", "num_jobs", "num_success"]
    task_fields = ["task", "config", "mean_test_f1", "std_test_f1", "mean_da_gain", "num_seeds"]
    _write_tsv(log_root / "summary_by_config.tsv", by_config, config_fields)
    _write_tsv(log_root / "summary_by_task.tsv", by_task, task_fields)

    if args.report:
        write_report(Path(args.report), log_root, summary, by_config, by_task)

    print(f"Wrote {output}")
    print(f"Wrote {log_root / 'summary_by_config.tsv'}")
    print(f"Wrote {log_root / 'summary_by_task.tsv'}")
    if args.report:
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
