"""Summarize v3.2.1 Stage 3b equivalence and overhead audit."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


FIELDS = [
    "task",
    "config",
    "status",
    "val_macro_f1",
    "test_macro_f1",
    "global_shift",
    "local_shift_abs_mean",
    "local_shift_clip_fraction",
    "alignment_entropy",
    "alignment_top1_mass",
    "pseudo_confidence",
    "pseudo_ratio",
    "source_loss",
    "target_loss",
    "total_loss",
    "epoch_time_s",
    "source_forward_time_ms",
    "teacher_pseudo_time_ms",
    "target_temporal_feature_time_ms",
    "target_normal_forward_time_ms",
    "partition_time_ms",
    "alignment_time_ms",
    "local_position_time_ms",
    "target_forward_from_features_time_ms",
    "backward_time_ms",
    "logging_time_ms",
    "runtime_seconds",
    "log",
    "diagnostic_log",
    "local_shift_log",
]


def _read_tsv(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _last_row(path: str) -> Dict[str, str]:
    rows = _read_tsv(path)
    return rows[-1] if rows else {}


def _last_match(path: str, pattern: str) -> Optional[re.Match[str]]:
    if not path or not os.path.exists(path):
        return None
    regex = re.compile(pattern)
    last = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = regex.search(line)
            if match:
                last = match
    return last


def _extract_validation_f1(path: str) -> str:
    match = _last_match(path, r"Validation result:\s*loss=([-+0-9.eE]+),\s*acc=([-+0-9.eE]+),\s*f1=([-+0-9.eE]+)")
    return match.group(3) if match else ""


def _extract_test_f1(path: str) -> str:
    match = _last_match(path, r"Test result for .*?:\s*accuracy=([-+0-9.eE]+),\s*f1=([-+0-9.eE]+)")
    return match.group(2) if match else ""


def _pick(row: Dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return str(value)
    return ""


def _float(value: str) -> Optional[float]:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except ValueError:
        return None


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summarize_job(job: Dict[str, str]) -> Dict[str, str]:
    run_log = job.get("log", "")
    diagnostic_log = job.get("diagnostic_log", "")
    local_shift_log = job.get("local_shift_log", "")
    local = _last_row(local_shift_log)
    diag = _last_row(diagnostic_log)
    is_local = bool(local)
    metric_row = local if is_local else diag
    row = {
        "task": job.get("task", ""),
        "config": job.get("config", ""),
        "status": job.get("status", ""),
        "val_macro_f1": _extract_validation_f1(run_log),
        "test_macro_f1": _extract_test_f1(run_log),
        "global_shift": _pick(local, "global_shift") or _pick(diag, "estimated_shift_t_to_s"),
        "local_shift_abs_mean": _pick(local, "local_shift_abs_mean"),
        "local_shift_clip_fraction": _pick(local, "local_shift_clip_fraction"),
        "alignment_entropy": _pick(local, "alignment_entropy"),
        "alignment_top1_mass": _pick(local, "alignment_top1_mass"),
        "pseudo_confidence": _pick(local, "target_pseudo_confidence") or _pick(diag, "teacher_pseudo_confidence_mean"),
        "pseudo_ratio": _pick(local, "target_pseudo_ratio") or _pick(diag, "teacher_pseudo_coverage"),
        "source_loss": _pick(metric_row, "source_loss"),
        "target_loss": _pick(metric_row, "target_loss"),
        "total_loss": _pick(metric_row, "total_loss"),
        "runtime_seconds": job.get("duration_seconds", ""),
        "log": run_log,
        "diagnostic_log": diagnostic_log,
        "local_shift_log": local_shift_log,
    }
    for name in FIELDS:
        if name.endswith("_time_ms") or name == "epoch_time_s":
            row[name] = _pick(local, name)
    return row


def _write_tsv(path: str, rows: Iterable[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: List[Dict[str, str]], fields: List[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            number = _float(value)
            if number is not None and field not in {"global_shift", "runtime_seconds", "status"}:
                value = f"{number:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _diff_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row["task"]][row["config"]] = row
    output = []
    for task, configs in sorted(by_task.items()):
        smooth = _float(configs.get("smooth_base", {}).get("test_macro_f1", ""))
        base_equiv = _float(configs.get("base_equiv", {}).get("test_macro_f1", ""))
        global_forward = _float(configs.get("global_forward", {}).get("test_macro_f1", ""))
        global_only = _float(configs.get("global_only", {}).get("test_macro_f1", ""))
        residual = _float(configs.get("residual", {}).get("test_macro_f1", ""))
        output.append(
            {
                "task": task,
                "smooth_base": smooth,
                "base_equiv": base_equiv,
                "global_forward": global_forward,
                "global_only": global_only,
                "residual": residual,
                "base_equiv-smooth": None if smooth is None or base_equiv is None else base_equiv - smooth,
                "global_forward-base_equiv": None
                if base_equiv is None or global_forward is None
                else global_forward - base_equiv,
                "global_only-global_forward": None
                if global_forward is None or global_only is None
                else global_only - global_forward,
                "residual-global_only": None
                if global_only is None or residual is None
                else residual - global_only,
            }
        )
    return output


def _write_report(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    diff = _diff_rows(rows)
    compact_fields = [
        "task",
        "config",
        "status",
        "val_macro_f1",
        "test_macro_f1",
        "global_shift",
        "local_shift_abs_mean",
        "local_shift_clip_fraction",
        "pseudo_confidence",
        "pseudo_ratio",
        "target_loss",
        "runtime_seconds",
    ]
    timing_fields = [
        "task",
        "config",
        "epoch_time_s",
        "source_forward_time_ms",
        "teacher_pseudo_time_ms",
        "target_temporal_feature_time_ms",
        "target_normal_forward_time_ms",
        "partition_time_ms",
        "alignment_time_ms",
        "local_position_time_ms",
        "target_forward_from_features_time_ms",
        "backward_time_ms",
        "logging_time_ms",
    ]
    diff_fields = [
        "task",
        "smooth_base",
        "base_equiv",
        "global_forward",
        "global_only",
        "residual",
        "base_equiv-smooth",
        "global_forward-base_equiv",
        "global_only-global_forward",
        "residual-global_only",
    ]
    threshold = 0.01
    break_lines = []
    control_lines = []
    forward_lines = []
    residual_lines = []
    for row in diff:
        task = str(row["task"])
        base_delta = row.get("base_equiv-smooth")
        forward_delta = row.get("global_forward-base_equiv")
        control_delta = row.get("global_only-global_forward")
        residual_delta = row.get("residual-global_only")
        first_break = "none"
        if isinstance(base_delta, float) and abs(base_delta) > threshold:
            first_break = "smooth_base -> base_equiv"
        elif isinstance(forward_delta, float) and abs(forward_delta) > threshold:
            first_break = "base_equiv -> global_forward"
        elif isinstance(control_delta, float) and abs(control_delta) > threshold:
            first_break = "global_forward -> global_only"
        elif isinstance(residual_delta, float) and abs(residual_delta) > threshold:
            first_break = "global_only -> residual"
        break_lines.append(f"- {task}: first_break={first_break}")
        control_lines.append(
            f"- {task}: global_only-global_forward={_fmt(control_delta)}; "
            f"valid_control={abs(control_delta) <= threshold if isinstance(control_delta, float) else 'unknown'}"
        )
        forward_lines.append(
            f"- {task}: global_forward-base_equiv={_fmt(forward_delta)}; "
            f"forward_equiv={abs(forward_delta) <= threshold if isinstance(forward_delta, float) else 'unknown'}"
        )
        residual_lines.append(f"- {task}: residual-global_only={_fmt(residual_delta)}")
    title = "v3.2.1 Stage 3c Equivalence Fix Report" if "stage3c" in os.path.basename(path).lower() else "v3.2.1 Stage 3b Equivalence Audit"
    text = [
        f"# {title}",
        "",
        "## 1. Comparison Table",
        "",
        _markdown_table(rows, compact_fields),
        "",
        "## 2. Equivalence Deltas",
        "",
        _markdown_table([{k: _fmt(v) for k, v in row.items()} for row in diff], diff_fields),
        "",
        "## 3. Timing Breakdown",
        "",
        _markdown_table(rows, timing_fields),
        "",
        "## 4. Where Equivalence First Breaks",
        "",
        "\n".join(break_lines),
        "",
        "## 5. Control Validity",
        "",
        "\n".join(control_lines),
        "",
        "## 6. Forward Equivalence",
        "",
        "\n".join(forward_lines),
        "",
        "## 7. Residual Signal",
        "",
        "\n".join(residual_lines),
        "",
        "## 8. Runtime Acceptability",
        "",
        "Inspect the timing table above. This report records timing only and does not judge method success.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    status_path = os.path.join(args.log_dir, "job_status.tsv")
    jobs = _read_tsv(status_path)
    if not jobs:
        raise SystemExit(f"No jobs found in {status_path}")
    rows = [_summarize_job(job) for job in jobs]
    rows.sort(key=lambda item: (item.get("task", ""), item.get("config", "")))
    _write_tsv(args.summary, rows)
    _write_report(args.report, rows)
    print(f"SUMMARY_WRITTEN|path={args.summary}|rows={len(rows)}")
    print(f"REPORT_WRITTEN|path={args.report}")


if __name__ == "__main__":
    main()
