"""Summarize v3.2.1 Stage 3 probe logs into one compact TSV."""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, Iterable, List, Optional


SUMMARY_FIELDS = [
    "task",
    "config",
    "source_kind",
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
    "target_loss",
    "runtime_seconds",
    "log",
    "diagnostic_log",
    "local_shift_log",
    "source_dir",
    "reference_path",
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


def _summarize_job(job: Dict[str, str]) -> Dict[str, str]:
    run_log = job.get("log", "")
    diagnostic_log = job.get("diagnostic_log", "")
    local_shift_log = job.get("local_shift_log", "")
    config = job.get("config", "")

    diag = _last_row(diagnostic_log)
    local = _last_row(local_shift_log)

    if config in {"global_only", "residual"}:
        metric_row = local
        global_shift = _pick(local, "global_shift")
        pseudo_conf = _pick(local, "target_pseudo_confidence", "pseudo_confidence")
        pseudo_ratio = _pick(local, "target_pseudo_ratio", "pseudo_ratio")
    else:
        metric_row = diag
        global_shift = _pick(diag, "estimated_shift_t_to_s", "target_to_source_shift", "global_shift")
        pseudo_conf = _pick(
            diag,
            "pseudo_confidence",
            "target_pseudo_confidence",
            "teacher_pseudo_confidence_mean",
        )
        pseudo_ratio = _pick(
            diag,
            "pseudo_ratio",
            "target_pseudo_ratio",
            "teacher_pseudo_coverage",
        )

    row = {
        "task": job.get("task", ""),
        "config": config,
        "source_kind": job.get("source_kind", ""),
        "status": job.get("status", ""),
        "val_macro_f1": _extract_validation_f1(run_log),
        "test_macro_f1": _extract_test_f1(run_log),
        "global_shift": global_shift,
        "local_shift_abs_mean": _pick(local, "local_shift_abs_mean"),
        "local_shift_clip_fraction": _pick(local, "local_shift_clip_fraction"),
        "alignment_entropy": _pick(local, "alignment_entropy"),
        "alignment_top1_mass": _pick(local, "alignment_top1_mass"),
        "pseudo_confidence": pseudo_conf,
        "pseudo_ratio": pseudo_ratio,
        "target_loss": _pick(metric_row, "target_loss"),
        "runtime_seconds": job.get("duration_seconds", ""),
        "log": run_log,
        "diagnostic_log": diagnostic_log,
        "local_shift_log": local_shift_log,
        "source_dir": job.get("source_dir", ""),
        "reference_path": job.get("reference_path", ""),
    }
    return row


def _write_tsv(path: str, rows: Iterable[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    status_path = os.path.join(args.log_dir, "job_status.tsv")
    jobs = _read_tsv(status_path)
    if not jobs:
        raise SystemExit(f"No jobs found in {status_path}")

    rows = [_summarize_job(job) for job in jobs]
    rows.sort(key=lambda item: (item.get("task", ""), item.get("config", "")))
    _write_tsv(args.output, rows)
    print(f"SUMMARY_WRITTEN|path={args.output}|rows={len(rows)}")


if __name__ == "__main__":
    main()
