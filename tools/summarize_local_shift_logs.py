"""Summarize v3.2.1 local-shift launcher logs.

Expected usage:

    python tools/summarize_local_shift_logs.py \
      --log_root logs/<RUN_TAG> \
      --output logs/<RUN_TAG>/summary.tsv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional


FIELDS = [
    "task",
    "config",
    "gpu",
    "pid",
    "status",
    "runtime_seconds",
    "val_macro_f1",
    "test_macro_f1",
    "global_shift",
    "local_shift_abs_mean",
    "local_shift_mean",
    "local_shift_std",
    "local_shift_clip_fraction",
    "residual_gate_keep_ratio",
    "residual_alpha",
    "residual_zero_mean",
    "alignment_entropy",
    "alignment_top1_mass",
    "alignment_valid_ratio",
    "alignment_fallback_ratio",
    "pseudo_confidence",
    "pseudo_ratio",
    "source_loss",
    "target_loss",
    "total_loss",
    "epoch_time_s",
    "partition_time_ms",
    "alignment_time_ms",
    "local_position_time_ms",
    "target_forward_from_features_time_ms",
    "log_path",
    "local_shift_log",
]


def _read_tsv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
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


def _pick(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return str(value)
    return ""


def _summarize(job: Dict[str, str]) -> Dict[str, str]:
    log_path = job.get("log_path") or job.get("log") or ""
    local_shift_log = job.get("local_shift_log") or os.path.join(
        os.path.dirname(log_path),
        "local_shift.tsv",
    )
    local = _last_row(local_shift_log)
    return {
        "task": job.get("task", ""),
        "config": job.get("config", ""),
        "gpu": job.get("gpu", ""),
        "pid": job.get("pid", ""),
        "status": job.get("status", ""),
        "runtime_seconds": job.get("runtime_seconds") or job.get("duration_seconds", ""),
        "val_macro_f1": _extract_validation_f1(log_path),
        "test_macro_f1": _extract_test_f1(log_path),
        "global_shift": _pick(local, "global_shift"),
        "local_shift_abs_mean": _pick(local, "local_shift_abs_mean"),
        "local_shift_mean": _pick(local, "local_shift_mean"),
        "local_shift_std": _pick(local, "local_shift_std"),
        "local_shift_clip_fraction": _pick(local, "local_shift_clip_fraction"),
        "residual_gate_keep_ratio": _pick(local, "residual_gate_keep_ratio"),
        "residual_alpha": _pick(local, "residual_alpha"),
        "residual_zero_mean": _pick(local, "residual_zero_mean"),
        "alignment_entropy": _pick(local, "alignment_entropy"),
        "alignment_top1_mass": _pick(local, "alignment_top1_mass"),
        "alignment_valid_ratio": _pick(local, "alignment_valid_ratio"),
        "alignment_fallback_ratio": _pick(local, "alignment_fallback_ratio"),
        "pseudo_confidence": _pick(local, "target_pseudo_confidence"),
        "pseudo_ratio": _pick(local, "target_pseudo_ratio"),
        "source_loss": _pick(local, "source_loss"),
        "target_loss": _pick(local, "target_loss"),
        "total_loss": _pick(local, "total_loss"),
        "epoch_time_s": _pick(local, "epoch_time_s"),
        "partition_time_ms": _pick(local, "partition_time_ms"),
        "alignment_time_ms": _pick(local, "alignment_time_ms"),
        "local_position_time_ms": _pick(local, "local_position_time_ms"),
        "target_forward_from_features_time_ms": _pick(local, "target_forward_from_features_time_ms"),
        "log_path": log_path,
        "local_shift_log": local_shift_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", nargs="?", help="legacy single TSV input; only counts tasks")
    parser.add_argument("--log_root", default="", help="launcher log root containing job_status.tsv")
    parser.add_argument("--output", default="", help="summary TSV output path")
    args = parser.parse_args()

    if args.log_root:
        status_path = os.path.join(args.log_root, "job_status.tsv")
        jobs = _read_tsv(status_path)
        if not jobs:
            raise SystemExit(f"No jobs found in {status_path}")
        rows = [_summarize(job) for job in jobs]
        rows.sort(key=lambda item: (item.get("task", ""), item.get("config", "")))
        output = args.output or os.path.join(args.log_root, "summary.tsv")
        with open(output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"SUMMARY_WRITTEN|path={output}|rows={len(rows)}")
        return

    if not args.tsv:
        raise SystemExit("Provide --log_root or a legacy TSV path")
    counts: Dict[str, int] = {}
    for row in _read_tsv(args.tsv):
        task = row.get("task", "unknown")
        counts[task] = counts.get(task, 0) + 1
    for task, count in sorted(counts.items()):
        print(f"{task}\t{count}")


if __name__ == "__main__":
    main()
