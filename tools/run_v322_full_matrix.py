#!/usr/bin/env python3
"""Resumable four-stage runner for the v3.2.2 full transfer matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DOMAIN_DATASETS = {
    # Canonical cleaned-code mapping used by launch_v28_cleaned_full12_source_train_4gpu.sh.
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}
DOMAIN_ORDER = ("AT1", "DK1", "FR1", "FR2")
FIXED_STRETCHES = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
STEP_ORDER = ("audit", "stretch", "global_only", "affine")
STEP_STATES = {
    "audit": "RUNNING_AUDIT",
    "stretch": "RUNNING_STRETCH",
    "global_only": "RUNNING_GLOBAL",
    "affine": "RUNNING_AFFINE",
}
DEFAULT_CHECKPOINT_PATTERNS = (
    "v28_cleaned_full12_source_{source}_smooth_k3_seed{seed}/fold_0/model.pt",
    "v28_cleaned_source_{source}_smooth_k3_seed{seed}/fold_0/model.pt",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MatrixUnit:
    source: str
    target: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.source}_to_{self.target}"

    @property
    def source_dataset(self) -> str:
        return DOMAIN_DATASETS[self.source]

    @property
    def target_dataset(self) -> str:
        return DOMAIN_DATASETS[self.target]


@dataclass(frozen=True)
class RunnerConfig:
    repository_root: Path
    data_root: Path
    output_root: Path
    repository_branch: str
    repository_commit: str
    gpus: Tuple[int, ...]
    max_workers: int
    python_executable: str = sys.executable
    checkpoint_patterns: Tuple[str, ...] = DEFAULT_CHECKPOINT_PATTERNS
    num_workers: int = 4
    data_loader_timeout: int = 60
    restart_failed: bool = False


@dataclass(frozen=True)
class UnitPaths:
    base: Path
    audit_json: Path
    audit_tsv: Path
    stretch_json: Path
    stretch_tsv: Path
    logs: Path
    status: Path
    done: Path
    lock: Path


@dataclass(frozen=True)
class UnitValidation:
    task: str
    seed: int
    state: str
    done: bool
    audit_valid: bool
    stretch_valid: bool
    global_valid: bool
    affine_valid: bool
    selected_shift: Optional[int] = None
    selected_stretch: Optional[float] = None
    stretch_margin: Optional[float] = None
    selected_at_boundary: Optional[bool] = None
    global_macro_f1: Optional[float] = None
    affine_macro_f1: Optional[float] = None
    delta_macro_f1: Optional[float] = None
    failure_message: str = ""


@dataclass(frozen=True)
class PreflightReport:
    passed: bool
    missing_checkpoints: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    checkpoint_rows: Tuple[Mapping[str, Any], ...] = ()


def build_units(seeds: Sequence[int] = (1, 2, 3)) -> List[MatrixUnit]:
    return [
        MatrixUnit(source, target, int(seed))
        for source in DOMAIN_ORDER
        for target in DOMAIN_ORDER
        if source != target
        for seed in seeds
    ]


def unit_paths(config: RunnerConfig, unit: MatrixUnit) -> UnitPaths:
    base = config.output_root / unit.name / f"seed_{unit.seed}"
    return UnitPaths(
        base=base,
        audit_json=base / "audit" / "temporal_audit.json",
        audit_tsv=base / "audit" / "temporal_audit.tsv",
        stretch_json=base / "stretch" / "stretch.json",
        stretch_tsv=base / "stretch" / "stretch.tsv",
        logs=base / "logs",
        status=base / "status.json",
        done=base / "DONE",
        lock=base / ".lock",
    )


def checkpoint_candidates(config: RunnerConfig, unit: MatrixUnit) -> List[Path]:
    return [
        config.repository_root
        / "outputs"
        / pattern.format(source=unit.source, target=unit.target, seed=unit.seed)
        for pattern in config.checkpoint_patterns
    ]


def resolve_checkpoint(config: RunnerConfig, unit: MatrixUnit) -> Path:
    for path in checkpoint_candidates(config, unit):
        if path.is_file():
            return path.resolve()
    choices = ", ".join(str(path) for path in checkpoint_candidates(config, unit))
    raise FileNotFoundError(f"missing checkpoint for {unit.name} seed={unit.seed}: {choices}")


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class UnitLock:
    def __init__(self, path: Path):
        self.path = path
        self.descriptor: Optional[int] = None

    def __enter__(self) -> "UnitLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.descriptor = os.open(
            str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
        os.write(self.descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty JSON: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected={expected!r}, actual={actual!r}")


def _status_document(paths: UnitPaths) -> Dict[str, Any]:
    if not paths.status.is_file():
        return {}
    return dict(_read_json(paths.status))


def validate_audit(config: RunnerConfig, unit: MatrixUnit, checkpoint: Path) -> Mapping[str, Any]:
    paths = unit_paths(config, unit)
    document = _read_json(paths.audit_json)
    if not paths.audit_tsv.is_file() or paths.audit_tsv.stat().st_size == 0:
        raise ValueError("missing temporal audit TSV")
    _require_equal(document.get("schema_version"), "v322-temporal-range-audit-v1", "audit schema")
    repository = document.get("repository", {})
    _require_equal(repository.get("branch"), config.repository_branch, "audit branch")
    _require_equal(repository.get("commit"), config.repository_commit, "audit commit")
    _require_equal(document.get("source", {}).get("domain"), unit.source_dataset, "audit source")
    _require_equal(document.get("target", {}).get("domain"), unit.target_dataset, "audit target")
    checkpoint_doc = document.get("checkpoint", {})
    _require_equal(checkpoint_doc.get("sha256"), file_sha256(checkpoint), "audit checkpoint")
    _require_equal(int(checkpoint_doc.get("source_seed")), unit.seed, "audit seed")
    if not document.get("embedding"):
        raise ValueError("audit embedding range is missing")
    return document


def validate_stretch(config: RunnerConfig, unit: MatrixUnit, checkpoint: Path) -> Mapping[str, Any]:
    paths = unit_paths(config, unit)
    document = _read_json(paths.stretch_json)
    if not paths.stretch_tsv.is_file() or paths.stretch_tsv.stat().st_size == 0:
        raise ValueError("missing stretch TSV")
    _require_equal(document.get("schema_version"), "v322-stretch-estimate-v1", "stretch schema")
    repository = document.get("repository", {})
    _require_equal(repository.get("branch"), config.repository_branch, "stretch branch")
    _require_equal(repository.get("commit"), config.repository_commit, "stretch commit")
    task = document.get("task", {})
    _require_equal(task.get("name"), unit.name, "stretch task")
    _require_equal(task.get("source"), unit.source_dataset, "stretch source")
    _require_equal(task.get("target"), unit.target_dataset, "stretch target")
    _require_equal(int(task.get("seed")), unit.seed, "stretch seed")
    _require_equal(document.get("checkpoint", {}).get("sha256"), file_sha256(checkpoint), "stretch checkpoint")
    if document.get("identity_audit", {}).get("passed") is not True:
        raise ValueError("stretch identity audit did not pass")
    stretch = document.get("stretch", {})
    selected = float(stretch.get("selected"))
    candidate_values = tuple(float(row["stretch"]) for row in stretch.get("candidates", []))
    if candidate_values != FIXED_STRETCHES:
        raise ValueError(f"unexpected stretch candidates: {candidate_values}")
    selected_rows = [row for row in stretch["candidates"] if float(row["stretch"]) == selected]
    if len(selected_rows) != 1:
        raise ValueError("selected stretch candidate must be unique")
    if selected_rows[0].get("valid") is not True or selected_rows[0].get("embedding_range_valid") is not True:
        raise ValueError("selected stretch candidate is invalid")
    if int(document.get("global_shift", {}).get("selected")) != document["global_shift"]["selected"]:
        raise ValueError("global shift must be an integer")
    return document


def _metric_path(output_path: Path, unit: MatrixUnit) -> Path:
    target_name = unit.target_dataset.replace("/", "_")
    return output_path / "fold_0" / f"test_metrics_{target_name}.json"


def validate_da_result(
    config: RunnerConfig,
    unit: MatrixUnit,
    checkpoint: Path,
    output_path: Path,
    mode: str,
) -> float:
    if mode not in {"global_only", "affine"}:
        raise ValueError(f"unknown DA mode: {mode}")
    train_config = _read_json(output_path / "train_config.json")
    _require_equal(train_config.get("source"), unit.source_dataset, "DA source")
    _require_equal(train_config.get("target"), unit.target_dataset, "DA target")
    _require_equal(int(train_config.get("seed")), unit.seed, "DA seed")
    _require_equal(train_config.get("timematch_shift_policy"), "fixed_initial_shift", "DA shift policy")
    if abs(float(train_config.get("timematch_shift_score_epsilon")) - 1e-5) > 1e-12:
        raise ValueError("DA epsilon mismatch")
    _require_equal(train_config.get("timematch_target_teacher_position_mode"), mode, "DA mode")
    if mode == "affine":
        paths = unit_paths(config, unit)
        _require_equal(Path(train_config.get("timematch_affine_stretch_json")).resolve(), paths.stretch_json.resolve(), "DA stretch JSON")
        _require_equal(train_config.get("timematch_affine_task"), unit.name, "DA affine task")
        _require_equal(train_config.get("timematch_affine_repository_branch"), config.repository_branch, "DA affine branch")
        _require_equal(train_config.get("timematch_affine_repository_commit"), config.repository_commit, "DA affine commit")
    if not (output_path / "fold_0" / "model.pt").is_file():
        raise ValueError(f"missing final student checkpoint: {output_path}")
    metrics = _read_json(_metric_path(output_path, unit))
    macro_f1 = float(metrics.get("macro_f1"))
    if not 0.0 <= macro_f1 <= 1.0:
        raise ValueError("test macro-F1 must be in [0, 1]")
    status = _status_document(unit_paths(config, unit))
    if status.get("checkpoint_sha256") and status["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("DA status checkpoint mismatch")
    return macro_f1


def validate_unit(config: RunnerConfig, unit: MatrixUnit) -> UnitValidation:
    paths = unit_paths(config, unit)
    status = _status_document(paths)
    state = str(status.get("state", "PENDING"))
    failure = str(status.get("failure_message", ""))
    audit_valid = stretch_valid = global_valid = affine_valid = False
    shift = None
    stretch_value = margin = None
    at_boundary = None
    global_f1 = affine_f1 = None
    try:
        checkpoint = resolve_checkpoint(config, unit)
        validate_audit(config, unit, checkpoint)
        audit_valid = True
        stretch_doc = validate_stretch(config, unit, checkpoint)
        stretch_valid = True
        shift = int(stretch_doc["global_shift"]["selected"])
        stretch_value = float(stretch_doc["stretch"]["selected"])
        margin = float(stretch_doc["stretch"]["margin"])
        at_boundary = bool(stretch_doc["stretch"]["at_boundary"])
        global_path = Path(status.get("global_output_path", ""))
        global_f1 = validate_da_result(config, unit, checkpoint, global_path, "global_only")
        global_valid = True
        affine_path = Path(status.get("affine_output_path", ""))
        affine_f1 = validate_da_result(config, unit, checkpoint, affine_path, "affine")
        affine_valid = True
    except (FileNotFoundError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        failure = str(error)
    done = bool(paths.done.is_file() and audit_valid and stretch_valid and global_valid and affine_valid)
    return UnitValidation(
        task=unit.name,
        seed=unit.seed,
        state="DONE" if done else state,
        done=done,
        audit_valid=audit_valid,
        stretch_valid=stretch_valid,
        global_valid=global_valid,
        affine_valid=affine_valid,
        selected_shift=shift,
        selected_stretch=stretch_value,
        stretch_margin=margin,
        selected_at_boundary=at_boundary,
        global_macro_f1=global_f1,
        affine_macro_f1=affine_f1,
        delta_macro_f1=(affine_f1 - global_f1) if global_f1 is not None and affine_f1 is not None else None,
        failure_message=failure,
    )


def first_required_step(config: RunnerConfig, unit: MatrixUnit) -> Optional[str]:
    paths = unit_paths(config, unit)
    try:
        checkpoint = resolve_checkpoint(config, unit)
        validate_audit(config, unit, checkpoint)
    except Exception:
        return "audit"
    try:
        validate_stretch(config, unit, checkpoint)
    except Exception:
        return "stretch"
    status = _status_document(paths)
    try:
        validate_da_result(config, unit, checkpoint, Path(status.get("global_output_path", "")), "global_only")
    except Exception:
        return "global_only"
    try:
        validate_da_result(config, unit, checkpoint, Path(status.get("affine_output_path", "")), "affine")
    except Exception:
        return "affine"
    return None


def next_attempt_dir(root: Path) -> Path:
    index = 1
    while (root / f"attempt_{index:03d}").exists():
        index += 1
    return root / f"attempt_{index:03d}"


def build_step_command(
    config: RunnerConfig,
    unit: MatrixUnit,
    paths: UnitPaths,
    step: str,
    *,
    checkpoint: Path,
    da_attempt: Optional[Path] = None,
) -> List[str]:
    python = config.python_executable
    root = config.repository_root
    if step == "audit":
        return [
            python, str(root / "tools" / "audit_v322_temporal_ranges.py"),
            "--data-root", str(config.data_root), "--source", unit.source_dataset,
            "--target", unit.target_dataset, "--checkpoint", str(checkpoint),
            "--source-method", "smooth_k3", "--source-seed", str(unit.seed),
            "--repository-branch", config.repository_branch,
            "--repository-commit", config.repository_commit,
            "--output-json", str(paths.audit_json), "--output-tsv", str(paths.audit_tsv),
        ]
    if step == "stretch":
        return [
            python, str(root / "tools" / "estimate_v322_stretch.py"),
            "--temporal-audit-json", str(paths.audit_json), "--checkpoint", str(checkpoint),
            "--data-root", str(config.data_root), "--source", unit.source_dataset,
            "--target", unit.target_dataset, "--task", unit.name,
            "--output-json", str(paths.stretch_json), "--output-tsv", str(paths.stretch_tsv),
            "--repository-branch", config.repository_branch,
            "--repository-commit", config.repository_commit, "--seed", str(unit.seed),
            "--device", "cuda", "--batch-size", "128", "--num-workers", str(config.num_workers),
            "--shift-score-epsilon", "1e-5", "--closed-set", "True",
        ]
    if step not in {"global_only", "affine"}:
        raise ValueError(f"unknown step: {step}")
    if da_attempt is None:
        da_attempt = paths.base / step / "attempt_001"
    command = [
        python, str(root / "train.py"), "--data_root", str(config.data_root),
        "--output_dir", str(da_attempt), "--closed_set", "True", "--with_shift_aug", "False",
        "--seed", str(unit.seed), "--device", "cuda", "--batch_size", "128",
        "--num_workers", str(config.num_workers), "--data_loader_timeout", str(config.data_loader_timeout),
        "-e", "run", "--source", unit.source_dataset, "--target", unit.target_dataset,
        "timematch", "--weights", str(checkpoint.parent.parent), "--epochs", "20",
        "--steps_per_epoch", "500", "--estimate_shift", "True", "--shift_estimator", "IS",
        "--sample_size", "100", "--timematch_shift_policy", "fixed_initial_shift",
        "--timematch_shift_score_epsilon", "1e-5", "--timematch_target_teacher_position_mode", step,
        "--timematch_diagnostic_task", unit.name, "--output_student", "True",
    ]
    if step == "affine":
        command.extend(
            [
                "--timematch_affine_stretch_json", str(paths.stretch_json),
                "--timematch_affine_task", unit.name,
                "--timematch_affine_repository_branch", config.repository_branch,
                "--timematch_affine_repository_commit", config.repository_commit,
            ]
        )
    return command


def _git_identity(root: Path) -> Optional[Tuple[str, str]]:
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(root), text=True, stderr=subprocess.DEVNULL
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return branch, commit


def preflight(
    config: RunnerConfig,
    *,
    units: Optional[Sequence[MatrixUnit]] = None,
    check_cli: bool = True,
) -> PreflightReport:
    units = list(units or build_units())
    errors: List[str] = []
    missing: List[str] = []
    rows: List[Mapping[str, Any]] = []
    if not config.repository_branch:
        errors.append("repository branch is empty")
    if re.fullmatch(r"[0-9a-fA-F]{40}", config.repository_commit) is None:
        errors.append("repository commit must be a full 40-character hash")
    identity = _git_identity(config.repository_root)
    if identity is not None and identity != (config.repository_branch, config.repository_commit):
        errors.append(
            "current Git identity does not match explicit identity: "
            f"current={identity}, explicit={(config.repository_branch, config.repository_commit)}"
        )
    if not config.gpus:
        errors.append("GPU list is empty")
    if config.max_workers < 1 or config.max_workers > len(config.gpus):
        errors.append("max_workers must be between 1 and the GPU count")
    for unit in units:
        try:
            checkpoint = resolve_checkpoint(config, unit)
            rows.append(
                {
                    "task": unit.name,
                    "seed": unit.seed,
                    "checkpoint": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                }
            )
        except FileNotFoundError as error:
            missing.append(str(error))
    required = [
        config.repository_root / "tools" / "audit_v322_temporal_ranges.py",
        config.repository_root / "tools" / "estimate_v322_stretch.py",
        config.repository_root / "train.py",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing CLI: {path}")
    parent = config.output_root if config.output_root.exists() else config.output_root.parent
    if not parent.exists() or not os.access(str(parent), os.W_OK):
        errors.append(f"output root is not writable: {config.output_root}")
    if check_cli and not errors:
        for path in required[:2]:
            result = subprocess.run(
                [config.python_executable, str(path), "--help"],
                cwd=str(config.repository_root), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False,
            )
            if result.returncode != 0:
                errors.append(f"CLI --help failed: {path}")
        result = subprocess.run(
            [config.python_executable, str(required[2]), "timematch", "--help"],
            cwd=str(config.repository_root), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False,
        )
        if result.returncode != 0:
            errors.append(f"TimeMatch CLI --help failed: {required[2]}")
    return PreflightReport(
        passed=not errors and not missing,
        missing_checkpoints=tuple(missing), errors=tuple(errors), checkpoint_rows=tuple(rows)
    )


def _base_status(config: RunnerConfig, unit: MatrixUnit, checkpoint: Path) -> Dict[str, Any]:
    return {
        "task": unit.name, "seed": unit.seed, "state": "PENDING", "current_step": "",
        "repository_branch": config.repository_branch,
        "repository_commit": config.repository_commit,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "audit_path": str(unit_paths(config, unit).audit_json),
        "stretch_path": str(unit_paths(config, unit).stretch_json),
        "selected_b": None, "selected_a": None, "anchor": None,
        "global_output_path": "", "affine_output_path": "",
        "started_at": utc_now(), "finished_at": None,
        "exit_codes": {}, "commands": {}, "processes": {}, "failure_message": "",
    }


def _archive_file(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = path.with_name(f"{path.name}.invalid_{stamp}")
        counter = 1
        while destination.exists():
            destination = path.with_name(f"{path.name}.invalid_{stamp}_{counter}")
            counter += 1
        path.replace(destination)


def _validate_step(config: RunnerConfig, unit: MatrixUnit, checkpoint: Path, step: str, status: Mapping[str, Any]) -> None:
    if step == "audit":
        validate_audit(config, unit, checkpoint)
    elif step == "stretch":
        validate_stretch(config, unit, checkpoint)
    elif step == "global_only":
        validate_da_result(config, unit, checkpoint, Path(status["global_output_path"]), step)
    elif step == "affine":
        validate_da_result(config, unit, checkpoint, Path(status["affine_output_path"]), step)
    else:
        raise ValueError(step)


def _run_command(command: Sequence[str], *, cwd: Path, gpu: int, log_path: Path) -> Tuple[int, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            list(command), cwd=str(cwd), env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
        return process.pid, process.wait()


def run_unit(
    config: RunnerConfig,
    unit: MatrixUnit,
    gpu: int,
    *,
    command_runner=_run_command,
) -> bool:
    paths = unit_paths(config, unit)
    paths.base.mkdir(parents=True, exist_ok=True)
    try:
        with UnitLock(paths.lock):
            checkpoint = resolve_checkpoint(config, unit)
            status = _status_document(paths) or _base_status(config, unit, checkpoint)
            if (
                status.get("repository_branch") != config.repository_branch
                or status.get("repository_commit") != config.repository_commit
                or status.get("checkpoint_sha256") != file_sha256(checkpoint)
            ):
                raise ValueError("existing status identity does not match current commit/checkpoint")
            for field_name in ("commands", "processes", "exit_codes"):
                status.setdefault(field_name, {})
            if status.get("state") == "FAILED" and not config.restart_failed:
                return False
            required = first_required_step(config, unit)
            if required is None:
                status.update(state="DONE", current_step="", finished_at=utc_now())
                atomic_write_json(paths.status, status)
                atomic_write_text(paths.done, "DONE\n")
                return True
            start_index = STEP_ORDER.index(required)
            for step in STEP_ORDER[start_index:]:
                if step == "audit":
                    _archive_file(paths.audit_json)
                    _archive_file(paths.audit_tsv)
                    log_path = paths.logs / "audit.log"
                    da_attempt = None
                elif step == "stretch":
                    _archive_file(paths.stretch_json)
                    _archive_file(paths.stretch_tsv)
                    log_path = paths.logs / "stretch.log"
                    da_attempt = None
                else:
                    da_attempt = next_attempt_dir(paths.base / step)
                    output_path = da_attempt / "run"
                    status[f"{'global' if step == 'global_only' else 'affine'}_output_path"] = str(output_path)
                    log_path = paths.logs / f"{step}.{da_attempt.name}.log"
                _archive_file(log_path)
                command = build_step_command(
                    config, unit, paths, step, checkpoint=checkpoint, da_attempt=da_attempt
                )
                status.update(state=STEP_STATES[step], current_step=step, assigned_gpu=gpu)
                status["commands"][step] = command
                status["processes"][step] = {"gpu": gpu, "log_path": str(log_path), "started_at": utc_now()}
                atomic_write_json(paths.status, status)
                pid, exit_code = command_runner(
                    command, cwd=config.repository_root, gpu=gpu, log_path=log_path
                )
                status["processes"][step].update(pid=pid, finished_at=utc_now(), exit_code=exit_code)
                status["exit_codes"][step] = exit_code
                atomic_write_json(paths.status, status)
                if exit_code != 0:
                    raise RuntimeError(f"{step} exited with status {exit_code}; log={log_path}")
                _validate_step(config, unit, checkpoint, step, status)
                if step == "stretch":
                    stretch = _read_json(paths.stretch_json)
                    status.update(
                        selected_b=int(stretch["global_shift"]["selected"]),
                        selected_a=float(stretch["stretch"]["selected"]),
                        anchor=float(stretch["position"]["anchor"]),
                    )
                atomic_write_json(paths.status, status)
            final = validate_unit(config, unit)
            if not (final.audit_valid and final.stretch_valid and final.global_valid and final.affine_valid):
                raise ValueError(final.failure_message or "final unit validation failed")
            status.update(state="DONE", current_step="", finished_at=utc_now(), failure_message="")
            atomic_write_json(paths.status, status)
            atomic_write_text(paths.done, "DONE\n")
            return True
    except FileExistsError:
        # Another controller owns this unit; never mutate its status concurrently.
        return False
    except Exception as error:
        status = _status_document(paths)
        status.update(
            task=unit.name, seed=unit.seed, state="FAILED", finished_at=utc_now(),
            failure_message=f"{type(error).__name__}: {error}",
        )
        atomic_write_json(paths.status, status)
        paths.logs.mkdir(parents=True, exist_ok=True)
        with (paths.logs / "controller_error.log").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(f"[{utc_now()}] {unit.name} seed={unit.seed}\n")
            handle.write(traceback.format_exc())
            handle.write("\n")
        return False


def run_matrix(
    config: RunnerConfig,
    units: Optional[Sequence[MatrixUnit]] = None,
    *,
    unit_runner=run_unit,
    check_cli: bool = True,
) -> bool:
    units = list(units or build_units())
    report = preflight(config, units=units, check_cli=check_cli)
    if not report.passed:
        for message in list(report.errors) + list(report.missing_checkpoints):
            print(f"PREFLIGHT_ERROR|{message}", flush=True)
        return False
    config.output_root.mkdir(parents=True, exist_ok=True)
    worker_gpus = config.gpus[: config.max_workers]
    assignments = [units[index::len(worker_gpus)] for index in range(len(worker_gpus))]
    outcomes: List[bool] = []
    outcome_lock = threading.Lock()

    def worker(gpu: int, assigned: Sequence[MatrixUnit]) -> None:
        for unit in assigned:
            result = unit_runner(config, unit, gpu)
            with outcome_lock:
                outcomes.append(result)

    threads = [
        threading.Thread(target=worker, args=(gpu, assigned), name=f"gpu-{gpu}")
        for gpu, assigned in zip(worker_gpus, assignments)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summarize_matrix(config, units=units)
    return len(outcomes) == len(units) and all(outcomes)


RESULT_FIELDS = (
    "task", "seed", "state", "audit_valid", "stretch_valid", "global_valid",
    "affine_valid", "selected_shift", "selected_stretch", "stretch_margin",
    "selected_at_boundary", "global_macro_f1", "affine_macro_f1",
    "delta_macro_f1", "failure_message",
)


def summarize_matrix(
    config: RunnerConfig, *, units: Optional[Sequence[MatrixUnit]] = None
) -> Mapping[str, Any]:
    units = list(units or build_units())
    validations = [validate_unit(config, unit) for unit in units]
    rows = [
        {field_name: getattr(item, field_name) for field_name in RESULT_FIELDS}
        for item in validations
    ]
    status = {
        "schema_version": "v322-full-matrix-status-v1",
        "repository": {"branch": config.repository_branch, "commit": config.repository_commit},
        "expected_units": len(units),
        "done": sum(item.done for item in validations),
        "failed": sum(item.state == "FAILED" for item in validations),
        "not_started": sum(item.state == "PENDING" for item in validations),
        "audit_success": sum(item.audit_valid for item in validations),
        "stretch_success": sum(item.stretch_valid for item in validations),
        "global_success": sum(item.global_valid for item in validations),
        "affine_success": sum(item.affine_valid for item in validations),
        "selected_at_boundary": sum(item.selected_at_boundary is True for item in validations),
        "rows": rows,
    }
    config.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.output_root / "matrix_status.json", status)
    tsv_path = config.output_root / "matrix_results.tsv"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".matrix_results.", suffix=".tmp", dir=str(config.output_root))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(tsv_path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": status, "rows": rows}


def _parse_gpus(value: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("GPU ids must be unique")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "run", "validate"):
        child = subparsers.add_parser(mode)
        child.add_argument("--repository-root", default=str(ROOT))
        child.add_argument("--data-root", required=True)
        child.add_argument("--output-root", required=True)
        child.add_argument("--repository-branch", required=True)
        child.add_argument("--repository-commit", required=True)
        child.add_argument("--gpus", type=_parse_gpus, required=True)
        child.add_argument("--max-workers", type=int, default=None)
        child.add_argument("--python", dest="python_executable", default=sys.executable)
        child.add_argument("--num-workers", type=int, default=4)
        child.add_argument("--data-loader-timeout", type=int, default=60)
        child.add_argument("--checkpoint-pattern", action="append", default=[])
        child.add_argument("--restart-failed", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> RunnerConfig:
    patterns = tuple(args.checkpoint_pattern) or DEFAULT_CHECKPOINT_PATTERNS
    max_workers = args.max_workers if args.max_workers is not None else len(args.gpus)
    return RunnerConfig(
        repository_root=Path(args.repository_root).expanduser().resolve(),
        data_root=Path(args.data_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        repository_branch=args.repository_branch,
        repository_commit=args.repository_commit,
        gpus=tuple(args.gpus), max_workers=max_workers,
        python_executable=args.python_executable,
        checkpoint_patterns=patterns, num_workers=args.num_workers,
        data_loader_timeout=args.data_loader_timeout,
        restart_failed=args.restart_failed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    if args.mode == "validate":
        summary = summarize_matrix(config)
        print(json.dumps(summary["status"], indent=2), flush=True)
        return 0 if summary["status"]["done"] == summary["status"]["expected_units"] else 1
    report = preflight(config)
    print(
        f"V322_MATRIX_PREFLIGHT|passed={report.passed}|units=36|"
        f"checkpoints={len(report.checkpoint_rows)}|missing={len(report.missing_checkpoints)}|"
        f"errors={len(report.errors)}",
        flush=True,
    )
    for message in list(report.errors) + list(report.missing_checkpoints):
        print(f"PREFLIGHT_ERROR|{message}", flush=True)
    if args.mode == "plan":
        return 0 if report.passed else 2
    if not report.passed:
        return 2
    return 0 if run_matrix(config) else 1


if __name__ == "__main__":
    raise SystemExit(main())
