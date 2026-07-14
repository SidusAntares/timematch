"""Validated fixed affine positions for the v3.2.2 target teacher path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import torch

from .affine_transform import transform_positions


STRETCH_ESTIMATE_SCHEMA = "v322-stretch-estimate-v1"


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TeacherAffineSpec:
    task: str
    source: str
    target: str
    seed: int
    checkpoint_sha256: str
    repository_branch: str
    repository_commit: str
    global_shift: int
    stretch: float
    anchor: float

    def validate_global_shift(self, estimated_shift: int) -> None:
        actual = int(estimated_shift)
        if actual != self.global_shift:
            raise ValueError(
                "v3.2.2 global shift mismatch: "
                f"formal_json={self.global_shift}, runtime={actual}, task={self.task}"
            )


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(
            f"v3.2.2 {name} mismatch: formal_json={actual!r}, runtime={expected!r}"
        )


def load_teacher_affine_spec(
    json_path: Path,
    *,
    checkpoint_path: Path,
    expected_task: str,
    expected_source: str,
    expected_target: str,
    expected_seed: int,
    expected_repository_branch: str,
    expected_repository_commit: str,
) -> TeacherAffineSpec:
    """Load a formal Stage 3 result and bind it to the current DA run."""

    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"formal stretch JSON does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != STRETCH_ESTIMATE_SCHEMA:
        raise ValueError(
            f"formal stretch JSON schema must be {STRETCH_ESTIMATE_SCHEMA!r}"
        )
    if document.get("identity_audit", {}).get("passed") is not True:
        raise ValueError("formal stretch JSON identity audit did not pass")

    repository = document.get("repository", {})
    repository_branch = str(repository.get("branch", ""))
    repository_commit = str(repository.get("commit", ""))
    expected_branch = str(expected_repository_branch)
    expected_commit = str(expected_repository_commit).lower()
    if not expected_branch:
        raise ValueError("expected repository branch must be non-empty")
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected repository commit must be a full 40-character hash")
    _require_equal(repository_branch, expected_branch, "repository branch")
    _require_equal(repository_commit.lower(), expected_commit, "repository commit")

    task = document.get("task", {})
    _require_equal(task.get("name"), str(expected_task), "task")
    _require_equal(task.get("source"), str(expected_source), "source domain")
    _require_equal(task.get("target"), str(expected_target), "target domain")
    _require_equal(int(task.get("seed")), int(expected_seed), "seed")

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {checkpoint}")
    expected_sha256 = str(document.get("checkpoint", {}).get("sha256", "")).lower()
    if len(expected_sha256) != 64:
        raise ValueError("formal stretch JSON checkpoint SHA256 is invalid")
    actual_sha256 = _file_sha256(checkpoint)
    if actual_sha256.lower() != expected_sha256:
        raise ValueError(
            "source checkpoint SHA256 mismatch: "
            f"formal_json={expected_sha256}, runtime={actual_sha256}"
        )

    stretch_section = document.get("stretch", {})
    selected_stretch = _finite_float(stretch_section.get("selected"), "stretch")
    selected_rows = [
        row
        for row in stretch_section.get("candidates", [])
        if float(row.get("stretch")) == selected_stretch
    ]
    if len(selected_rows) != 1:
        raise ValueError("formal JSON must contain exactly one selected stretch candidate")
    selected_row: Mapping[str, Any] = selected_rows[0]
    if (
        selected_row.get("valid") is not True
        or selected_row.get("embedding_range_valid") is not True
    ):
        raise ValueError("selected stretch candidate is not valid")

    return TeacherAffineSpec(
        task=str(task["name"]),
        source=str(task["source"]),
        target=str(task["target"]),
        seed=int(task["seed"]),
        checkpoint_sha256=actual_sha256,
        repository_branch=repository_branch,
        repository_commit=repository_commit.lower(),
        global_shift=int(document["global_shift"]["selected"]),
        stretch=selected_stretch,
        anchor=_finite_float(document["position"]["anchor"], "anchor"),
    )


def resolve_target_teacher_positions(
    positions: torch.Tensor,
    global_shift: int,
    affine_spec: Optional[TeacherAffineSpec],
) -> torch.Tensor:
    """Return global-only positions or the validated fixed affine alternative."""

    if affine_spec is None:
        return positions + int(global_shift)
    affine_spec.validate_global_shift(global_shift)
    return transform_positions(
        positions,
        shift=int(global_shift),
        stretch=affine_spec.stretch,
        anchor=affine_spec.anchor,
    ).discrete_positions


__all__ = [
    "TeacherAffineSpec",
    "load_teacher_affine_spec",
    "resolve_target_teacher_positions",
]
