"""Pure candidate selection and serialization for v3.2.2 stretch estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .affine_transform import AffineTransformResult
from .range_gate import RangeStatistics


TEMPORAL_AUDIT_SCHEMA = "v322-temporal-range-audit-v1"
STRETCH_ESTIMATE_SCHEMA = "v322-stretch-estimate-v1"
STRETCH_CANDIDATES: Tuple[float, ...] = (
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
)


@dataclass(frozen=True)
class StretchCandidateResult:
    stretch: float
    score: Optional[float]
    valid: bool
    embedding_range_valid: bool
    transformed_min: float
    transformed_max: float
    source_overlap_span: float
    source_overlap_ratio: float
    in_source_point_ratio: float
    duplicate_count: int
    duplicate_ratio: float


@dataclass(frozen=True)
class StretchEstimateResult:
    global_shift: int
    anchor: float
    selected_stretch: float
    selected_score: float
    second_best_stretch: Optional[float]
    second_best_score: Optional[float]
    top1_top2_margin: Optional[float]
    selected_at_boundary: bool
    candidates: Tuple[StretchCandidateResult, ...]


def fixed_stretch_candidates() -> Tuple[float, ...]:
    """Return the immutable v3.2.2 candidate grid."""

    return STRETCH_CANDIDATES


def create_target_train_indices(
    source_count: int,
    target_count: int,
    *,
    val_ratio: float,
    test_ratio: float,
) -> set:
    """Reproduce train.py's source-then-target fold shuffle for target train."""

    target_train = None
    for domain_index, count in enumerate((source_count, target_count)):
        indices = list(range(int(count)))
        random.shuffle(indices)
        test_count = int(test_ratio * count)
        val_count = int(val_ratio * count)
        train_count = count - test_count - val_count
        if domain_index == 1:
            target_train = set(indices[:train_count])
    if target_train is None:
        raise RuntimeError("target train split was not constructed")
    return target_train


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar_mean(value: torch.Tensor, name: str) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty tensor")
    return _finite_float(value.to(torch.float64).mean().item(), name)


def build_candidate_result(
    stretch: float,
    score: Optional[float],
    transformed: AffineTransformResult,
    statistics: RangeStatistics,
) -> StretchCandidateResult:
    """Combine one score with affine and range diagnostics.

    No overlap or duplicate threshold is applied. A candidate is valid exactly
    when all transformed points stay inside the positional embedding range.
    """

    stretch_value = _finite_float(stretch, "stretch")
    if stretch_value <= 0:
        raise ValueError("stretch must be positive")
    if transformed.discrete_positions.numel() == 0:
        raise ValueError("transformed positions must be nonempty")
    embedding_valid = bool(statistics.embedding_range_valid.all().item())
    if score is None:
        if embedding_valid:
            raise ValueError("a valid candidate must have a score")
        score_value = None
    else:
        score_value = _finite_float(score, "score")
    positions = transformed.discrete_positions.to(torch.float64)
    return StretchCandidateResult(
        stretch=stretch_value,
        score=score_value,
        valid=embedding_valid,
        embedding_range_valid=embedding_valid,
        transformed_min=float(positions.min().item()),
        transformed_max=float(positions.max().item()),
        source_overlap_span=_scalar_mean(
            statistics.source_overlap_span, "source_overlap_span"
        ),
        source_overlap_ratio=_scalar_mean(
            statistics.source_overlap_ratio, "source_overlap_ratio"
        ),
        in_source_point_ratio=_scalar_mean(
            statistics.in_source_point_ratio, "in_source_point_ratio"
        ),
        duplicate_count=int(statistics.duplicate_count.to(torch.long).sum().item()),
        duplicate_ratio=_scalar_mean(statistics.duplicate_ratio, "duplicate_ratio"),
    )


def _selection_key(candidate: StretchCandidateResult) -> Tuple[float, int, float, float]:
    if candidate.score is None:
        raise ValueError("valid candidates must have finite scores")
    return (
        -candidate.score,
        0 if candidate.stretch == 1.0 else 1,
        abs(candidate.stretch - 1.0),
        candidate.stretch,
    )


def build_estimate_result(
    candidates: Iterable[StretchCandidateResult],
    *,
    global_shift: int,
    anchor: float,
) -> StretchEstimateResult:
    """Select a valid candidate using deterministic, order-independent rules."""

    rows = tuple(sorted(tuple(candidates), key=lambda row: row.stretch))
    if not rows:
        raise ValueError("at least one stretch candidate is required")
    if len({row.stretch for row in rows}) != len(rows):
        raise ValueError("stretch candidates must be unique")
    valid_rows = sorted((row for row in rows if row.valid), key=_selection_key)
    if not valid_rows:
        raise ValueError("all stretch candidates are invalid")
    selected = valid_rows[0]
    second = valid_rows[1] if len(valid_rows) > 1 else None
    return StretchEstimateResult(
        global_shift=int(global_shift),
        anchor=_finite_float(anchor, "anchor"),
        selected_stretch=selected.stretch,
        selected_score=selected.score,
        second_best_stretch=None if second is None else second.stretch,
        second_best_score=None if second is None else second.score,
        top1_top2_margin=(
            None
            if second is None
            else float(selected.score) - float(second.score)
        ),
        selected_at_boundary=selected.stretch in {
            min(STRETCH_CANDIDATES),
            max(STRETCH_CANDIDATES),
        },
        candidates=rows,
    )


def candidate_json_records(result: StretchEstimateResult) -> List[Dict[str, Any]]:
    """Return JSON-safe candidates sorted by ascending stretch."""

    return [asdict(row) for row in sorted(result.candidates, key=lambda row: row.stretch)]


def candidate_tsv_records(
    result: StretchEstimateResult,
    *,
    task: str,
) -> List[Dict[str, Any]]:
    """Return stable TSV rows while retaining score rank separately."""

    ranked = sorted((row for row in result.candidates if row.valid), key=_selection_key)
    rank_by_stretch = {row.stretch: rank for rank, row in enumerate(ranked, start=1)}
    records = []
    for row in sorted(result.candidates, key=lambda item: item.stretch):
        records.append(
            {
                "task": task,
                "global_shift": result.global_shift,
                "anchor": result.anchor,
                **asdict(row),
                "rank": rank_by_stretch.get(row.stretch),
                "selected": row.stretch == result.selected_stretch,
                "at_boundary": row.stretch in {
                    min(STRETCH_CANDIDATES),
                    max(STRETCH_CANDIDATES),
                },
            }
        )
    return records


def validate_temporal_audit(
    document: Mapping[str, Any],
    *,
    source: str,
    target: str,
) -> Dict[str, Any]:
    """Validate and extract immutable inputs from a formal Stage 2 audit."""

    if document.get("schema_version") != TEMPORAL_AUDIT_SCHEMA:
        raise ValueError(
            f"temporal audit schema must be {TEMPORAL_AUDIT_SCHEMA!r}"
        )
    audit_source = document.get("source", {}).get("domain")
    audit_target = document.get("target", {}).get("domain")
    if (audit_source, audit_target) != (source, target):
        raise ValueError(
            "temporal audit task does not match CLI task: "
            f"audit={audit_source}->{audit_target}, cli={source}->{target}"
        )
    repository_commit = str(document.get("repository", {}).get("commit", "")).strip()
    checkpoint_sha256 = str(document.get("checkpoint", {}).get("sha256", "")).strip()
    if not repository_commit:
        raise ValueError("temporal audit repository commit is missing")
    if len(checkpoint_sha256) != 64:
        raise ValueError("temporal audit checkpoint SHA256 is missing or invalid")

    source_row = document["source"]
    target_row = document["target"]
    embedding = document["embedding"]
    comparison = document["comparison"]
    return {
        "source_range": (
            _finite_float(source_row["position_min"], "source position_min"),
            _finite_float(source_row["position_max"], "source position_max"),
        ),
        "target_range": (
            _finite_float(target_row["position_min"], "target position_min"),
            _finite_float(target_row["position_max"], "target position_max"),
        ),
        "embedding_legal_range": (
            int(embedding["legal_raw_min"]),
            int(embedding["legal_raw_max"]),
        ),
        "anchor": _finite_float(comparison["source_midpoint"], "source midpoint"),
        "checkpoint_path": str(document["checkpoint"]["path"]),
        "checkpoint_sha256": checkpoint_sha256,
        "audit_repository_branch": str(document["repository"]["branch"]),
        "audit_repository_commit": repository_commit,
        "audit_schema_version": str(document["schema_version"]),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_sha256(path: Path, expected_sha256: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    actual = file_sha256(resolved)
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError(
            "checkpoint SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual}, path={resolved}"
        )
    return actual


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the raw bytes of a detached CPU-contiguous tensor."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    value = tensor.detach().cpu().contiguous()
    raw_bytes = bytes(value.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(raw_bytes).hexdigest()


def repeated_score_diagnostics(value: Any, score_fn) -> Dict[str, float]:
    """Call one score function twice with the exact same object."""

    first = _finite_float(score_fn(value), "first repeated score")
    second = _finite_float(score_fn(value), "second repeated score")
    return {
        "first_score": first,
        "second_score": second,
        "same_softmax_score_repeat_abs_diff": abs(first - second),
    }


def _identity_failed_checks(
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    same_softmax_score_repeat_abs_diff: float,
    baseline_affine_per_batch_score_abs_diff: float,
    identity_atol: float,
) -> List[str]:
    failed = []
    for comparison_name in (
        "baseline1_vs_baseline2",
        "affine1_vs_affine2",
        "baseline1_vs_affine1",
    ):
        values = comparisons[comparison_name]
        for boolean_name in ("positions_equal", "argmax_equal"):
            if boolean_name in values and values[boolean_name] is not True:
                failed.append(f"{comparison_name}.{boolean_name}")
        for numeric_name in (
            "logits_max_abs_diff",
            "softmax_max_abs_diff",
            "score_abs_diff",
        ):
            if numeric_name in values:
                difference = _finite_float(
                    values[numeric_name], f"{comparison_name}.{numeric_name}"
                )
                if difference > identity_atol:
                    failed.append(f"{comparison_name}.{numeric_name}")
    if same_softmax_score_repeat_abs_diff > identity_atol:
        failed.append("same_softmax_score_repeat_abs_diff")
    if baseline_affine_per_batch_score_abs_diff > identity_atol:
        failed.append("baseline_affine_per_batch_score_abs_diff")
    return failed


def build_identity_debug_report(
    *,
    shift: Mapping[str, Any],
    anchor: float,
    positions: Mapping[str, Any],
    execution: Mapping[str, Any],
    comparisons: Mapping[str, Mapping[str, Any]],
    same_softmax_score_repeat_abs_diff: float,
    baseline_affine_per_batch_score_abs_diff: float,
    global_selected_score_abs_diff: float,
    identity_atol: float,
) -> Dict[str, Any]:
    """Build a JSON-safe, fail-closed identity diagnostic document."""

    tolerance = _finite_float(identity_atol, "identity_atol")
    if tolerance < 0:
        raise ValueError("identity_atol must be nonnegative")
    repeat_difference = _finite_float(
        same_softmax_score_repeat_abs_diff,
        "same_softmax_score_repeat_abs_diff",
    )
    per_batch_difference = _finite_float(
        baseline_affine_per_batch_score_abs_diff,
        "baseline_affine_per_batch_score_abs_diff",
    )
    global_difference = _finite_float(
        global_selected_score_abs_diff,
        "global_selected_score_abs_diff",
    )
    comparison_copy = {
        name: dict(comparisons[name])
        for name in (
            "baseline1_vs_baseline2",
            "affine1_vs_affine2",
            "baseline1_vs_affine1",
        )
    }
    failed_checks = _identity_failed_checks(
        comparison_copy,
        same_softmax_score_repeat_abs_diff=repeat_difference,
        baseline_affine_per_batch_score_abs_diff=per_batch_difference,
        identity_atol=tolerance,
    )
    return {
        "shift": dict(shift),
        "anchor": _finite_float(anchor, "anchor"),
        "positions": dict(positions),
        "execution": dict(execution),
        "comparisons": comparison_copy,
        "same_softmax_score_repeat_abs_diff": repeat_difference,
        "baseline_affine_per_batch_score_abs_diff": per_batch_difference,
        "global_selected_score_abs_diff": global_difference,
        "identity_atol": tolerance,
        "failed_checks": failed_checks,
        "passed": not failed_checks,
    }


def build_formal_identity_audit(
    *,
    positions_equal: bool,
    logits_max_abs_diff: float,
    softmax_max_abs_diff: float,
    argmax_equal: bool,
    score_abs_diff: float,
    packed_a1_softmax_max_abs_diff: float,
    packed_a1_score_abs_diff: float,
    global_selected_score_abs_diff: float,
    identity_atol: float,
) -> Dict[str, Any]:
    """Build the formal gate from independent baseline and a=1 forwards only."""

    tolerance = _finite_float(identity_atol, "identity_atol")
    if tolerance < 0:
        raise ValueError("identity_atol must be nonnegative")
    audit = {
        "positions_equal": bool(positions_equal),
        "logits_max_abs_diff": _finite_float(
            logits_max_abs_diff, "logits_max_abs_diff"
        ),
        "softmax_max_abs_diff": _finite_float(
            softmax_max_abs_diff, "softmax_max_abs_diff"
        ),
        "argmax_equal": bool(argmax_equal),
        "score_abs_diff": _finite_float(score_abs_diff, "score_abs_diff"),
        "packed_a1_softmax_max_abs_diff": _finite_float(
            packed_a1_softmax_max_abs_diff,
            "packed_a1_softmax_max_abs_diff",
        ),
        "packed_a1_score_abs_diff": _finite_float(
            packed_a1_score_abs_diff, "packed_a1_score_abs_diff"
        ),
        "global_selected_score_abs_diff": _finite_float(
            global_selected_score_abs_diff, "global_selected_score_abs_diff"
        ),
        "identity_atol": tolerance,
    }
    audit["passed"] = bool(
        audit["positions_equal"]
        and audit["argmax_equal"]
        and audit["logits_max_abs_diff"] <= tolerance
        and audit["softmax_max_abs_diff"] <= tolerance
        and audit["score_abs_diff"] <= tolerance
    )
    return audit


def finalize_identity_debug_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Persist an identity report before applying the existing fail-closed gate."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    require_identity_audit_passed(report)


def require_identity_audit_passed(identity_audit: Mapping[str, Any]) -> None:
    if identity_audit.get("passed") is not True:
        raise ValueError("identity audit failed; refusing formal stretch output")


__all__ = [
    "STRETCH_CANDIDATES",
    "STRETCH_ESTIMATE_SCHEMA",
    "StretchCandidateResult",
    "StretchEstimateResult",
    "build_formal_identity_audit",
    "build_identity_debug_report",
    "build_candidate_result",
    "build_estimate_result",
    "candidate_json_records",
    "candidate_tsv_records",
    "create_target_train_indices",
    "file_sha256",
    "finalize_identity_debug_report",
    "fixed_stretch_candidates",
    "require_identity_audit_passed",
    "repeated_score_diagnostics",
    "tensor_sha256",
    "validate_checkpoint_sha256",
    "validate_temporal_audit",
]
