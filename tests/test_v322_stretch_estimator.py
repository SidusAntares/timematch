import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from methods.temporal_alignment.affine_transform import transform_positions
from methods.temporal_alignment.range_gate import compute_range_statistics
from methods.temporal_alignment.stretch_estimator import (
    STRETCH_CANDIDATES,
    build_formal_identity_audit,
    StretchCandidateResult,
    build_identity_debug_report,
    build_candidate_result,
    build_estimate_result,
    candidate_json_records,
    candidate_tsv_records,
    create_target_train_indices,
    finalize_identity_debug_report,
    require_identity_audit_passed,
    repeated_score_diagnostics,
    tensor_sha256,
    validate_checkpoint_sha256,
    validate_temporal_audit,
)


def candidate(stretch, score, *, valid=True):
    return StretchCandidateResult(
        stretch=float(stretch),
        score=float(score),
        valid=valid,
        embedding_range_valid=valid,
        transformed_min=10.0,
        transformed_max=20.0,
        source_overlap_span=10.0,
        source_overlap_ratio=1.0,
        in_source_point_ratio=1.0,
        duplicate_count=0,
        duplicate_ratio=0.0,
    )


def audit_document(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return checkpoint, {
        "schema_version": "v322-temporal-range-audit-v1",
        "repository": {
            "branch": "exp/v322-affine-temporal-alignment",
            "commit": "26c2fae6c8f90825222362df753c1754bad5a4ca",
        },
        "source": {
            "domain": "austria/33UVP/2017",
            "position_min": 0.0,
            "position_max": 358.0,
        },
        "target": {
            "domain": "france/31TCJ/2017",
            "position_min": 25.0,
            "position_max": 360.0,
        },
        "comparison": {"source_midpoint": 179.0},
        "embedding": {"legal_raw_min": -100, "legal_raw_max": 464},
        "checkpoint": {"path": str(checkpoint), "sha256": digest},
    }


def test_fixed_candidates():
    assert STRETCH_CANDIDATES == (
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


def test_selection_is_input_order_independent_and_uses_highest_score():
    rows = [candidate(0.8, 1.0), candidate(1.0, 3.0), candidate(1.2, 2.0)]
    forward = build_estimate_result(rows, global_shift=7, anchor=100.0)
    reverse = build_estimate_result(list(reversed(rows)), global_shift=7, anchor=100.0)
    assert forward == reverse
    assert forward.selected_stretch == 1.0
    assert forward.second_best_stretch == 1.2
    assert forward.top1_top2_margin == pytest.approx(1.0)


def test_tie_prefers_one_then_distance_then_smaller_stretch():
    with_one = build_estimate_result(
        [candidate(0.95, 2.0), candidate(1.0, 2.0), candidate(1.05, 2.0)],
        global_shift=0,
        anchor=0.0,
    )
    assert with_one.selected_stretch == 1.0

    nearest = build_estimate_result(
        [candidate(0.8, 2.0), candidate(1.05, 2.0)],
        global_shift=0,
        anchor=0.0,
    )
    assert nearest.selected_stretch == 1.05

    lower = build_estimate_result(
        [candidate(0.95, 2.0), candidate(1.05, 2.0)],
        global_shift=0,
        anchor=0.0,
    )
    assert lower.selected_stretch == 0.95


def test_invalid_candidates_are_excluded_and_all_invalid_raises():
    result = build_estimate_result(
        [candidate(0.8, 100.0, valid=False), candidate(1.0, 1.0)],
        global_shift=0,
        anchor=0.0,
    )
    assert result.selected_stretch == 1.0
    with pytest.raises(ValueError, match="all stretch candidates are invalid"):
        build_estimate_result(
            [candidate(0.8, 1.0, valid=False), candidate(1.2, 2.0, valid=False)],
            global_shift=0,
            anchor=0.0,
        )


def test_boundary_saturation():
    low = build_estimate_result(
        [candidate(0.8, 2.0), candidate(1.0, 1.0)], global_shift=0, anchor=0.0
    )
    high = build_estimate_result(
        [candidate(1.0, 1.0), candidate(1.2, 2.0)], global_shift=0, anchor=0.0
    )
    middle = build_estimate_result(
        [candidate(0.95, 2.0), candidate(1.0, 1.0)], global_shift=0, anchor=0.0
    )
    assert low.selected_at_boundary
    assert high.selected_at_boundary
    assert not middle.selected_at_boundary


def test_build_candidate_aggregates_transform_and_range_diagnostics():
    positions = torch.tensor([[10, 20, 30]], dtype=torch.long)
    transformed = transform_positions(positions, shift=5, stretch=1.0, anchor=20.0)
    statistics = compute_range_statistics(
        positions,
        transformed.discrete_positions,
        source_min=10,
        source_max=40,
        embedding_index_min=-100,
        embedding_index_max=464,
    )
    row = build_candidate_result(1.0, 0.7, transformed, statistics)
    assert row.valid
    assert row.embedding_range_valid
    assert row.transformed_min == 15.0
    assert row.transformed_max == 35.0
    assert row.source_overlap_span == 20.0
    assert row.in_source_point_ratio == 1.0


def test_embedding_out_of_range_makes_candidate_invalid():
    positions = torch.tensor([[400, 450]], dtype=torch.long)
    transformed = transform_positions(positions, shift=100, stretch=1.0, anchor=425.0)
    statistics = compute_range_statistics(
        positions,
        transformed.discrete_positions,
        source_min=0,
        source_max=500,
        embedding_index_min=-100,
        embedding_index_max=464,
    )
    row = build_candidate_result(1.0, 0.7, transformed, statistics)
    assert not row.embedding_range_valid
    assert not row.valid


def test_invalid_candidate_can_record_no_score():
    positions = torch.tensor([[400, 450]], dtype=torch.long)
    transformed = transform_positions(positions, shift=100, stretch=1.0, anchor=425.0)
    statistics = compute_range_statistics(
        positions,
        transformed.discrete_positions,
        source_min=0,
        source_max=500,
        embedding_index_min=-100,
        embedding_index_max=464,
    )
    row = build_candidate_result(1.0, None, transformed, statistics)
    assert row.score is None
    assert not row.valid


def test_stretch_one_positions_are_exact_identity_shift():
    positions = torch.tensor([[10, 11, 12]], dtype=torch.long)
    result = transform_positions(positions, shift=7, stretch=1.0, anchor=10.5)
    assert torch.equal(result.discrete_positions, positions + 7)


def test_temporal_audit_validation_and_input_immutability(tmp_path):
    _, document = audit_document(tmp_path)
    before = copy.deepcopy(document)
    parsed = validate_temporal_audit(
        document,
        source="austria/33UVP/2017",
        target="france/31TCJ/2017",
    )
    assert parsed["anchor"] == 179.0
    assert parsed["embedding_legal_range"] == (-100, 464)
    assert document == before


def test_temporal_audit_rejects_schema_and_task_mismatch(tmp_path):
    _, document = audit_document(tmp_path)
    bad_schema = copy.deepcopy(document)
    bad_schema["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        validate_temporal_audit(
            bad_schema,
            source="austria/33UVP/2017",
            target="france/31TCJ/2017",
        )
    with pytest.raises(ValueError, match="task does not match"):
        validate_temporal_audit(
            document,
            source="france/31TCJ/2017",
            target="austria/33UVP/2017",
        )


def test_checkpoint_sha256_validation(tmp_path):
    checkpoint, document = audit_document(tmp_path)
    validate_checkpoint_sha256(checkpoint, document["checkpoint"]["sha256"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        validate_checkpoint_sha256(checkpoint, document["checkpoint"]["sha256"])


def test_json_and_tsv_records_are_stably_sorted_without_mutation():
    rows = [candidate(1.2, 1.0), candidate(0.8, 3.0), candidate(1.0, 2.0)]
    before = copy.deepcopy(rows)
    result = build_estimate_result(rows, global_shift=4, anchor=100.0)
    json_rows = candidate_json_records(result)
    tsv_rows = candidate_tsv_records(result, task="AT1_to_FR2")
    assert [row["stretch"] for row in json_rows] == [0.8, 1.0, 1.2]
    assert [row["stretch"] for row in tsv_rows] == [0.8, 1.0, 1.2]
    assert [row["rank"] for row in tsv_rows] == [1, 2, 3]
    assert tsv_rows[0]["selected"] is True
    assert rows == before
    json.dumps(json_rows)


def test_identity_gate_rejects_failed_audit():
    require_identity_audit_passed({"passed": True})
    with pytest.raises(ValueError, match="identity audit failed"):
        require_identity_audit_passed({"passed": False})


def test_cli_source_does_not_read_target_labels():
    source = (
        Path(__file__).resolve().parents[1] / "tools" / "estimate_v322_stretch.py"
    ).read_text(encoding="utf-8")
    assert 'sample["label"]' not in source
    assert "sample['label']" not in source
    assert "from train import" not in source


def test_target_train_split_is_deterministic_and_consumes_source_shuffle_first():
    import random

    random.seed(1)
    first = create_target_train_indices(10, 12, val_ratio=0.1, test_ratio=0.2)
    random.seed(1)
    second = create_target_train_indices(10, 12, val_ratio=0.1, test_ratio=0.2)
    assert first == second
    assert len(first) == 9


def _identity_comparison(**overrides):
    values = {
        "positions_equal": True,
        "logits_max_abs_diff": 0.0,
        "softmax_max_abs_diff": 0.0,
        "argmax_equal": True,
        "score_abs_diff": 0.0,
    }
    values.update(overrides)
    return values


def test_identity_debug_failure_is_written_before_gate(tmp_path):
    report = build_identity_debug_report(
        shift={"value": 3, "dtype": "int", "shape": []},
        anchor=179.0,
        positions={"baseline_affine_equal": True},
        execution={"spatial_features_reused": True},
        comparisons={
            "baseline1_vs_baseline2": _identity_comparison(),
            "affine1_vs_affine2": _identity_comparison(),
            "baseline1_vs_affine1": _identity_comparison(
                logits_max_abs_diff=2e-7
            ),
        },
        same_softmax_score_repeat_abs_diff=0.0,
        baseline_affine_per_batch_score_abs_diff=0.0,
        global_selected_score_abs_diff=0.0,
        identity_atol=1e-7,
    )
    output = tmp_path / "identity_debug.json"

    with pytest.raises(ValueError, match="identity audit failed"):
        finalize_identity_debug_report(report, output)

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["passed"] is False
    assert stored["failed_checks"] == [
        "baseline1_vs_affine1.logits_max_abs_diff"
    ]


def test_identity_debug_keeps_repeat_and_cross_path_comparisons_separate():
    report = build_identity_debug_report(
        shift={"value": 0, "dtype": "int", "shape": []},
        anchor=0.0,
        positions={"baseline_affine_equal": True},
        execution={"spatial_features_reused": True},
        comparisons={
            "baseline1_vs_baseline2": _identity_comparison(
                logits_max_abs_diff=2e-7
            ),
            "affine1_vs_affine2": _identity_comparison(
                softmax_max_abs_diff=3e-7
            ),
            "baseline1_vs_affine1": _identity_comparison(argmax_equal=False),
        },
        same_softmax_score_repeat_abs_diff=4e-7,
        baseline_affine_per_batch_score_abs_diff=0.0,
        global_selected_score_abs_diff=0.0,
        identity_atol=1e-7,
    )

    assert report["failed_checks"] == [
        "baseline1_vs_baseline2.logits_max_abs_diff",
        "affine1_vs_affine2.softmax_max_abs_diff",
        "baseline1_vs_affine1.argmax_equal",
        "same_softmax_score_repeat_abs_diff",
    ]
    assert set(report["comparisons"]) == {
        "baseline1_vs_baseline2",
        "affine1_vs_affine2",
        "baseline1_vs_affine1",
    }


def test_repeated_score_diagnostics_reuses_the_same_softmax_object():
    softmax = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    seen = []

    def score_fn(value):
        seen.append(value)
        return float(value[:, 1].mean().item())

    result = repeated_score_diagnostics(softmax, score_fn)
    assert result["same_softmax_score_repeat_abs_diff"] == 0.0
    assert seen[0] is softmax
    assert seen[1] is softmax


def test_tensor_sha256_uses_cpu_contiguous_bytes():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert tensor_sha256(tensor) == tensor_sha256(tensor.t().contiguous().t())
    assert tensor_sha256(tensor) != tensor_sha256(tensor + 1.0)


def test_identity_debug_pass_does_not_write_formal_outputs(tmp_path):
    report = build_identity_debug_report(
        shift={"value": 0, "dtype": "int", "shape": []},
        anchor=0.0,
        positions={"baseline_affine_equal": True},
        execution={"spatial_features_reused": True},
        comparisons={
            "baseline1_vs_baseline2": _identity_comparison(),
            "affine1_vs_affine2": _identity_comparison(),
            "baseline1_vs_affine1": _identity_comparison(),
        },
        same_softmax_score_repeat_abs_diff=0.0,
        baseline_affine_per_batch_score_abs_diff=0.0,
        global_selected_score_abs_diff=0.0,
        identity_atol=1e-7,
    )
    debug_output = tmp_path / "identity_debug.json"
    finalize_identity_debug_report(report, debug_output)

    assert debug_output.is_file()
    assert not (tmp_path / "stretch.json").exists()
    assert not (tmp_path / "stretch.tsv").exists()


def test_formal_identity_gate_uses_independent_a1_not_packed_differences():
    audit = build_formal_identity_audit(
        positions_equal=True,
        logits_max_abs_diff=0.0,
        softmax_max_abs_diff=0.0,
        argmax_equal=True,
        score_abs_diff=0.0,
        packed_a1_softmax_max_abs_diff=2e-7,
        packed_a1_score_abs_diff=3e-7,
        global_selected_score_abs_diff=4e-7,
        identity_atol=1e-7,
    )

    assert audit["passed"] is True
    assert audit["packed_a1_softmax_max_abs_diff"] == pytest.approx(2e-7)
    assert audit["packed_a1_score_abs_diff"] == pytest.approx(3e-7)
    assert audit["global_selected_score_abs_diff"] == pytest.approx(4e-7)


def test_formal_identity_gate_rejects_independent_score_difference():
    audit = build_formal_identity_audit(
        positions_equal=True,
        logits_max_abs_diff=0.0,
        softmax_max_abs_diff=0.0,
        argmax_equal=True,
        score_abs_diff=2e-7,
        packed_a1_softmax_max_abs_diff=0.0,
        packed_a1_score_abs_diff=0.0,
        global_selected_score_abs_diff=0.0,
        identity_atol=1e-7,
    )

    assert audit["passed"] is False
