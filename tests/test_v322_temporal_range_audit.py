import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from methods.temporal_alignment.temporal_range_audit import (
    compare_temporal_summaries,
    derive_embedding_range,
    load_domain_temporal_data,
    positions_from_dates,
    summarize_temporal_positions,
)
from tools.audit_v322_temporal_ranges import main


class LabelAccessForbidden(dict):
    def __getitem__(self, key):
        if key == "label":
            raise AssertionError("target labels must not be accessed")
        return super().__getitem__(key)


def _write_metadata(root: Path, dataset_name: str, start_date, dates, sample_count: int):
    metadata_dir = root / dataset_name / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "start_date": start_date,
        "dates": dates,
        "parcels": [LabelAccessForbidden(label=index) for index in range(sample_count)],
    }
    with (metadata_dir / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)


def _summary(domain, start_date, sequences, legal_min=-100, legal_max=464):
    return summarize_temporal_positions(
        sequences,
        domain=domain,
        split="all",
        start_date=start_date,
        embedding_legal_raw_min=legal_min,
        embedding_legal_raw_max=legal_max,
    )


def _minimal_cli_args(tmp_path: Path, output_name: str = "audit"):
    data_root = tmp_path / "data"
    source = "austria/33UVP/2017"
    target = "france/31TCJ/2017"
    _write_metadata(data_root, source, 20170101, [20170101, 20170111], 2)
    _write_metadata(data_root, target, 20170101, [20170101, 20170106], 1)
    checkpoint = tmp_path / f"{output_name}.pt"
    checkpoint.write_bytes(b"fixed-checkpoint")
    output_json = tmp_path / f"{output_name}.json"
    output_tsv = tmp_path / f"{output_name}.tsv"
    return [
        "--data-root",
        str(data_root),
        "--source",
        source,
        "--target",
        target,
        "--checkpoint",
        str(checkpoint),
        "--source-method",
        "smooth_k3",
        "--source-seed",
        "1",
        "--output-json",
        str(output_json),
        "--output-tsv",
        str(output_tsv),
    ], output_json


def test_positions_from_dates_matches_dataset_absolute_day_definition():
    positions = positions_from_dates(20170110, [20170101, 20170110, 20170120])

    assert positions.dtype == np.int64
    assert positions.tolist() == [9, 0, 10]


def test_embedding_range_is_derived_from_ltae_index_expression():
    embedding = derive_embedding_range(max_position=365, max_temporal_shift=100)

    assert embedding.table_length == 565
    assert embedding.internal_index_offset == 100
    assert embedding.legal_raw_min == -100
    assert embedding.legal_raw_max == 464


def test_summary_preserves_input_order_and_reports_required_statistics():
    sequences = [
        np.array([-2, 0, 0, 5], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0, 3, 2], dtype=np.int64),
    ]
    originals = [sequence.copy() for sequence in sequences]

    summary = _summary("source", 20170101, sequences, legal_min=-1, legal_max=4)

    assert summary.sample_count == 3
    assert summary.start_date_raw == "20170101"
    assert summary.start_date_parsed == "2017-01-01"
    assert summary.position_dtype == "int64"
    assert summary.position_min == -2
    assert summary.position_max == 5
    assert summary.negative_position_count == 1
    assert summary.total_position_count == 8
    assert summary.duplicate_position_sample_count == 1
    assert summary.non_increasing_sample_count == 2
    assert summary.sequence_length_min == 1
    assert summary.sequence_length_max == 4
    assert summary.sequence_length_mean == pytest.approx(8 / 3)
    assert summary.temporal_span_min == 0
    assert summary.temporal_span_max == 7
    assert summary.temporal_span_mean == pytest.approx(10 / 3)
    assert summary.gap_min == -1
    assert summary.gap_max == 5
    assert summary.gap_mean == pytest.approx(1.8)
    assert summary.embedding_out_of_range_point_count == 2
    assert summary.embedding_out_of_range_sample_count == 1
    for sequence, original in zip(sequences, originals):
        assert np.array_equal(sequence, original)


def test_single_point_does_not_contribute_to_gap_distribution():
    summary = _summary("source", 20170101, [np.array([4], dtype=np.int64)])

    assert summary.temporal_span_min == 0
    assert summary.temporal_span_max == 0
    assert summary.gap_min is None
    assert summary.gap_max is None
    assert summary.gap_mean is None


def test_empty_split_is_rejected():
    with pytest.raises(ValueError, match="empty split"):
        _summary("source", 20170101, [])


def test_comparison_reports_equal_and_different_start_dates():
    source = _summary("source", 20170101, [np.array([0, 10])])
    same_target = _summary("target", 20170101, [np.array([5, 15])])
    later_target = _summary("target", 20170103, [np.array([5, 15])])

    same = compare_temporal_summaries(source, same_target)
    different = compare_temporal_summaries(source, later_target)

    assert same.same_start_date is True
    assert same.start_date_difference_days == 0
    assert same.raw_range_overlap_span == 5
    assert same.raw_range_overlap_ratio_source == pytest.approx(0.5)
    assert same.raw_range_overlap_ratio_target == pytest.approx(0.5)
    assert same.union_span == 15
    assert same.source_midpoint == 5
    assert same.target_midpoint == 10
    assert different.same_start_date is False
    assert different.start_date_difference_days == 2


def test_comparison_zero_span_definition_is_stable():
    same_source = _summary("source", 20170101, [np.array([5])])
    same_target = _summary("target", 20170101, [np.array([5])])
    other_target = _summary("target", 20170101, [np.array([6])])

    same = compare_temporal_summaries(same_source, same_target)
    separate = compare_temporal_summaries(same_source, other_target)

    assert same.raw_range_overlap_span == 0
    assert same.raw_range_overlap_ratio_source == 1
    assert same.raw_range_overlap_ratio_target == 1
    assert separate.raw_range_overlap_ratio_source == 0
    assert separate.raw_range_overlap_ratio_target == 0


def test_metadata_loader_uses_dates_without_accessing_labels(tmp_path):
    dataset_name = "austria/33UVP/2017"
    _write_metadata(tmp_path, dataset_name, 20170101, [20170101, 20170111], 3)

    data = load_domain_temporal_data(tmp_path, dataset_name, split="all")

    assert data.sample_count == 3
    assert data.start_date_raw == "20170101"
    assert data.position_sequence.tolist() == [0, 10]


def test_non_all_split_requires_explicit_indices(tmp_path):
    dataset_name = "france/31TCJ/2017"
    _write_metadata(tmp_path, dataset_name, 20170101, [20170101], 2)

    with pytest.raises(ValueError, match="indices"):
        load_domain_temporal_data(tmp_path, dataset_name, split="train")


def test_cli_writes_json_tsv_and_checkpoint_sha256_without_target_labels(tmp_path):
    source = "austria/33UVP/2017"
    target = "france/31TCJ/2017"
    _write_metadata(tmp_path, source, 20170101, [20170101, 20170111], 2)
    _write_metadata(tmp_path, target, 20170103, [20170103, 20170108, 20170118], 1)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixed-checkpoint")
    output_json = tmp_path / "audit.json"
    output_tsv = tmp_path / "audit.tsv"

    result = main(
        [
            "--data-root",
            str(tmp_path),
            "--source",
            source,
            "--target",
            target,
            "--source-split",
            "all",
            "--target-split",
            "all",
            "--checkpoint",
            str(checkpoint),
            "--source-method",
            "smooth_k3",
            "--source-seed",
            "1",
            "--output-json",
            str(output_json),
            "--output-tsv",
            str(output_tsv),
        ]
    )

    assert result == 0
    document = json.loads(output_json.read_text(encoding="utf-8"))
    assert document["schema_version"] == "v322-temporal-range-audit-v1"
    assert document["dataset"]["position_unit"] == "days"
    assert document["embedding"] == {
        "table_length": 565,
        "internal_index_offset": 100,
        "legal_raw_min": -100,
        "legal_raw_max": 464,
    }
    assert document["source"]["sample_count"] == 2
    assert document["target"]["sample_count"] == 1
    assert document["comparison"]["start_date_difference_days"] == 2
    assert document["checkpoint"]["sha256"] == hashlib.sha256(
        b"fixed-checkpoint"
    ).hexdigest()
    assert document["checkpoint"]["source_method"] == "smooth_k3"
    assert document["checkpoint"]["source_domain"] == source
    assert document["checkpoint"]["source_seed"] == 1
    assert "label" not in json.dumps(document).lower()

    with output_tsv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["domain_role"] for row in rows] == ["source", "target"]
    assert [row["domain"] for row in rows] == [source, target]
    assert rows[0]["position_min"] == "0"
    assert rows[1]["gap_mean"] == "7.500000"


def test_explicit_repository_identity_overrides_git_detection(tmp_path):
    arguments, output_json = _minimal_cli_args(tmp_path, "explicit_override")
    arguments.extend(
        [
            "--repository-root",
            str(Path(__file__).resolve().parents[1]),
            "--repository-branch",
            "explicit/branch",
            "--repository-commit",
            "abcdef1",
        ]
    )

    assert main(arguments) == 0

    repository = json.loads(output_json.read_text(encoding="utf-8"))["repository"]
    assert repository == {"branch": "explicit/branch", "commit": "abcdef1"}


def test_explicit_repository_identity_succeeds_without_git_metadata(tmp_path):
    arguments, output_json = _minimal_cli_args(tmp_path, "explicit_no_git")
    non_repository = tmp_path / "not-a-git-repository"
    non_repository.mkdir()
    arguments.extend(
        [
            "--repository-root",
            str(non_repository),
            "--repository-branch",
            "exp/v322-test",
            "--repository-commit",
            "1234567abcdef",
        ]
    )

    assert main(arguments) == 0

    repository = json.loads(output_json.read_text(encoding="utf-8"))["repository"]
    assert repository["branch"] == "exp/v322-test"
    assert repository["commit"] == "1234567abcdef"


def test_missing_repository_commit_fails_without_git_metadata(tmp_path):
    arguments, _ = _minimal_cli_args(tmp_path, "missing_commit")
    non_repository = tmp_path / "not-a-git-repository"
    non_repository.mkdir()
    arguments.extend(
        [
            "--repository-root",
            str(non_repository),
            "--repository-branch",
            "exp/v322-test",
        ]
    )

    with pytest.raises(ValueError, match="repository commit.*provide --repository-commit"):
        main(arguments)


@pytest.mark.parametrize("invalid_commit", ["", "abc123", "not-a-hash", "a" * 41])
def test_explicit_repository_commit_must_be_a_valid_abbreviated_hash(
    tmp_path, invalid_commit
):
    arguments, _ = _minimal_cli_args(tmp_path, f"invalid_{len(invalid_commit)}")
    arguments.extend(
        [
            "--repository-branch",
            "exp/v322-test",
            "--repository-commit",
            invalid_commit,
        ]
    )

    with pytest.raises(ValueError, match="7 to 40 hexadecimal"):
        main(arguments)
