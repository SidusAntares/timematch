import pytest
import torch

from methods.temporal_alignment.range_gate import (
    apply_shift_only_fallback,
    compute_range_statistics,
)


def _stats(original, transformed, **kwargs):
    return compute_range_statistics(
        torch.tensor(original, dtype=torch.long),
        torch.tensor(transformed, dtype=torch.long),
        source_min=kwargs.pop("source_min", 0),
        source_max=kwargs.pop("source_max", 10),
        embedding_index_min=kwargs.pop("embedding_index_min", -5),
        embedding_index_max=kwargs.pop("embedding_index_max", 15),
        **kwargs,
    )


def test_all_points_inside_source_and_embedding_ranges():
    stats = _stats([[1, 4, 9]], [[2, 5, 8]])

    assert stats.valid_point_count.tolist() == [3]
    assert stats.in_source_range_count.tolist() == [3]
    assert stats.out_of_source_range_count.tolist() == [0]
    assert stats.in_embedding_range_count.tolist() == [3]
    assert stats.embedding_range_valid.tolist() == [True]
    assert stats.in_source_point_ratio.tolist() == [1.0]


def test_partial_source_overlap_statistics():
    stats = _stats([[0, 5, 10]], [[-5, 5, 15]])

    assert stats.in_source_range_count.tolist() == [1]
    assert stats.out_of_source_range_count.tolist() == [2]
    assert stats.source_overlap_span.tolist() == [10.0]
    assert stats.source_overlap_ratio.tolist() == pytest.approx([0.5])
    assert stats.in_source_point_ratio.tolist() == pytest.approx([1 / 3])
    assert stats.max_gap.tolist() == [10.0]


def test_no_source_overlap():
    stats = _stats([[0, 1]], [[20, 25]], embedding_index_max=30)

    assert stats.source_overlap_span.tolist() == [0.0]
    assert stats.source_overlap_ratio.tolist() == [0.0]
    assert stats.in_source_point_ratio.tolist() == [0.0]


def test_embedding_range_invalid_when_any_valid_point_is_outside():
    stats = _stats([[0, 1, 2]], [[-6, 0, 16]])

    assert stats.in_embedding_range_count.tolist() == [1]
    assert stats.out_of_embedding_range_count.tolist() == [2]
    assert stats.embedding_range_valid.tolist() == [False]


def test_original_and_transformed_spans_are_reported():
    stats = _stats([[1, 5, 11]], [[2, 6, 10]])

    assert stats.original_span.tolist() == [10.0]
    assert stats.transformed_span.tolist() == [8.0]


def test_single_valid_point_uses_stable_zero_span_definition():
    stats_inside = _stats([[3]], [[4]])
    stats_outside = _stats([[3]], [[20]], embedding_index_max=30)

    assert stats_inside.transformed_span.tolist() == [0.0]
    assert stats_inside.source_overlap_ratio.tolist() == [1.0]
    assert stats_inside.max_gap.tolist() == [0.0]
    assert stats_outside.source_overlap_ratio.tolist() == [0.0]


def test_no_valid_points_return_neutral_counts_and_invalid_embedding_status():
    mask = torch.tensor([[False, False]])
    stats = _stats([[0, 10]], [[0, 10]], valid_mask=mask)

    assert stats.valid_point_count.tolist() == [0]
    assert stats.original_span.tolist() == [0.0]
    assert stats.source_overlap_ratio.tolist() == [0.0]
    assert stats.duplicate_ratio.tolist() == [0.0]
    assert stats.embedding_range_valid.tolist() == [False]


def test_valid_mask_excludes_invalid_values():
    mask = torch.tensor([[True, True, False]])
    stats = _stats([[0, 5, 999]], [[1, 6, 999]], valid_mask=mask)

    assert stats.valid_point_count.tolist() == [2]
    assert stats.transformed_span.tolist() == [5.0]
    assert stats.embedding_range_valid.tolist() == [True]


def test_duplicate_statistics_are_computed_from_adjacent_valid_positions():
    stats = _stats([[0, 1, 2, 3]], [[0, 0, 1, 1]])

    assert stats.duplicate_count.tolist() == [2]
    assert stats.duplicate_ratio.tolist() == pytest.approx([2 / 3])


def test_fallback_rejects_samples_with_exact_shift_only_positions():
    original = torch.tensor([[0, 2], [10, 12]], dtype=torch.long)
    affine = torch.tensor([[1, 4], [8, 9]], dtype=torch.long)

    output = apply_shift_only_fallback(
        original,
        shift=torch.tensor([3, -2]),
        use_affine=torch.tensor([False, True]),
        affine_positions=affine,
    )

    assert output.dtype == torch.long
    assert output.tolist() == [[3, 5], [8, 9]]


def test_fallback_supports_scalar_decision():
    original = torch.tensor([[0, 2], [10, 12]], dtype=torch.long)
    affine = torch.tensor([[1, 4], [8, 9]], dtype=torch.long)

    accepted = apply_shift_only_fallback(original, 2, True, affine)
    rejected = apply_shift_only_fallback(original, 2, False, affine)

    assert torch.equal(accepted, affine)
    assert torch.equal(rejected, original + 2)


def test_fallback_does_not_modify_inputs_or_clip_output():
    original = torch.tensor([[0, 2]], dtype=torch.long)
    affine = torch.tensor([[-100, 1000]], dtype=torch.long)
    original_copy = original.clone()
    affine_copy = affine.clone()

    output = apply_shift_only_fallback(original, -50, True, affine)

    assert output.tolist() == [[-100, 1000]]
    assert torch.equal(original, original_copy)
    assert torch.equal(affine, affine_copy)


def test_range_parameters_can_be_batch_wise():
    stats = compute_range_statistics(
        torch.tensor([[0, 5], [10, 15]], dtype=torch.long),
        torch.tensor([[0, 5], [10, 15]], dtype=torch.long),
        source_min=torch.tensor([0, 12]),
        source_max=torch.tensor([5, 20]),
        embedding_index_min=torch.tensor([-1, 9]),
        embedding_index_max=torch.tensor([6, 16]),
    )

    assert stats.in_source_range_count.tolist() == [2, 1]
    assert stats.embedding_range_valid.tolist() == [True, True]


@pytest.mark.parametrize(
    "transformed",
    [
        torch.tensor([[True, False]], dtype=torch.bool),
        torch.tensor([[1 + 0j, 2 + 0j]], dtype=torch.complex64),
    ],
)
def test_transformed_positions_must_be_real_numeric_tensor(transformed):
    with pytest.raises(TypeError, match="real numeric"):
        compute_range_statistics(
            torch.tensor([[0, 1]], dtype=torch.long),
            transformed,
            source_min=0,
            source_max=10,
            embedding_index_min=-5,
            embedding_index_max=15,
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_transformed_positions_must_be_finite(nonfinite):
    with pytest.raises(ValueError, match="transformed_positions.*finite"):
        compute_range_statistics(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[0.0, nonfinite]], dtype=torch.float32),
            source_min=0,
            source_max=10,
            embedding_index_min=-5,
            embedding_index_max=15,
        )


@pytest.mark.parametrize(
    "parameter_name",
    ["source_min", "source_max", "embedding_index_min", "embedding_index_max"],
)
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_range_parameters_must_be_finite(parameter_name, nonfinite):
    parameters = {
        "source_min": 0,
        "source_max": 10,
        "embedding_index_min": -5,
        "embedding_index_max": 15,
    }
    parameters[parameter_name] = nonfinite

    with pytest.raises(ValueError, match=f"{parameter_name}.*finite"):
        compute_range_statistics(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[0, 1]], dtype=torch.long),
            **parameters,
        )
