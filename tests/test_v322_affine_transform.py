import pytest
import torch

from methods.temporal_alignment.affine_transform import transform_positions


def test_identity_fast_path_matches_integer_shift_exactly():
    positions = torch.tensor([[0, 3, 9], [2, 5, 8]], dtype=torch.long)

    result = transform_positions(positions, shift=4, stretch=1.0, anchor=5.5)

    assert result.discrete_positions.dtype == torch.long
    assert torch.equal(result.discrete_positions, positions + 4)
    assert torch.equal(result.continuous_positions, (positions + 4).to(torch.float32))


@pytest.mark.parametrize("stretch", [0.5, 1.5])
def test_positive_stretch_preserves_continuous_order(stretch):
    positions = torch.tensor([[0, 2, 7, 11]], dtype=torch.long)

    result = transform_positions(positions, shift=0, stretch=stretch, anchor=5)

    assert result.continuous_monotonic.tolist() == [True]
    assert result.discrete_nondecreasing.tolist() == [True]


def test_rounding_duplicates_are_allowed_and_counted():
    positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    result = transform_positions(positions, shift=0, stretch=0.4, anchor=0)

    assert result.discrete_positions.tolist() == [[0, 0, 1, 1]]
    assert result.duplicate_count.tolist() == [2]
    assert result.duplicate_ratio.tolist() == pytest.approx([2 / 3])


@pytest.mark.parametrize("stretch", [0.0, -0.5])
def test_nonpositive_stretch_is_rejected(stretch):
    positions = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="stretch"):
        transform_positions(positions, shift=0, stretch=stretch, anchor=0)


def test_batch_parameters_are_broadcast_per_sample():
    positions = torch.tensor([[0, 2, 4], [10, 12, 14]], dtype=torch.long)

    result = transform_positions(
        positions,
        shift=torch.tensor([1, -2]),
        stretch=torch.tensor([1.0, 0.5]),
        anchor=torch.tensor([0.0, 10.0]),
    )

    assert result.discrete_positions.tolist() == [[1, 3, 5], [8, 9, 10]]


def test_valid_mask_excludes_invalid_positions_from_statistics():
    positions = torch.tensor([[0, 1, 2, 100]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, True, False]])

    result = transform_positions(
        positions,
        shift=0,
        stretch=0.4,
        anchor=0,
        valid_mask=valid_mask,
    )

    assert result.duplicate_count.tolist() == [1]
    assert result.duplicate_ratio.tolist() == pytest.approx([0.5])
    assert result.continuous_monotonic.tolist() == [True]


def test_invalid_positions_keep_their_original_values():
    positions = torch.tensor([[0, 1, 2, 100]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, False, False]])

    result = transform_positions(
        positions,
        shift=5,
        stretch=0.5,
        anchor=0,
        valid_mask=valid_mask,
    )

    assert result.continuous_positions.tolist() == [[5.0, 5.5, 2.0, 100.0]]
    assert result.discrete_positions.tolist() == [[5, 6, 2, 100]]
    assert result.duplicate_count.tolist() == [0]


def test_single_valid_position_has_zero_duplicate_ratio():
    positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    valid_mask = torch.tensor([[False, True, False]])

    result = transform_positions(positions, shift=0, stretch=1, anchor=0, valid_mask=valid_mask)

    assert result.duplicate_count.tolist() == [0]
    assert result.duplicate_ratio.tolist() == [0.0]
    assert result.continuous_monotonic.tolist() == [True]
    assert result.discrete_nondecreasing.tolist() == [True]


def test_input_tensor_is_not_modified():
    positions = torch.tensor([[1, 4, 9]], dtype=torch.long)
    original = positions.clone()

    transform_positions(positions, shift=-3, stretch=1.2, anchor=4)

    assert torch.equal(positions, original)


def test_transform_does_not_clip_positions():
    positions = torch.tensor([[0, 10]], dtype=torch.long)

    result = transform_positions(positions, shift=-100, stretch=3, anchor=0)

    assert result.discrete_positions.tolist() == [[-100, -70]]


def test_round_uses_half_to_even_for_halfway_values():
    positions = torch.tensor([[10], [11]], dtype=torch.long)

    result = transform_positions(
        positions,
        shift=0,
        stretch=torch.tensor([0.5, 0.5]),
        anchor=torch.tensor([11.0, 12.0]),
    )

    assert result.continuous_positions.tolist() == [[10.5], [11.5]]
    assert result.discrete_positions.tolist() == [[10], [12]]


def test_positions_must_be_integer_tensor():
    with pytest.raises(TypeError, match="integer"):
        transform_positions(torch.tensor([[0.0, 1.0]]), shift=0, stretch=1, anchor=0)


def test_nonintegral_shift_is_rejected():
    positions = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="shift"):
        transform_positions(positions, shift=0.5, stretch=1, anchor=0)
