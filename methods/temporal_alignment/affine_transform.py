"""Pure affine transformations for discrete temporal positions."""

from dataclasses import dataclass
from typing import Optional, Union

import torch


NumberOrTensor = Union[int, float, torch.Tensor]
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class AffineTransformResult:
    """Result tensors for a batch of affine temporal transformations.

    Position tensors have shape ``[B, T]``. Counts, ratios, and monotonicity
    flags have shape ``[B]``. Positions use the caller's temporal unit,
    normally days. Invalid positions retain their original values and are
    excluded from all diagnostics when ``valid_mask`` is supplied.
    """

    continuous_positions: torch.Tensor
    discrete_positions: torch.Tensor
    duplicate_count: torch.Tensor
    duplicate_ratio: torch.Tensor
    continuous_monotonic: torch.Tensor
    discrete_nondecreasing: torch.Tensor


def _validate_positions(positions: torch.Tensor) -> None:
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim != 2:
        raise ValueError("positions must have shape [B, T]")
    if positions.dtype not in _INTEGER_DTYPES:
        raise TypeError("positions must be an integer tensor")


def _batch_parameter(
    value: NumberOrTensor,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.expand(batch_size)
    if tensor.ndim == 1 and tensor.shape[0] == batch_size:
        return tensor
    raise ValueError(f"{name} must be a scalar or have shape [B]")


def _valid_mask(
    valid_mask: Optional[torch.Tensor],
    positions: torch.Tensor,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(positions, dtype=torch.bool)
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != positions.shape:
        raise ValueError("valid_mask must have shape [B, T]")
    return valid_mask.to(device=positions.device, dtype=torch.bool)


def transform_positions(
    positions: torch.Tensor,
    shift: NumberOrTensor,
    stretch: NumberOrTensor,
    anchor: NumberOrTensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> AffineTransformResult:
    """Apply a batch-wise affine transform to integer temporal positions.

    Args:
        positions: Integer tensor ``[B, T]`` in the dataset's temporal unit.
        shift: Integral scalar or tensor ``[B]`` in the same temporal unit.
        stretch: Positive scalar or tensor ``[B]``.
        anchor: Scalar or tensor ``[B]`` in the same temporal unit.
        valid_mask: Optional boolean tensor ``[B, T]``. Masked positions keep
            their original values and are excluded from diagnostics.

    Returns:
        Continuous affine positions and rounded ``torch.long`` positions,
        together with per-sample adjacency diagnostics.

    Raises:
        ValueError: If a stretch is nonpositive or a shift is nonintegral.

    No clipping or sorting is performed. ``torch.round`` uses round-half-to-even
    for halfway values. Samples with ``stretch == 1`` use an explicit integer
    identity path: ``discrete_positions = positions + shift``.
    """

    _validate_positions(positions)
    batch_size = positions.shape[0]
    mask = _valid_mask(valid_mask, positions)
    shift_values = _batch_parameter(
        shift, name="shift", batch_size=batch_size, device=positions.device
    )
    stretch_values = _batch_parameter(
        stretch, name="stretch", batch_size=batch_size, device=positions.device
    )
    anchor_values = _batch_parameter(
        anchor, name="anchor", batch_size=batch_size, device=positions.device
    )

    if not torch.isfinite(stretch_values).all() or (stretch_values <= 0).any():
        raise ValueError("stretch must contain finite positive values")
    if not torch.isfinite(shift_values).all() or not torch.equal(
        shift_values, torch.round(shift_values)
    ):
        raise ValueError("shift must contain finite integral values")
    if not torch.isfinite(anchor_values).all():
        raise ValueError("anchor must contain finite values")

    positions_float = positions.to(torch.float32)
    shift_column = shift_values[:, None]
    stretch_column = stretch_values[:, None]
    anchor_column = anchor_values[:, None]
    identity_mask = stretch_values == 1

    continuous = stretch_column * (positions_float - anchor_column) + anchor_column + shift_column
    identity_continuous = positions_float + shift_column
    continuous = torch.where(identity_mask[:, None], identity_continuous, continuous)

    discrete = torch.round(continuous).to(torch.long)
    identity_discrete = positions + shift_values.to(torch.long)[:, None]
    discrete = torch.where(identity_mask[:, None], identity_discrete, discrete)
    continuous = torch.where(mask, continuous, positions_float)
    discrete = torch.where(mask, discrete, positions.to(torch.long))

    duplicate_counts = []
    duplicate_ratios = []
    continuous_monotonic = []
    discrete_nondecreasing = []
    for batch_index in range(batch_size):
        valid_continuous = continuous[batch_index][mask[batch_index]]
        valid_discrete = discrete[batch_index][mask[batch_index]]
        pair_count = max(valid_discrete.numel() - 1, 0)
        if pair_count == 0:
            duplicate_count = 0
            duplicate_ratio = 0.0
            continuous_is_monotonic = True
            discrete_is_nondecreasing = True
        else:
            continuous_differences = torch.diff(valid_continuous)
            discrete_differences = torch.diff(valid_discrete)
            duplicate_count = int((discrete_differences == 0).sum().item())
            duplicate_ratio = duplicate_count / pair_count
            continuous_is_monotonic = bool((continuous_differences > 0).all().item())
            discrete_is_nondecreasing = bool((discrete_differences >= 0).all().item())
        duplicate_counts.append(duplicate_count)
        duplicate_ratios.append(duplicate_ratio)
        continuous_monotonic.append(continuous_is_monotonic)
        discrete_nondecreasing.append(discrete_is_nondecreasing)

    return AffineTransformResult(
        continuous_positions=continuous,
        discrete_positions=discrete,
        duplicate_count=torch.tensor(duplicate_counts, device=positions.device, dtype=torch.long),
        duplicate_ratio=torch.tensor(
            duplicate_ratios, device=positions.device, dtype=torch.float32
        ),
        continuous_monotonic=torch.tensor(
            continuous_monotonic, device=positions.device, dtype=torch.bool
        ),
        discrete_nondecreasing=torch.tensor(
            discrete_nondecreasing, device=positions.device, dtype=torch.bool
        ),
    )
