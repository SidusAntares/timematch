"""Pure range diagnostics and shift-only fallback for temporal positions."""

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
class RangeStatistics:
    """Per-sample range diagnostics; every field has shape ``[B]``."""

    valid_point_count: torch.Tensor
    in_source_range_count: torch.Tensor
    out_of_source_range_count: torch.Tensor
    in_embedding_range_count: torch.Tensor
    out_of_embedding_range_count: torch.Tensor
    original_span: torch.Tensor
    transformed_span: torch.Tensor
    source_overlap_span: torch.Tensor
    source_overlap_ratio: torch.Tensor
    in_source_point_ratio: torch.Tensor
    max_gap: torch.Tensor
    duplicate_count: torch.Tensor
    duplicate_ratio: torch.Tensor
    embedding_range_valid: torch.Tensor


def _validate_position_pair(
    original_positions: torch.Tensor,
    transformed_positions: torch.Tensor,
) -> None:
    if not isinstance(original_positions, torch.Tensor) or not isinstance(
        transformed_positions, torch.Tensor
    ):
        raise TypeError("position inputs must be torch.Tensor instances")
    if original_positions.ndim != 2 or transformed_positions.shape != original_positions.shape:
        raise ValueError("position inputs must share shape [B, T]")
    if original_positions.dtype not in _INTEGER_DTYPES:
        raise TypeError("original_positions must be an integer tensor")
    if transformed_positions.dtype == torch.bool or torch.is_complex(transformed_positions):
        raise TypeError("transformed_positions must be a real numeric tensor")
    if transformed_positions.dtype not in _INTEGER_DTYPES and not torch.is_floating_point(
        transformed_positions
    ):
        raise TypeError("transformed_positions must be a real numeric tensor")


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


def compute_range_statistics(
    original_positions: torch.Tensor,
    transformed_positions: torch.Tensor,
    source_min: NumberOrTensor,
    source_max: NumberOrTensor,
    embedding_index_min: NumberOrTensor,
    embedding_index_max: NumberOrTensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> RangeStatistics:
    """Compute source-range and embedding-range diagnostics.

    Args:
        original_positions: Integer tensor ``[B, T]``.
        transformed_positions: Finite real numeric tensor ``[B, T]`` in the
            same temporal unit. Discrete transformed positions should be
            supplied when checking an embedding table.
        source_min/source_max: Scalar or ``[B]`` source temporal bounds.
        embedding_index_min/embedding_index_max: Scalar or ``[B]`` legal
            embedding-position bounds before any model-internal offset.
        valid_mask: Optional boolean tensor ``[B, T]``.

    Zero-span definition: a nonempty transformed point interval has overlap
    ratio 1 when its point lies inside the closed source range, otherwise 0.
    An empty sample receives zero counts/ratios and
    ``embedding_range_valid=False``. No thresholds or gate decisions are made.
    """

    _validate_position_pair(original_positions, transformed_positions)
    batch_size = original_positions.shape[0]
    device = original_positions.device
    transformed_positions = transformed_positions.to(device=device)
    if not torch.isfinite(transformed_positions).all():
        raise ValueError("transformed_positions must contain only finite values")
    mask = _valid_mask(valid_mask, original_positions)
    source_min_values = _batch_parameter(
        source_min, name="source_min", batch_size=batch_size, device=device
    )
    source_max_values = _batch_parameter(
        source_max, name="source_max", batch_size=batch_size, device=device
    )
    embedding_min_values = _batch_parameter(
        embedding_index_min,
        name="embedding_index_min",
        batch_size=batch_size,
        device=device,
    )
    embedding_max_values = _batch_parameter(
        embedding_index_max,
        name="embedding_index_max",
        batch_size=batch_size,
        device=device,
    )
    for name, values in (
        ("source_min", source_min_values),
        ("source_max", source_max_values),
        ("embedding_index_min", embedding_min_values),
        ("embedding_index_max", embedding_max_values),
    ):
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
    if (source_min_values > source_max_values).any():
        raise ValueError("source_min must not exceed source_max")
    if (embedding_min_values > embedding_max_values).any():
        raise ValueError("embedding_index_min must not exceed embedding_index_max")

    rows = {field: [] for field in RangeStatistics.__dataclass_fields__}
    original_float = original_positions.to(torch.float32)
    transformed_float = transformed_positions.to(torch.float32)

    for batch_index in range(batch_size):
        valid_original = original_float[batch_index][mask[batch_index]]
        valid_transformed = transformed_float[batch_index][mask[batch_index]]
        valid_count = valid_transformed.numel()
        source_low = source_min_values[batch_index]
        source_high = source_max_values[batch_index]
        embedding_low = embedding_min_values[batch_index]
        embedding_high = embedding_max_values[batch_index]

        if valid_count == 0:
            values = {
                "valid_point_count": 0,
                "in_source_range_count": 0,
                "out_of_source_range_count": 0,
                "in_embedding_range_count": 0,
                "out_of_embedding_range_count": 0,
                "original_span": 0.0,
                "transformed_span": 0.0,
                "source_overlap_span": 0.0,
                "source_overlap_ratio": 0.0,
                "in_source_point_ratio": 0.0,
                "max_gap": 0.0,
                "duplicate_count": 0,
                "duplicate_ratio": 0.0,
                "embedding_range_valid": False,
            }
        else:
            in_source = (valid_transformed >= source_low) & (valid_transformed <= source_high)
            in_embedding = (valid_transformed >= embedding_low) & (
                valid_transformed <= embedding_high
            )
            transformed_min = valid_transformed.min()
            transformed_max = valid_transformed.max()
            transformed_span = transformed_max - transformed_min
            overlap_span = torch.clamp(
                torch.minimum(transformed_max, source_high)
                - torch.maximum(transformed_min, source_low),
                min=0,
            )
            if transformed_span.item() == 0:
                overlap_ratio = float(in_source.all().item())
            else:
                overlap_ratio = float((overlap_span / transformed_span).item())

            if valid_count < 2:
                max_gap = 0.0
                duplicate_count = 0
                duplicate_ratio = 0.0
            else:
                differences = torch.diff(valid_transformed)
                max_gap = float(differences.max().item())
                duplicate_count = int((differences == 0).sum().item())
                duplicate_ratio = duplicate_count / (valid_count - 1)

            values = {
                "valid_point_count": valid_count,
                "in_source_range_count": int(in_source.sum().item()),
                "out_of_source_range_count": int((~in_source).sum().item()),
                "in_embedding_range_count": int(in_embedding.sum().item()),
                "out_of_embedding_range_count": int((~in_embedding).sum().item()),
                "original_span": float((valid_original.max() - valid_original.min()).item()),
                "transformed_span": float(transformed_span.item()),
                "source_overlap_span": float(overlap_span.item()),
                "source_overlap_ratio": overlap_ratio,
                "in_source_point_ratio": float(in_source.float().mean().item()),
                "max_gap": max_gap,
                "duplicate_count": duplicate_count,
                "duplicate_ratio": duplicate_ratio,
                "embedding_range_valid": bool(in_embedding.all().item()),
            }

        for field, value in values.items():
            rows[field].append(value)

    integer_fields = {
        "valid_point_count",
        "in_source_range_count",
        "out_of_source_range_count",
        "in_embedding_range_count",
        "out_of_embedding_range_count",
        "duplicate_count",
    }
    tensor_rows = {}
    for field, values in rows.items():
        if field == "embedding_range_valid":
            dtype = torch.bool
        elif field in integer_fields:
            dtype = torch.long
        else:
            dtype = torch.float32
        tensor_rows[field] = torch.tensor(values, device=device, dtype=dtype)
    return RangeStatistics(**tensor_rows)


def apply_shift_only_fallback(
    original_positions: torch.Tensor,
    shift: NumberOrTensor,
    use_affine: Union[bool, torch.Tensor],
    affine_positions: torch.Tensor,
) -> torch.Tensor:
    """Select affine or exact shift-only positions for each batch sample.

    Args:
        original_positions: Integer tensor ``[B, T]``.
        shift: Integral scalar or tensor ``[B]``.
        use_affine: Boolean scalar or tensor ``[B]`` selected externally.
        affine_positions: Integer tensor ``[B, T]``.

    Returns:
        A new ``torch.long`` tensor. Rejected samples are exactly
        ``original_positions + shift``. No clipping or gate decision occurs.
    """

    _validate_position_pair(original_positions, affine_positions)
    if affine_positions.dtype not in _INTEGER_DTYPES:
        raise TypeError("affine_positions must be an integer tensor")
    batch_size = original_positions.shape[0]
    shift_values = _batch_parameter(
        shift, name="shift", batch_size=batch_size, device=original_positions.device
    )
    if not torch.isfinite(shift_values).all() or not torch.equal(
        shift_values, torch.round(shift_values)
    ):
        raise ValueError("shift must contain finite integral values")

    decision = torch.as_tensor(use_affine, device=original_positions.device, dtype=torch.bool)
    if decision.ndim == 0:
        decision = decision.expand(batch_size)
    elif decision.ndim != 1 or decision.shape[0] != batch_size:
        raise ValueError("use_affine must be a scalar or have shape [B]")

    shifted_positions = original_positions + shift_values.to(torch.long)[:, None]
    return torch.where(decision[:, None], affine_positions.to(torch.long), shifted_positions)
