"""Pure temporal-position transformations and diagnostics."""

from .affine_transform import AffineTransformResult, transform_positions
from .range_gate import RangeStatistics, apply_shift_only_fallback, compute_range_statistics

__all__ = [
    "AffineTransformResult",
    "RangeStatistics",
    "apply_shift_only_fallback",
    "compute_range_statistics",
    "transform_positions",
]
