"""Multi-scale temporal structure shaping components."""

from .taxonomy import (
    ACTIVE_STRUCTURE_VIEWS,
    STRUCTURE_VIEW_CHOICES,
    STRUCTURE_VIEW_REGISTRY,
    get_structure_view,
    is_supported_structure_view,
    normalize_structure_view,
)

__all__ = [
    "ACTIVE_STRUCTURE_VIEWS",
    "STRUCTURE_VIEW_CHOICES",
    "STRUCTURE_VIEW_REGISTRY",
    "get_structure_view",
    "is_supported_structure_view",
    "normalize_structure_view",
]
