"""Canonical structure-view taxonomy for source-side temporal shaping.

This file is the single source of truth for structure-view names.  The old
research code used many version strings directly inside the parser, launchers,
and loss dispatch; keeping them here avoids parser/loss whitelist drift.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureView:
    name: str
    family: str
    description: str
    segmented: bool
    intra_granularity: str
    aliases: tuple[str, ...] = ()
    active: bool = True


STRUCTURE_VIEW_REGISTRY = {
    "global_compact": StructureView(
        name="global_compact",
        family="global",
        description="Non-segmented global class compactness over whole-series representations.",
        segmented=False,
        intra_granularity="global mean",
        aliases=("noseg_global_compact",),
    ),
    "segmented_transition": StructureView(
        name="segmented_transition",
        family="segment",
        description="Segment-level compactness plus trend/transition/boundary constraints.",
        segmented=True,
        intra_granularity="segment mean",
        aliases=(
            "segment_boundary_window_residual",
            "segment_boundary_window",
            "boundary_window_segment",
            "v243",
        ),
    ),
    "segmented_light_transition": StructureView(
        name="segmented_light_transition",
        family="segment",
        description="Segment-level compactness with weak transition or disabled boundary terms.",
        segmented=True,
        intra_granularity="segment mean",
        aliases=("segment_transition_residual", "segment_transition", "segment_inter", "v241"),
    ),
    "pointwise_dynamics": StructureView(
        name="pointwise_dynamics",
        family="trajectory",
        description="Pointwise curve compactness against class prototype plus prototype dynamics.",
        segmented=False,
        intra_granularity="pointwise curve",
        aliases=(
            "trajectory_prototype_dynamics",
            "whole_curve_prototype_dynamics",
            "prototype_dynamics",
            "v244",
        ),
    ),
    "global_trajectory": StructureView(
        name="global_trajectory",
        family="trajectory",
        description="Pooled trajectory compactness plus weak time-aware prototype dynamics.",
        segmented=False,
        intra_granularity="meanmax pooled",
        aliases=("trajectory_prototype_dynamics_v244b", "trajectory_global_prototype_dynamics", "v244b"),
    ),
    # Deprecated legacy views retained only so old configs fail less abruptly.
    "compactness": StructureView(
        name="compactness",
        family="legacy",
        description="Legacy phase compactness objective.",
        segmented=True,
        intra_granularity="phase mean",
        aliases=(),
        active=False,
    ),
    "multi_component": StructureView(
        name="multi_component",
        family="legacy",
        description="Legacy amplitude/interphase multi-component objective.",
        segmented=True,
        intra_granularity="phase mean",
        aliases=("multicomponent", "v232"),
        active=False,
    ),
    "profiled_components": StructureView(
        name="profiled_components",
        family="legacy",
        description="Legacy profiled shape objective.",
        segmented=True,
        intra_granularity="phase mean",
        aliases=("profiled", "v233"),
        active=False,
    ),
    "trend_residual": StructureView(
        name="trend_residual",
        family="legacy",
        description="Legacy trend residual objective.",
        segmented=True,
        intra_granularity="phase mean",
        aliases=("trend", "v234", "segment_trend_residual", "segment_trend", "v240"),
        active=False,
    ),
    "trend_seasonal_residual": StructureView(
        name="trend_seasonal_residual",
        family="legacy",
        description="Legacy trend-seasonal residual objective.",
        segmented=True,
        intra_granularity="phase mean",
        aliases=("trend_season", "season_pattern", "v235"),
        active=False,
    ),
    "segment_transition_semantic": StructureView(
        name="segment_transition_semantic",
        family="legacy",
        description="Legacy semantic segment transition objective.",
        segmented=True,
        intra_granularity="semantic segment mean",
        aliases=("v242",),
        active=False,
    ),
    "segment_boundary_window_warp_residual": StructureView(
        name="segment_boundary_window_warp_residual",
        family="legacy",
        description="Legacy GTW-inspired relaxed boundary objective.",
        segmented=True,
        intra_granularity="segment mean",
        aliases=("warp_boundary_window_segment",),
        active=False,
    ),
}


def _alias_map():
    mapping = {}
    for name, view in STRUCTURE_VIEW_REGISTRY.items():
        mapping[name] = name
        for alias in view.aliases:
            mapping[alias] = name
    return mapping


_ALIASES = _alias_map()
STRUCTURE_VIEW_CHOICES = sorted(_ALIASES.keys())
ACTIVE_STRUCTURE_VIEWS = tuple(
    name for name, view in STRUCTURE_VIEW_REGISTRY.items() if view.active
)


def normalize_structure_view(name):
    key = str(name or "compactness").lower()
    return _ALIASES.get(key)


def is_supported_structure_view(name):
    return normalize_structure_view(name) is not None


def get_structure_view(name):
    normalized = normalize_structure_view(name)
    if normalized is None:
        raise ValueError(f"Unsupported temporal structure view: {name}")
    return STRUCTURE_VIEW_REGISTRY[normalized]
