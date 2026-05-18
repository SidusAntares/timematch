"""Verify source-structure view aliases after cleanup.

This is a lightweight CPU-only guard. It does not run training; it checks that
the old experiment config names still resolve to a documented view.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from ideas.temporal_structure.taxonomy import (
    ACTIVE_STRUCTURE_VIEWS,
    STRUCTURE_VIEW_CHOICES,
    get_structure_view,
    normalize_structure_view,
)


EXPECTED_ALIASES = {
    "compactness": "compactness",
    "multi_component": "multi_component",
    "profiled_components": "profiled_components",
    "trend_residual": "trend_residual",
    "trend_seasonal_residual": "trend_seasonal_residual",
    "segment_transition_residual": "segmented_light_transition",
    "segment_transition_semantic": "segment_transition_semantic",
    "segment_boundary_window_residual": "segmented_transition",
    "segment_boundary_window_warp_residual": "segment_boundary_window_warp_residual",
    "trajectory_prototype_dynamics": "pointwise_dynamics",
    "trajectory_prototype_dynamics_v244b": "global_trajectory",
    "global_compact": "global_compact",
    "segmented_transition": "segmented_transition",
    "segmented_light_transition": "segmented_light_transition",
    "pointwise_dynamics": "pointwise_dynamics",
    "global_trajectory": "global_trajectory",
}


def main():
    errors = []
    for alias, expected in EXPECTED_ALIASES.items():
        actual = normalize_structure_view(alias)
        if actual != expected:
            errors.append(f"{alias}: expected {expected}, got {actual}")
            continue
        view = get_structure_view(alias)
        if view.name != expected:
            errors.append(f"{alias}: get_structure_view returned {view.name}, expected {expected}")
        if alias not in STRUCTURE_VIEW_CHOICES:
            errors.append(f"{alias}: missing from train.py argparse choices")

    for active_name in ACTIVE_STRUCTURE_VIEWS:
        view = get_structure_view(active_name)
        if not view.active:
            errors.append(f"{active_name}: listed active but metadata.active is false")

    if errors:
        raise SystemExit("Structure view taxonomy verification failed:\n" + "\n".join(errors))

    print("Structure view taxonomy verification passed.")
    print("Active views:", ", ".join(ACTIVE_STRUCTURE_VIEWS))


if __name__ == "__main__":
    main()
