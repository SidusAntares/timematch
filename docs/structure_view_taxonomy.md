# Multi-Scale Temporal Structure Shaping

The current mainline treats source-side structure regularization as a family of
temporal structure views, not as a single fixed loss. The source reshaper is the
common module; each view describes which temporal scale should be made clearer
before TimeMatch domain adaptation.

## Active Views

| View | Scale | Segmented | Intra Granularity | Purpose |
|---|---|---:|---|---|
| `global_compact` | global semantic | no | global mean | Make whole-series class representations compact. |
| `segmented_transition` | segment/transition | yes | segment mean | Preserve stage-level class structure and inter-stage transitions. |
| `segmented_light_transition` | segment/weak transition | yes | segment mean | Keep segment structure with weaker transition or boundary pressure. |
| `pointwise_dynamics` | pointwise trajectory | no | pointwise curve | Strongly align same-class curves and prototype dynamics. |
| `global_trajectory` | global trajectory | no | meanmax pooled | Compact pooled trajectories and weakly align prototype dynamics. |

## Interpretation

- `global_compact` corresponds to the successful no-segment strong-intra view.
- `segmented_transition` corresponds to the v2.4.3b boundary-window transition
  family.
- `pointwise_dynamics` corresponds to v2.4.4a.
- `global_trajectory` corresponds to v2.4.4b.

The view taxonomy is the bridge from empirical sweeps to a unified method:
different source-target pairs can require different temporal scales, but they
are all instances of source-side temporal structure shaping.

## Current Workflow

1. Use the gain-validation launchers to identify which view improves each task.
2. Record the best result in a structure-view table.
3. Validate at least one source-side structure view on a non-crop time-series
   dataset, such as HAR.
4. Only after the broad effect is established, revisit automatic view/weight
   adaptation.

## Code Pointers

- `ideas/temporal_structure/taxonomy.py` is the single source of truth for view
  names and aliases.
- `ideas/temporal_structure/reshaper.py` contains the source-side temporal
  reshaper and its regularizers.
- `ideas/source_phase_compactness.py` remains the compatibility wrapper for the
  existing loss implementation while the views are gradually extracted.
- `launchers/gain_validation/` contains the active experiment scripts.
