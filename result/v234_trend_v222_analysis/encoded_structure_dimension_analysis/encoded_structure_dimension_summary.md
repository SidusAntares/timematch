# Encoded Structure Dimension Analysis

This report analyzes source-side encoded feature curves only.
The goal is to compare four structural dimension families against downstream transfer `macro_f1`.
Primary correlations are computed against `source_mean_f1`, which matches the source-only design goal.

## Family Definitions

- `amplitude_spread`: excursion magnitude, phase energy distribution, and late/early energy bias of encoded class curves.
- `inter_phase_contrast`: adjacent phase jumps, boundary sharpness, and long-range phase displacement.
- `intra_phase_compact_smooth`: within-phase radius, phase-wise compactness heterogeneity, and internal step smoothness.
- `trend_activity`: path length, directionality, curvature, and burstiness across encoded time trajectories.

## Family Summary

| family | top_metric | top_corr | top_abs_corr | avg_abs_corr | metric_count |
|---|---|---:|---:|---:|---:|
| inter_phase_contrast | source_encoded_interphase_adjacent_jump_cv | 0.7413 | 0.7413 | 0.6668 | 5 |
| intra_phase_compact_smooth | source_encoded_intraphase_radius_mean | -0.8525 | 0.8525 | 0.6603 | 4 |
| trend_activity | source_encoded_trend_burstiness_mean | 0.8482 | 0.8482 | 0.6448 | 4 |
| amplitude_spread | source_encoded_amplitude_phase_energy_cv_mean | 0.9449 | 0.9449 | 0.6387 | 4 |

## Top Source-Mean Metrics

| metric | family | corr_with_source_mean_f1 | abs_corr |
|---|---|---:|---:|
| source_encoded_amplitude_phase_energy_cv_mean | amplitude_spread | 0.9449 | 0.9449 |
| source_encoded_intraphase_radius_mean | intra_phase_compact_smooth | -0.8525 | 0.8525 |
| source_encoded_trend_burstiness_mean | trend_activity | 0.8482 | 0.8482 |
| source_encoded_amplitude_excursion_cv | amplitude_spread | 0.8140 | 0.8140 |
| source_encoded_interphase_adjacent_jump_cv | inter_phase_contrast | 0.7413 | 0.7413 |
| source_encoded_intraphase_late_early_radius_ratio_mean | intra_phase_compact_smooth | 0.7310 | 0.7310 |
| source_encoded_amplitude_excursion_mean | amplitude_spread | -0.7204 | 0.7204 |
| source_encoded_interphase_boundary_sharpness_mean | inter_phase_contrast | 0.6992 | 0.6992 |
| source_encoded_interphase_global_separation_mean | inter_phase_contrast | -0.6878 | 0.6878 |
| source_encoded_interphase_adjacent_jump_mean | inter_phase_contrast | -0.6413 | 0.6413 |
| source_encoded_trend_path_length_mean | trend_activity | -0.6254 | 0.6254 |
| source_encoded_trend_curvature_mean | trend_activity | -0.5833 | 0.5833 |

## Source Summary Table

| source | source_mean_f1 | source_f1_std | amplitude_excursion_mean | interphase_adjacent_jump_mean | intraphase_radius_mean | trend_path_length_mean |
|---|---:|---:|---:|---:|---:|---:|
| AT1 | 0.6741 | 0.0528 | 7.1488 | 1.7804 | 2.5416 | 48.9503 |
| DK1 | 0.5669 | 0.0506 | 6.6272 | 1.6012 | 2.6415 | 41.7884 |
| FR1 | 0.7479 | 0.0650 | 3.8521 | 1.0187 | 1.5432 | 24.8857 |
| FR2 | 0.6949 | 0.0237 | 5.4420 | 1.4638 | 2.1363 | 32.1959 |
