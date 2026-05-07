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
| amplitude_spread | source_encoded_amplitude_late_early_energy_ratio_mean | 0.7011 | 0.7011 | 0.5457 | 4 |
| intra_phase_compact_smooth | source_encoded_intraphase_late_early_radius_ratio_mean | 0.9662 | 0.9662 | 0.5138 | 4 |
| inter_phase_contrast | source_encoded_interphase_adjacent_jump_mean | -0.5554 | 0.5554 | 0.3419 | 5 |
| trend_activity | source_encoded_trend_burstiness_mean | 0.5264 | 0.5264 | 0.2778 | 4 |

## Top Source-Mean Metrics

| metric | family | corr_with_source_mean_f1 | abs_corr |
|---|---|---:|---:|
| source_encoded_intraphase_late_early_radius_ratio_mean | intra_phase_compact_smooth | 0.9662 | 0.9662 |
| source_encoded_amplitude_late_early_energy_ratio_mean | amplitude_spread | 0.7011 | 0.7011 |
| source_encoded_intraphase_radius_phase_cv_mean | intra_phase_compact_smooth | 0.5603 | 0.5603 |
| source_encoded_amplitude_excursion_cv | amplitude_spread | 0.5557 | 0.5557 |
| source_encoded_interphase_adjacent_jump_mean | inter_phase_contrast | -0.5554 | 0.5554 |
| source_encoded_trend_burstiness_mean | trend_activity | 0.5264 | 0.5264 |
| source_encoded_intraphase_radius_mean | intra_phase_compact_smooth | -0.5091 | 0.5091 |
| source_encoded_interphase_global_separation_mean | inter_phase_contrast | -0.5088 | 0.5088 |
| source_encoded_amplitude_excursion_mean | amplitude_spread | -0.4935 | 0.4935 |
| source_encoded_amplitude_phase_energy_cv_mean | amplitude_spread | -0.4324 | 0.4324 |
| source_encoded_trend_directionality_mean | trend_activity | -0.4182 | 0.4182 |
| source_encoded_interphase_adjacent_jump_cv | inter_phase_contrast | 0.3795 | 0.3795 |

## Source Summary Table

| source | source_mean_f1 | source_f1_std | amplitude_excursion_mean | interphase_adjacent_jump_mean | intraphase_radius_mean | trend_path_length_mean |
|---|---:|---:|---:|---:|---:|---:|
| AT1 | 0.6866 | 0.0776 | 9.4382 | 2.2762 | 3.7012 | 69.0529 |
| DK1 | 0.5625 | 0.0971 | 8.4972 | 2.0843 | 3.5025 | 55.7250 |
| FR1 | 0.7286 | 0.0351 | 6.3398 | 1.4659 | 2.5841 | 44.5397 |
| FR2 | 0.6214 | 0.0778 | 8.9091 | 2.4279 | 3.1382 | 51.5472 |
