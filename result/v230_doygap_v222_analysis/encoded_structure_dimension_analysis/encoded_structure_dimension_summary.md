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
| inter_phase_contrast | source_encoded_interphase_global_separation_mean | -0.7962 | 0.7962 | 0.6611 | 5 |
| intra_phase_compact_smooth | source_encoded_intraphase_late_early_radius_ratio_mean | 0.7872 | 0.7872 | 0.6413 | 4 |
| amplitude_spread | source_encoded_amplitude_excursion_cv | 0.7507 | 0.7507 | 0.6032 | 4 |
| trend_activity | source_encoded_trend_path_length_mean | -0.5058 | 0.5058 | 0.3634 | 4 |

## Top Source-Mean Metrics

| metric | family | corr_with_source_mean_f1 | abs_corr |
|---|---|---:|---:|
| source_encoded_interphase_global_separation_mean | inter_phase_contrast | -0.7962 | 0.7962 |
| source_encoded_intraphase_late_early_radius_ratio_mean | intra_phase_compact_smooth | 0.7872 | 0.7872 |
| source_encoded_intraphase_radius_mean | intra_phase_compact_smooth | -0.7632 | 0.7632 |
| source_encoded_amplitude_excursion_cv | amplitude_spread | 0.7507 | 0.7507 |
| source_encoded_interphase_adjacent_jump_mean | inter_phase_contrast | -0.7456 | 0.7456 |
| source_encoded_interphase_early_late_shift_mean | inter_phase_contrast | -0.7363 | 0.7363 |
| source_encoded_amplitude_excursion_mean | amplitude_spread | -0.7081 | 0.7081 |
| source_encoded_interphase_adjacent_jump_cv | inter_phase_contrast | 0.5808 | 0.5808 |
| source_encoded_intraphase_radius_phase_cv_mean | intra_phase_compact_smooth | 0.5701 | 0.5701 |
| source_encoded_trend_path_length_mean | trend_activity | -0.5058 | 0.5058 |
| source_encoded_amplitude_late_early_energy_ratio_mean | amplitude_spread | -0.5007 | 0.5007 |
| source_encoded_trend_curvature_mean | trend_activity | -0.4655 | 0.4655 |

## Source Summary Table

| source | source_mean_f1 | source_f1_std | amplitude_excursion_mean | interphase_adjacent_jump_mean | intraphase_radius_mean | trend_path_length_mean |
|---|---:|---:|---:|---:|---:|---:|
| AT1 | 0.6778 | 0.0530 | 6.1293 | 1.4917 | 2.3413 | 44.1617 |
| DK1 | 0.5334 | 0.0225 | 5.9078 | 1.4709 | 2.3499 | 37.0511 |
| FR1 | 0.7529 | 0.0708 | 3.3239 | 0.8653 | 1.2712 | 20.7600 |
| FR2 | 0.6358 | 0.0548 | 5.0611 | 1.3298 | 1.9920 | 30.3584 |
