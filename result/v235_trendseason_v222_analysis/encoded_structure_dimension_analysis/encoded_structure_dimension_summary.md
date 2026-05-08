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
| intra_phase_compact_smooth | source_encoded_intraphase_late_early_radius_ratio_mean | 0.7925 | 0.7925 | 0.5901 | 4 |
| inter_phase_contrast | source_encoded_interphase_global_separation_mean | -0.5659 | 0.5659 | 0.5164 | 5 |
| amplitude_spread | source_encoded_amplitude_excursion_cv | 0.5806 | 0.5806 | 0.4798 | 4 |
| trend_activity | source_encoded_trend_burstiness_mean | 0.4890 | 0.4890 | 0.4082 | 4 |

## Top Source-Mean Metrics

| metric | family | corr_with_source_mean_f1 | abs_corr |
|---|---|---:|---:|
| source_encoded_intraphase_late_early_radius_ratio_mean | intra_phase_compact_smooth | 0.7925 | 0.7925 |
| source_encoded_intraphase_radius_mean | intra_phase_compact_smooth | -0.6502 | 0.6502 |
| source_encoded_amplitude_excursion_cv | amplitude_spread | 0.5806 | 0.5806 |
| source_encoded_intraphase_radius_phase_cv_mean | intra_phase_compact_smooth | 0.5751 | 0.5751 |
| source_encoded_interphase_global_separation_mean | inter_phase_contrast | -0.5659 | 0.5659 |
| source_encoded_interphase_adjacent_jump_cv | inter_phase_contrast | 0.5589 | 0.5589 |
| source_encoded_amplitude_excursion_mean | amplitude_spread | -0.5570 | 0.5570 |
| source_encoded_interphase_adjacent_jump_mean | inter_phase_contrast | -0.5380 | 0.5380 |
| source_encoded_trend_burstiness_mean | trend_activity | 0.4890 | 0.4890 |
| source_encoded_interphase_boundary_sharpness_mean | inter_phase_contrast | 0.4624 | 0.4624 |
| source_encoded_interphase_early_late_shift_mean | inter_phase_contrast | -0.4566 | 0.4566 |
| source_encoded_trend_path_length_mean | trend_activity | -0.4016 | 0.4016 |

## Source Summary Table

| source | source_mean_f1 | source_f1_std | amplitude_excursion_mean | interphase_adjacent_jump_mean | intraphase_radius_mean | trend_path_length_mean |
|---|---:|---:|---:|---:|---:|---:|
| AT1 | 0.6864 | 0.0528 | 6.2762 | 1.5764 | 2.4056 | 44.1442 |
| DK1 | 0.5299 | 0.0604 | 5.6785 | 1.4027 | 2.3444 | 36.6019 |
| FR1 | 0.7396 | 0.0531 | 3.1266 | 0.7836 | 1.2305 | 20.1744 |
| FR2 | 0.6344 | 0.0911 | 5.1127 | 1.3793 | 1.9141 | 28.5246 |
