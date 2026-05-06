# Encoded Structure Dimension Analysis

This report analyzes source-side encoded feature curves only.
The goal is to compare four structural dimension families against downstream transfer `macro_f1`.

## Family Definitions

- `amplitude_spread`: temporal excursion and unfolding magnitude of encoded class curves.
- `inter_phase_contrast`: how strongly different phases separate inside encoded class curves.
- `intra_phase_compact_smooth`: within-phase radius and within-phase smoothness of encoded curves.
- `trend_activity`: how actively encoded curves move across time and how directional that motion is.

## Family Summary

| family | top_metric | top_corr | top_abs_corr | avg_abs_corr | metric_count |
|---|---|---:|---:|---:|---:|
| intra_phase_compact_smooth | source_encoded_phase_within_radius_mean | -0.5247 | 0.5247 | 0.3097 | 3 |
| amplitude_spread | source_encoded_amplitude_l2_std | 0.4757 | 0.4757 | 0.2755 | 3 |
| trend_activity | source_encoded_activity_path_length_mean | -0.2864 | 0.2864 | 0.2395 | 3 |
| inter_phase_contrast | source_encoded_phase_early_late_distance_mean | 0.1825 | 0.1825 | 0.1691 | 3 |

## Top Overall Metrics

| metric | family | corr_with_target_f1 | abs_corr |
|---|---|---:|---:|
| source_encoded_phase_within_radius_mean | intra_phase_compact_smooth | -0.5247 | 0.5247 |
| source_encoded_amplitude_l2_std | amplitude_spread | 0.4757 | 0.4757 |
| source_encoded_activity_path_length_mean | trend_activity | -0.2864 | 0.2864 |
| source_encoded_activity_directionality_mean | trend_activity | 0.2862 | 0.2862 |
| source_encoded_amplitude_l2_mean | amplitude_spread | -0.2764 | 0.2764 |
| source_encoded_phase_internal_smoothness_mean | intra_phase_compact_smooth | -0.2611 | 0.2611 |
| source_encoded_phase_early_late_distance_mean | inter_phase_contrast | 0.1825 | 0.1825 |
| source_encoded_phase_adjacent_contrast_mean | inter_phase_contrast | -0.1672 | 0.1672 |
| source_encoded_phase_global_contrast_mean | inter_phase_contrast | -0.1575 | 0.1575 |
| source_encoded_activity_step_norm_std | trend_activity | 0.1460 | 0.1460 |
| source_encoded_phase_within_radius_std | intra_phase_compact_smooth | -0.1433 | 0.1433 |
| source_encoded_temporal_norm_std_mean | amplitude_spread | 0.0745 | 0.0745 |

## Source-Level Metric Table

| source | target | target_f1 | source_encoded_amplitude_l2_mean | source_encoded_phase_adjacent_contrast_mean | source_encoded_phase_within_radius_mean | source_encoded_activity_path_length_mean |
|---|---|---:|---:|---:|---:|---:|
| FR1 | FR2 | 0.7856 | 6.2778 | 1.4994 | 2.5830 | 44.7085 |
| FR1 | DK1 | 0.6765 | 6.2778 | 1.4994 | 2.5830 | 44.7085 |
| FR1 | AT1 | 0.7045 | 6.2778 | 1.4994 | 2.5830 | 44.7085 |
| FR2 | FR1 | 0.7559 | 8.7205 | 2.3975 | 3.0806 | 50.6810 |
| FR2 | DK1 | 0.6928 | 8.7205 | 2.3975 | 3.0806 | 50.6810 |
| FR2 | AT1 | 0.6805 | 8.7205 | 2.3975 | 3.0806 | 50.6810 |
| DK1 | FR1 | 0.5965 | 8.4445 | 2.0828 | 3.5819 | 56.2095 |
| DK1 | FR2 | 0.5128 | 8.4445 | 2.0828 | 3.5819 | 56.2095 |
| DK1 | AT1 | 0.6841 | 8.4445 | 2.0828 | 3.5819 | 56.2095 |
| AT1 | FR1 | 0.6657 | 9.4959 | 2.3538 | 3.6470 | 69.6078 |
| AT1 | FR2 | 0.6241 | 9.4959 | 2.3538 | 3.6470 | 69.6078 |
| AT1 | DK1 | 0.7338 | 9.4959 | 2.3538 | 3.6470 | 69.6078 |
