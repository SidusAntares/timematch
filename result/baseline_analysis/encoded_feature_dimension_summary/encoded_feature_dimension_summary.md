# Encoded Feature-Curve Dimension Summary

This report keeps only encoded source feature-curve metrics.
Raw-source metrics, target-only metrics, and gap metrics are excluded.

## Ranked encoded metrics

| metric | dimension | transfer_corr | phase_corr | per_class_corr | source_spread | class_spread | combined_score |
|---|---|---:|---:|---:|---:|---:|---:|
| source_ndvi_q90_range_mean | global-feature-structure | 0.6196 | 0.0000 | 0.1992 | 0.0000 | 0.7361 | 0.2014 |
| source_ndvi_q90_range_std | global-feature-structure | 0.3689 | 0.0000 | 0.1055 | 0.0000 | 0.8689 | 0.1100 |
| source_phase_compactness_p4 | phase-intra-compactness | 0.0000 | 0.5064 | 0.0000 | 0.3835 | 0.0000 | 0.1098 |
| source_phase_compactness_p2 | phase-intra-compactness | 0.0000 | 0.6872 | 0.0000 | 1.0359 | 0.0000 | 0.1013 |
| source_phase_compactness_p1 | phase-intra-compactness | 0.0000 | 0.6259 | 0.0000 | 1.2928 | 0.0000 | 0.0819 |
| source_curve_activity_range | global-feature-structure | -0.5061 | 0.0000 | -0.1214 | 0.7961 | 0.9042 | 0.0811 |
| source_class_curve_variance_mean | global-feature-structure | 0.2708 | 0.0000 | 0.1044 | 0.0000 | 1.0464 | 0.0764 |
| source_phase_margin_p3 | phase-inter-contrast | 0.0000 | 0.5612 | 0.0000 | 1.5366 | 0.0000 | 0.0664 |
| source_phase_compactness_p5 | phase-intra-compactness | 0.0000 | 0.4207 | 0.0000 | 0.9576 | 0.0000 | 0.0645 |
| source_phase_compactness_p3 | phase-intra-compactness | 0.0000 | 0.4978 | 0.0000 | 1.5929 | 0.0000 | 0.0576 |
| source_curve_spread | global-feature-structure | -0.4330 | 0.0000 | -0.1095 | 0.9546 | 1.2249 | 0.0548 |
| source_phase_separability_p3 | phase-inter-contrast | 0.0000 | 0.3506 | 0.0000 | 1.0425 | 0.0000 | 0.0515 |
| source_fisher_ratio | effect-discriminability | 0.4941 | 0.0000 | 0.1765 | 1.5972 | 1.4830 | 0.0438 |
| source_phase_separability_p1 | phase-inter-contrast | 0.0000 | -0.2481 | 0.0000 | 0.8782 | 0.0000 | 0.0396 |
| source_phase_margin_p2 | phase-inter-contrast | 0.0000 | 0.2654 | 0.0000 | 1.0496 | 0.0000 | 0.0388 |
| source_phase_boundary_3 | other-encoded | 0.0000 | -0.2421 | 0.0000 | 1.0287 | 0.0000 | 0.0358 |
| source_phase_separability_p2 | phase-inter-contrast | 0.0000 | 0.1125 | 0.0000 | 1.0272 | 0.0000 | 0.0167 |
| source_phase_separability_p5 | phase-inter-contrast | 0.0000 | 0.1197 | 0.0000 | 1.6669 | 0.0000 | 0.0135 |
| source_phase_margin_p4 | phase-inter-contrast | 0.0000 | 0.1208 | 0.0000 | 1.8021 | 0.0000 | 0.0129 |
| source_phase_separability_p4 | phase-inter-contrast | 0.0000 | 0.1031 | 0.0000 | 1.6788 | 0.0000 | 0.0115 |
| source_phase_margin_p5 | phase-inter-contrast | 0.0000 | 0.0838 | 0.0000 | 1.3888 | 0.0000 | 0.0105 |
| source_phase_boundary_4 | other-encoded | 0.0000 | -0.0346 | 0.0000 | 0.1134 | 0.0000 | 0.0093 |
| source_phase_boundary_1 | other-encoded | 0.0000 | 0.0561 | 0.0000 | 1.0287 | 0.0000 | 0.0083 |
| source_phase_boundary_2 | other-encoded | 0.0000 | -0.0541 | 0.0000 | 1.7011 | 0.0000 | 0.0060 |
| source_phase_margin_p1 | phase-inter-contrast | 0.0000 | -0.0095 | 0.0000 | 1.9884 | 0.0000 | 0.0010 |

## Dimension summary

| dimension | top_metric | top_score | avg_score | metric_count |
|---|---|---:|---:|---:|
| global-feature-structure | source_ndvi_q90_range_mean | 0.2014 | 0.1047 | 5 |
| phase-intra-compactness | source_phase_compactness_p4 | 0.1098 | 0.0830 | 5 |
| effect-discriminability | source_fisher_ratio | 0.0438 | 0.0438 | 1 |
| phase-inter-contrast | source_phase_margin_p3 | 0.0664 | 0.0262 | 10 |
| other-encoded | source_phase_boundary_3 | 0.0358 | 0.0149 | 4 |
