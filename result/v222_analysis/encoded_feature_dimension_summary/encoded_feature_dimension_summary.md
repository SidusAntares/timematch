# Encoded Feature-Curve Dimension Summary

This report keeps only encoded source feature-curve metrics.
Raw-source metrics, target-only metrics, and gap metrics are excluded.

## Ranked encoded metrics

| metric | dimension | transfer_corr | phase_corr | per_class_corr | source_spread | class_spread | combined_score |
|---|---|---:|---:|---:|---:|---:|---:|
| source_ndvi_q90_range_mean | global-feature-structure | 0.3797 | 0.0000 | 0.1092 | 0.0000 | 0.7307 | 0.1223 |
| source_ndvi_q90_range_std | global-feature-structure | 0.3428 | 0.0000 | 0.0759 | 0.0000 | 0.5915 | 0.1172 |
| source_phase_boundary_3 | other-encoded | 0.0000 | -0.4681 | 0.0000 | 0.3079 | 0.0000 | 0.1074 |
| source_phase_separability_p1 | phase-inter-contrast | 0.0000 | 0.6263 | 0.0000 | 0.7729 | 0.0000 | 0.1060 |
| source_phase_margin_p2 | phase-inter-contrast | 0.0000 | 0.4285 | 0.0000 | 0.5374 | 0.0000 | 0.0836 |
| source_phase_margin_p1 | phase-inter-contrast | 0.0000 | 0.5437 | 0.0000 | 1.3537 | 0.0000 | 0.0693 |
| source_phase_compactness_p2 | phase-intra-compactness | 0.0000 | 0.4517 | 0.0000 | 1.0812 | 0.0000 | 0.0651 |
| source_phase_margin_p3 | phase-inter-contrast | 0.0000 | 0.5814 | 0.0000 | 1.7044 | 0.0000 | 0.0645 |
| source_phase_boundary_1 | other-encoded | 0.0000 | -0.2101 | 0.0000 | 0.0000 | 0.0000 | 0.0630 |
| source_phase_compactness_p4 | phase-intra-compactness | 0.0000 | 0.5849 | 0.0000 | 1.8132 | 0.0000 | 0.0624 |
| source_phase_compactness_p1 | phase-intra-compactness | 0.0000 | 0.5372 | 0.0000 | 1.6113 | 0.0000 | 0.0617 |
| source_phase_compactness_p3 | phase-intra-compactness | 0.0000 | 0.5977 | 0.0000 | 1.9121 | 0.0000 | 0.0616 |
| source_phase_boundary_4 | other-encoded | 0.0000 | -0.4247 | 0.0000 | 1.0934 | 0.0000 | 0.0609 |
| source_phase_separability_p3 | phase-inter-contrast | 0.0000 | 0.4927 | 0.0000 | 1.7570 | 0.0000 | 0.0536 |
| source_phase_margin_p4 | phase-inter-contrast | 0.0000 | 0.4767 | 0.0000 | 1.9681 | 0.0000 | 0.0482 |
| source_phase_separability_p4 | phase-inter-contrast | 0.0000 | 0.3757 | 0.0000 | 1.4711 | 0.0000 | 0.0456 |
| source_phase_compactness_p5 | phase-intra-compactness | 0.0000 | 0.3952 | 0.0000 | 1.8084 | 0.0000 | 0.0422 |
| source_fisher_ratio | effect-discriminability | 0.3982 | 0.0000 | 0.1197 | 1.9767 | 1.4048 | 0.0312 |
| source_phase_separability_p2 | phase-inter-contrast | 0.0000 | 0.2103 | 0.0000 | 1.3037 | 0.0000 | 0.0274 |
| source_phase_boundary_2 | other-encoded | 0.0000 | -0.0984 | 0.0000 | 0.4401 | 0.0000 | 0.0205 |
| source_class_curve_variance_mean | global-feature-structure | 0.0616 | 0.0000 | 0.0398 | 0.0000 | 0.9562 | 0.0198 |
| source_curve_spread | global-feature-structure | -0.1903 | 0.0000 | -0.0235 | 1.8004 | 1.5486 | 0.0140 |
| source_curve_activity_range | global-feature-structure | -0.2047 | 0.0000 | -0.0267 | 1.9171 | 1.9045 | 0.0127 |
| source_phase_separability_p5 | phase-inter-contrast | 0.0000 | -0.0104 | 0.0000 | 1.5772 | 0.0000 | 0.0012 |
| source_phase_margin_p5 | phase-inter-contrast | 0.0000 | 0.0056 | 0.0000 | 1.7794 | 0.0000 | 0.0006 |

## Dimension summary

| dimension | top_metric | top_score | avg_score | metric_count |
|---|---|---:|---:|---:|
| other-encoded | source_phase_boundary_3 | 0.1074 | 0.0629 | 4 |
| phase-intra-compactness | source_phase_compactness_p2 | 0.0651 | 0.0586 | 5 |
| global-feature-structure | source_ndvi_q90_range_mean | 0.1223 | 0.0572 | 5 |
| phase-inter-contrast | source_phase_separability_p1 | 0.1060 | 0.0500 | 10 |
| effect-discriminability | source_fisher_ratio | 0.0312 | 0.0312 | 1 |
