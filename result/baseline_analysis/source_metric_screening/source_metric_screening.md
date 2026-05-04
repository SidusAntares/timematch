# Source Metric Screening

This report screens source-side metrics only.
Target-only and gap-style indicators are intentionally excluded.

## Top ranked source metrics

| metric | family | transfer_corr | phase_corr | per_class_corr | source_spread | class_spread | combined_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_source_curve_spread | raw-shape | 0.6464 | 0.0000 | 0.1816 | 0.0000 | 0.6822 | 0.2137 |
| raw_source_phase_separability_p1 | raw-phase | 0.0000 | 0.6903 | 0.0000 | 0.0000 | 0.0000 | 0.2071 |
| raw_source_phase_margin_p5 | raw-phase | 0.0000 | 0.6782 | 0.0000 | 0.0000 | 0.0000 | 0.2035 |
| raw_source_ndvi_q90_range_mean | raw-shape | 0.6196 | 0.0000 | 0.1992 | 0.0000 | 0.7361 | 0.2014 |
| source_ndvi_q90_range_mean | encoded-discriminability | 0.6196 | 0.0000 | 0.1992 | 0.0000 | 0.7361 | 0.2014 |
| raw_source_phase_separability_p5 | raw-phase | 0.0000 | 0.6561 | 0.0000 | 0.0000 | 0.0000 | 0.1968 |
| raw_source_phase_margin_p1 | raw-phase | 0.0000 | 0.6368 | 0.0000 | 0.0000 | 0.0000 | 0.1910 |
| raw_source_phase_boundary_1 | raw-phase | 0.0000 | 0.6008 | 0.0000 | 0.0000 | 0.0000 | 0.1802 |
| raw_source_phase_boundary_2 | raw-phase | 0.0000 | 0.5464 | 0.0000 | 0.0000 | 0.0000 | 0.1639 |
| raw_source_phase_margin_p4 | raw-phase | 0.0000 | 0.4792 | 0.0000 | 0.0000 | 0.0000 | 0.1438 |
| raw_source_phase_margin_p2 | raw-phase | 0.0000 | 0.4352 | 0.0000 | 0.0000 | 0.0000 | 0.1306 |
| raw_source_ndvi_q90_range_std | raw-shape | 0.3689 | 0.0000 | 0.1055 | 0.0000 | 0.8689 | 0.1100 |

## Family summary

| family | top_metric | top_score | avg_score | metric_count |
|---|---:|---:|---:|---:|
| raw-shape | raw_source_curve_spread | 0.2137 | 0.1161 | 6 |
| raw-phase | raw_source_phase_separability_p1 | 0.2071 | 0.1101 | 19 |
| encoded-discriminability | source_ndvi_q90_range_mean | 0.2014 | 0.0946 | 6 |
| encoded-phase | source_phase_compactness_p4 | 0.1098 | 0.0388 | 19 |
