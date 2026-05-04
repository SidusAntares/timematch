#!/bin/bash
set -euo pipefail

python analysis/screen_source_metrics.py \
  --transfer_overall_csv result/baseline_analysis/transfer_correlation_summary/overall_metric_correlations.csv \
  --transfer_source_spread_csv result/baseline_analysis/transfer_correlation_summary/source_correlation_spread.csv \
  --phase_overall_csv result/baseline_analysis/phase_correlation_summary/overall_metric_correlations.csv \
  --phase_source_spread_csv result/baseline_analysis/phase_correlation_summary/source_correlation_spread.csv \
  --per_class_overall_csv result/baseline_analysis/per_class_correlation_summary/overall_metric_correlations.csv \
  --per_class_spread_csv result/baseline_analysis/per_class_correlation_summary/class_correlation_spread.csv \
  --output_dir result/baseline_analysis/source_metric_screening
