#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-111}"
PHASE_PARTITION_MODE="${PHASE_PARTITION_MODE:-structure}"
PHASE_COUNT="${PHASE_COUNT:-5}"
MAX_FEATURE_SAMPLES="${MAX_FEATURE_SAMPLES:-2048}"
TEMPORAL_GRID_SIZE="${TEMPORAL_GRID_SIZE:-30}"
MAX_ACF_LAG="${MAX_ACF_LAG:-10}"

BASELINE_DIR="result/baseline_analysis"
V222_DIR="result/v222_analysis"

cd "$ROOT_DIR"

for path in \
  "$BASELINE_DIR/transfer_correlation_summary/overall_metric_correlations.csv" \
  "$BASELINE_DIR/transfer_correlation_summary/source_correlation_spread.csv" \
  "$BASELINE_DIR/phase_correlation_summary/overall_metric_correlations.csv" \
  "$BASELINE_DIR/phase_correlation_summary/source_correlation_spread.csv" \
  "$BASELINE_DIR/per_class_correlation_summary/overall_metric_correlations.csv" \
  "$BASELINE_DIR/per_class_correlation_summary/class_correlation_spread.csv"; do
  if [ ! -f "$path" ]; then
    echo "[ERROR] Missing baseline prerequisite: $path" >&2
    exit 1
  fi
done

python analysis/recompute_transfer_metrics.py \
  --data_root "$DATA_ROOT" \
  --outputs_root "$OUTPUTS_ROOT" \
  --output_csv "$V222_DIR/transfer_metrics_v222.csv" \
  --phase_output_csv "$V222_DIR/phase_metrics_v222.csv" \
  --closed_set True \
  --device "$DEVICE" \
  --phase_partition_mode "$PHASE_PARTITION_MODE" \
  --phase_count "$PHASE_COUNT" \
  --max_feature_samples "$MAX_FEATURE_SAMPLES" \
  --temporal_grid_size "$TEMPORAL_GRID_SIZE" \
  --max_acf_lag "$MAX_ACF_LAG" \
  --num_workers "$NUM_WORKERS" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --experiment_suffix "_sourcephasecompact_p5_v222_s010_rel003"

python analysis/summarize_metric_correlations.py \
  --input_csv "$V222_DIR/transfer_metrics_v222.csv" \
  --output_dir "$V222_DIR/transfer_correlation_summary"

python analysis/summarize_metric_correlations.py \
  --input_csv "$V222_DIR/phase_metrics_v222.csv" \
  --output_dir "$V222_DIR/phase_correlation_summary"

python analysis/summarize_per_class_correlations.py \
  --metrics_csv "$V222_DIR/transfer_metrics_v222.csv" \
  --outputs_root "$OUTPUTS_ROOT" \
  --output_dir "$V222_DIR/per_class_correlation_summary"

python analysis/summarize_encoded_feature_dimensions.py \
  --transfer_overall_csv "$BASELINE_DIR/transfer_correlation_summary/overall_metric_correlations.csv" \
  --transfer_source_spread_csv "$BASELINE_DIR/transfer_correlation_summary/source_correlation_spread.csv" \
  --phase_overall_csv "$BASELINE_DIR/phase_correlation_summary/overall_metric_correlations.csv" \
  --phase_source_spread_csv "$BASELINE_DIR/phase_correlation_summary/source_correlation_spread.csv" \
  --per_class_overall_csv "$BASELINE_DIR/per_class_correlation_summary/overall_metric_correlations.csv" \
  --per_class_spread_csv "$BASELINE_DIR/per_class_correlation_summary/class_correlation_spread.csv" \
  --output_dir "$BASELINE_DIR/encoded_feature_dimension_summary"

python analysis/summarize_encoded_feature_dimensions.py \
  --transfer_overall_csv "$V222_DIR/transfer_correlation_summary/overall_metric_correlations.csv" \
  --transfer_source_spread_csv "$V222_DIR/transfer_correlation_summary/source_correlation_spread.csv" \
  --phase_overall_csv "$V222_DIR/phase_correlation_summary/overall_metric_correlations.csv" \
  --phase_source_spread_csv "$V222_DIR/phase_correlation_summary/source_correlation_spread.csv" \
  --per_class_overall_csv "$V222_DIR/per_class_correlation_summary/overall_metric_correlations.csv" \
  --per_class_spread_csv "$V222_DIR/per_class_correlation_summary/class_correlation_spread.csv" \
  --output_dir "$V222_DIR/encoded_feature_dimension_summary"
