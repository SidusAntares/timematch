#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
DEVICE="${DEVICE:-cuda}"
PHASE_PARTITION_MODE="${PHASE_PARTITION_MODE:-structure}"
PHASE_COUNT="${PHASE_COUNT:-5}"
MAX_FEATURE_SAMPLES="${MAX_FEATURE_SAMPLES:-2048}"
TEMPORAL_GRID_SIZE="${TEMPORAL_GRID_SIZE:-30}"
MAX_ACF_LAG="${MAX_ACF_LAG:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-111}"

cd "$ROOT_DIR"

python analysis/recompute_transfer_metrics.py \
  --data_root "$DATA_ROOT" \
  --outputs_root "$OUTPUTS_ROOT" \
  --output_csv "result/v2_analysis/transfer_metrics_v2_sourcephasecompact.csv" \
  --phase_output_csv "result/v2_analysis/source_phase_metrics_v2_sourcephasecompact.csv" \
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
  --experiment_suffix "_sourcephasecompact_p5"
