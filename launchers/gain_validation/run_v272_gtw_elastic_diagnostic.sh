#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GTW_RUN_TAG="${GTW_RUN_TAG:-gtw_full_20260526_225343}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/${GTW_RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/${GTW_RUN_TAG}}"
DIAG_ROOT="${DIAG_ROOT:-$ROOT_DIR/outputs/v272_gtw_elastic_diag_${STAMP}}"
TASKS="${TASKS:-FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$DIAG_ROOT"

python analysis/v272_gtw_elastic_diagnostic.py \
  --data_root "${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --tasks "$TASKS" \
  --checkpoint_mode "${CHECKPOINT_MODE:-G_global}" \
  --band_ratios "${BAND_RATIOS:-0.10,0.30,0.50}" \
  --max_source_batches "${MAX_SOURCE_BATCHES:-24}" \
  --max_target_batches "${MAX_TARGET_BATCHES:-24}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --device "$DEVICE" \
  --out_csv "$DIAG_ROOT/elastic_diagnostic.csv"

echo "GTW elastic diagnostic finished."
echo "CSV: $DIAG_ROOT/elastic_diagnostic.csv"
