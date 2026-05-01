#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-$ROOT_DIR/analysis/recompute_transfer_metrics.py}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
PHASE_OUTPUT_CSV="${PHASE_OUTPUT_CSV:-result/baseline_analysis/source_phase_self_structure_metrics_closed.csv}"
PHASE_PARTITION_MODE="${PHASE_PARTITION_MODE:-structure}"
PHASE_COUNT="${PHASE_COUNT:-3}"

cd "$ROOT_DIR"

if [[ ! -f "$ANALYSIS_SCRIPT" ]]; then
  echo "[ERROR] Missing analysis script: $ANALYSIS_SCRIPT"
  exit 1
fi

python "$ANALYSIS_SCRIPT" \
  --data_root "$DATA_ROOT" \
  --outputs_root "$OUTPUTS_ROOT" \
  --phase_output_csv "$PHASE_OUTPUT_CSV" \
  --closed_set True \
  --phase_only True \
  --phase_partition_mode "$PHASE_PARTITION_MODE" \
  --phase_count "$PHASE_COUNT"
