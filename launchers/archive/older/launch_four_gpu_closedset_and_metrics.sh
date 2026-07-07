#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs/closedset_and_metrics_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

echo "[INFO] Logs will be written to: $LOG_DIR"

nohup env CUDA_VISIBLE_DEVICES=0 PARTITION=A DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  bash "$SCRIPT_DIR/closed_set/run_timematch_closed_set_partition.sh" \
  > "$LOG_DIR/gpu0_closedset_partition_A.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 PARTITION=B DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  bash "$SCRIPT_DIR/closed_set/run_timematch_closed_set_partition.sh" \
  > "$LOG_DIR/gpu1_closedset_partition_B.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  bash "$SCRIPT_DIR/analysis/run_recompute_metrics_open.sh" \
  > "$LOG_DIR/gpu2_recompute_open_metrics.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  bash "$SCRIPT_DIR/analysis/run_recompute_metrics_closed.sh" \
  > "$LOG_DIR/gpu3_recompute_closed_metrics.log" 2>&1 &

echo "[INFO] Submitted four background jobs."
echo "[INFO] Tail logs with:"
echo "  tail -f $LOG_DIR/gpu0_closedset_partition_A.log"
echo "  tail -f $LOG_DIR/gpu1_closedset_partition_B.log"
echo "  tail -f $LOG_DIR/gpu2_recompute_open_metrics.log"
echo "  tail -f $LOG_DIR/gpu3_recompute_closed_metrics.log"
