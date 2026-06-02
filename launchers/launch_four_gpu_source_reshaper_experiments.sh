#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs/source_reshaper_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

echo "[INFO] Logs will be written to: $LOG_DIR"

RESHAPER_A="${RESHAPER_A:-residual_temporal_conv}"
STRENGTH_A="${STRENGTH_A:-0.05}"
KERNEL_A="${KERNEL_A:-3}"
REG_A="${REG_A:-0.05}"
TAG_A="${TAG_A:-${RESHAPER_A}_s${STRENGTH_A}_k${KERNEL_A}_r${REG_A}}"

RESHAPER_B="${RESHAPER_B:-residual_temporal_conv}"
STRENGTH_B="${STRENGTH_B:-0.10}"
KERNEL_B="${KERNEL_B:-3}"
REG_B="${REG_B:-0.05}"
TAG_B="${TAG_B:-${RESHAPER_B}_s${STRENGTH_B}_k${KERNEL_B}_r${REG_B}}"

nohup env CUDA_VISIBLE_DEVICES=0 PARTITION=A DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_FEATURE_RESHAPER="$RESHAPER_A" SOURCE_FEATURE_RESHAPER_STRENGTH="$STRENGTH_A" SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="$KERNEL_A" SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$REG_A" RESHAPER_TAG="$TAG_A" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_reshaper_partition.sh" \
  > "$LOG_DIR/gpu0_${TAG_A}_partition_A.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 PARTITION=B DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_FEATURE_RESHAPER="$RESHAPER_A" SOURCE_FEATURE_RESHAPER_STRENGTH="$STRENGTH_A" SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="$KERNEL_A" SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$REG_A" RESHAPER_TAG="$TAG_A" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_reshaper_partition.sh" \
  > "$LOG_DIR/gpu1_${TAG_A}_partition_B.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 PARTITION=A DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_FEATURE_RESHAPER="$RESHAPER_B" SOURCE_FEATURE_RESHAPER_STRENGTH="$STRENGTH_B" SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="$KERNEL_B" SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$REG_B" RESHAPER_TAG="$TAG_B" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_reshaper_partition.sh" \
  > "$LOG_DIR/gpu2_${TAG_B}_partition_A.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 PARTITION=B DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_FEATURE_RESHAPER="$RESHAPER_B" SOURCE_FEATURE_RESHAPER_STRENGTH="$STRENGTH_B" SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="$KERNEL_B" SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$REG_B" RESHAPER_TAG="$TAG_B" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_reshaper_partition.sh" \
  > "$LOG_DIR/gpu3_${TAG_B}_partition_B.log" 2>&1 &

echo "[INFO] Submitted four background jobs."
echo "[INFO] Condition A: $TAG_A"
echo "[INFO] Condition B: $TAG_B"
echo "[INFO] Tail logs with:"
echo "  tail -f $LOG_DIR/gpu0_${TAG_A}_partition_A.log"
echo "  tail -f $LOG_DIR/gpu1_${TAG_A}_partition_B.log"
echo "  tail -f $LOG_DIR/gpu2_${TAG_B}_partition_A.log"
echo "  tail -f $LOG_DIR/gpu3_${TAG_B}_partition_B.log"
