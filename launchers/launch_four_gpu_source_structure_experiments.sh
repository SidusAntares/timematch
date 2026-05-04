#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs/source_structure_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

echo "[INFO] Logs will be written to: $LOG_DIR"

TRANSFORM_A="${TRANSFORM_A:-phase_center_blend}"
STRENGTH_A="${STRENGTH_A:-0.15}"
PHASE_COUNT_A="${PHASE_COUNT_A:-5}"
TAG_A="${TAG_A:-${TRANSFORM_A}_s${STRENGTH_A}_p${PHASE_COUNT_A}}"

TRANSFORM_B="${TRANSFORM_B:-middle_phase_deviation_boost}"
STRENGTH_B="${STRENGTH_B:-0.20}"
PHASE_COUNT_B="${PHASE_COUNT_B:-5}"
TAG_B="${TAG_B:-${TRANSFORM_B}_s${STRENGTH_B}_p${PHASE_COUNT_B}}"

nohup env CUDA_VISIBLE_DEVICES=0 PARTITION=A DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_STRUCTURE_TRANSFORM="$TRANSFORM_A" SOURCE_STRUCTURE_STRENGTH="$STRENGTH_A" SOURCE_STRUCTURE_PHASE_COUNT="$PHASE_COUNT_A" STRUCTURE_TAG="$TAG_A" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_structured_partition.sh" \
  > "$LOG_DIR/gpu0_${TAG_A}_partition_A.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 PARTITION=B DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_STRUCTURE_TRANSFORM="$TRANSFORM_A" SOURCE_STRUCTURE_STRENGTH="$STRENGTH_A" SOURCE_STRUCTURE_PHASE_COUNT="$PHASE_COUNT_A" STRUCTURE_TAG="$TAG_A" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_structured_partition.sh" \
  > "$LOG_DIR/gpu1_${TAG_A}_partition_B.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 PARTITION=A DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_STRUCTURE_TRANSFORM="$TRANSFORM_B" SOURCE_STRUCTURE_STRENGTH="$STRENGTH_B" SOURCE_STRUCTURE_PHASE_COUNT="$PHASE_COUNT_B" STRUCTURE_TAG="$TAG_B" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_structured_partition.sh" \
  > "$LOG_DIR/gpu2_${TAG_B}_partition_A.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 PARTITION=B DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  SOURCE_STRUCTURE_TRANSFORM="$TRANSFORM_B" SOURCE_STRUCTURE_STRENGTH="$STRENGTH_B" SOURCE_STRUCTURE_PHASE_COUNT="$PHASE_COUNT_B" STRUCTURE_TAG="$TAG_B" \
  bash "$SCRIPT_DIR/ideas/run_timematch_closed_set_sourcephasecompact_structured_partition.sh" \
  > "$LOG_DIR/gpu3_${TAG_B}_partition_B.log" 2>&1 &

echo "[INFO] Submitted four background jobs."
echo "[INFO] Condition A: $TAG_A"
echo "[INFO] Condition B: $TAG_B"
echo "[INFO] Tail logs with:"
echo "  tail -f $LOG_DIR/gpu0_${TAG_A}_partition_A.log"
echo "  tail -f $LOG_DIR/gpu1_${TAG_A}_partition_B.log"
echo "  tail -f $LOG_DIR/gpu2_${TAG_B}_partition_A.log"
echo "  tail -f $LOG_DIR/gpu3_${TAG_B}_partition_B.log"
