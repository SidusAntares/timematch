#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v233_profiled_quickcheck_v222}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
export DEVICE="${DEVICE:-cuda}"

export RESHAPER_TAG="${RESHAPER_TAG:-v233_profiled_s010_rel003}"
export SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
export SOURCE_FEATURE_RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

export SOURCE_PHASE_PARTITION_MODE="${SOURCE_PHASE_PARTITION_MODE:-doy_gap}"
export SOURCE_PHASE_COUNT="${SOURCE_PHASE_COUNT:-5}"
export SOURCE_PHASE_GAP_THRESHOLD="${SOURCE_PHASE_GAP_THRESHOLD:-45}"
export SOURCE_PHASE_MIN_POINTS="${SOURCE_PHASE_MIN_POINTS:-3}"
export SOURCE_PHASE_MAX_POINTS="${SOURCE_PHASE_MAX_POINTS:-8}"
export SOURCE_PHASE_MAX_SPAN="${SOURCE_PHASE_MAX_SPAN:-120}"
export SOURCE_PHASE_MIN_SAMPLE_POINTS="${SOURCE_PHASE_MIN_SAMPLE_POINTS:-2}"

export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-profiled_components}"
export SOURCE_STRUCTURE_INTRA_TRADE_OFF="${SOURCE_STRUCTURE_INTRA_TRADE_OFF:-1.0}"
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.12}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-80}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"

run_one() {
  local gpu_id="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local source_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  local log_file="$LOG_DIR/gpu${gpu_id}_${source_tile}_${RUN_TAG}.log"

  (
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$target_dataset" \
      bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_source_block.sh"
  ) > "$log_file" 2>&1 &
}

# Representative quick-check tasks:
# - FR1 -> AT1: sensitive to phase/loss changes and often reveals whether FR1 over-regularizes
# - FR2 -> DK1: hardest failure point in v2.3.1/v2.3.2
# - DK1 -> FR2: reverse hard case, exposes whether weak-source handling improves
# - AT1 -> DK1: relatively strong task, checks whether gains survive on a good source
run_one 0 "france/30TXT/2017" "austria/33UVP/2017"
run_one 1 "france/31TCJ/2017" "denmark/32VNH/2017"
run_one 2 "denmark/32VNH/2017" "france/31TCJ/2017"
run_one 3 "austria/33UVP/2017" "denmark/32VNH/2017"

wait

echo "Logs saved to: $LOG_DIR"
echo "Quick-check training logs:"
echo "  $LOG_DIR/gpu0_30TXT_${RUN_TAG}.log"
echo "  $LOG_DIR/gpu1_31TCJ_${RUN_TAG}.log"
echo "  $LOG_DIR/gpu2_32VNH_${RUN_TAG}.log"
echo "  $LOG_DIR/gpu3_33UVP_${RUN_TAG}.log"
