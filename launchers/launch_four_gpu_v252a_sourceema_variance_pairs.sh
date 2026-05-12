#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v252a_sourceema_variance_pairs}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
export DEVICE="${DEVICE:-cuda}"

export SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
export SOURCE_FEATURE_RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

export SOURCE_PHASE_PARTITION_MODE="${SOURCE_PHASE_PARTITION_MODE:-doy_gap}"
export SOURCE_PHASE_COUNT="${SOURCE_PHASE_COUNT:-5}"
export SOURCE_SEGMENT_PARTITION_MODE="${SOURCE_SEGMENT_PARTITION_MODE:-$SOURCE_PHASE_PARTITION_MODE}"
export SOURCE_SEGMENT_COUNT="${SOURCE_SEGMENT_COUNT:-$SOURCE_PHASE_COUNT}"
export SOURCE_PHASE_GAP_THRESHOLD="${SOURCE_PHASE_GAP_THRESHOLD:-45}"
export SOURCE_PHASE_MIN_POINTS="${SOURCE_PHASE_MIN_POINTS:-3}"
export SOURCE_PHASE_MAX_POINTS="${SOURCE_PHASE_MAX_POINTS:-8}"
export SOURCE_PHASE_MAX_SPAN="${SOURCE_PHASE_MAX_SPAN:-120}"
export SOURCE_PHASE_MIN_SAMPLE_POINTS="${SOURCE_PHASE_MIN_SAMPLE_POINTS:-2}"

export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
export SOURCE_STRUCTURE_INTRA_TRADE_OFF="${SOURCE_STRUCTURE_INTRA_TRADE_OFF:-1.0}"
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.05}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.02}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.20}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

export SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-30,50,70,100}"
export SOURCE_CHECKPOINT_DIRNAME="${SOURCE_CHECKPOINT_DIRNAME:-checkpoints}"
export SOURCE_CHECKPOINT_EMA="${SOURCE_CHECKPOINT_EMA:-True}"
export SOURCE_CHECKPOINT_EMA_DECAY="${SOURCE_CHECKPOINT_EMA_DECAY:-0.999}"
export SOURCE_CHECKPOINT_EMA_SUFFIX="${SOURCE_CHECKPOINT_EMA_SUFFIX:-_ema}"
export SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-checkpoints/epoch_30.pt,checkpoints/epoch_30_ema.pt,checkpoints/epoch_50.pt,checkpoints/epoch_50_ema.pt,checkpoints/epoch_70.pt,checkpoints/epoch_70_ema.pt,checkpoints/epoch_100.pt,checkpoints/epoch_100_ema.pt}"

run_one() {
  local gpu_id="$1"
  local run_suffix="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local source_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  local target_tile
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  local reshaper_tag="v252a_boundarywindow_s010_rel003_ckptbank_ema_${source_tile}_to_${target_tile}_${run_suffix}"
  local log_file="$LOG_DIR/gpu${gpu_id}_${source_tile}_to_${target_tile}_${run_suffix}.log"

  (
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      RUN_TAG="${RUN_TAG}_${run_suffix}" \
      RESHAPER_TAG="$reshaper_tag" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$target_dataset" \
      bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_source_block.sh"
  ) > "$log_file" 2>&1 &
}

# Pair A: same task, two independent source-only trainings
run_one 0 "runA" "france/31TCJ/2017" "denmark/32VNH/2017"
run_one 1 "runB" "france/31TCJ/2017" "denmark/32VNH/2017"

# Pair B: same task, two independent source-only trainings
run_one 2 "runA" "denmark/32VNH/2017" "france/31TCJ/2017"
run_one 3 "runB" "denmark/32VNH/2017" "france/31TCJ/2017"

wait

echo "Logs saved to: $LOG_DIR"
echo "Variance-pair logs:"
echo "  $LOG_DIR/gpu0_31TCJ_to_32VNH_runA.log"
echo "  $LOG_DIR/gpu1_31TCJ_to_32VNH_runB.log"
echo "  $LOG_DIR/gpu2_32VNH_to_31TCJ_runA.log"
echo "  $LOG_DIR/gpu3_32VNH_to_31TCJ_runB.log"
