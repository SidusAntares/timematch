#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v243b_exact_history_4tasks}"
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
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

pids=()
names=()

run_one() {
  local gpu_id="$1"
  local label="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local trend="$5"
  local segment_inter="$6"
  local boundary="$7"
  local tag="$8"

  local log_file="$LOG_DIR/gpu${gpu_id}_${label}_${tag}.log"
  names+=("$label")

  echo "START|$label|gpu=$gpu_id|trend=$trend|segment_inter=$segment_inter|boundary=$boundary|log=$log_file"
  (
    export RESHAPER_TAG="$tag"
    export SOURCE_STRUCTURE_TREND_TRADE_OFF="$trend"
    export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="$segment_inter"
    export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="$boundary"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$target_dataset" \
      bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_source_block.sh"
  ) > "$log_file" 2>&1 &
  pids+=("$!")
}

# Historical v2.4.3b key tasks. Keep source epoch at 50 to match the strong quickcheck regime.
run_one 0 "FR1_to_AT1" "france/30TXT/2017" "austria/33UVP/2017" "0.03" "0.01" "0.10" "v243b_FR1_to_AT1_t003_si001_b010"
run_one 1 "FR2_to_DK1" "france/31TCJ/2017" "denmark/32VNH/2017" "0.05" "0.02" "0.20" "v243b_FR2_to_DK1_t005_si002_b020"
run_one 2 "DK1_to_FR2" "denmark/32VNH/2017" "france/31TCJ/2017" "0.05" "0.02" "0.20" "v243b_DK1_to_FR2_t005_si002_b020"
run_one 3 "AT1_to_DK1" "austria/33UVP/2017" "denmark/32VNH/2017" "0.05" "0.02" "0.20" "v243b_AT1_to_DK1_t005_si002_b020"

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "DONE|${names[$idx]}"
  else
    status="$?"
    echo "FAIL|${names[$idx]}|status=$status"
    failed=1
  fi
done

echo "Logs saved to: $LOG_DIR"
exit "$failed"
