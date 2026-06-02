#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v243b_paired_lift_2tasks_5seeds}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
SEEDS="${SEEDS:-1 2 3 4 5}"

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
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

pids=()
labels=()
failed=0
job_index=0

wait_batch() {
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "DONE|${labels[$i]}"
    else
      local status="$?"
      echo "FAIL|${labels[$i]}|status=$status"
      failed=1
    fi
  done
  pids=()
  labels=()
}

run_one() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local setting="$5"
  local intra="$6"
  local trend="$7"
  local segment_inter="$8"
  local boundary="$9"
  local weight_tag="${10}"
  local gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  local tag="v243b_pair_${task}_seed${seed}_${setting}_${weight_tag}"
  local label="${task}|seed=${seed}|${setting}|gpu=${gpu}|${weight_tag}"
  local log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${setting}_${weight_tag}.log"

  echo "START|$label|log=$log_file"
  (
    export SEED="$seed"
    export RESHAPER_TAG="$tag"
    export SOURCE_STRUCTURE_INTRA_TRADE_OFF="$intra"
    export SOURCE_STRUCTURE_TREND_TRADE_OFF="$trend"
    export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="$segment_inter"
    export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="$boundary"
    CUDA_VISIBLE_DEVICES="$gpu" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$target_dataset" \
      bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_source_block.sh"
  ) > "$log_file" 2>&1 &

  pids+=("$!")
  labels+=("$label")
  job_index=$((job_index + 1))

  if [ "${#pids[@]}" -ge "${#GPU_IDS[@]}" ]; then
    wait_batch
  fi
}

for seed in $SEEDS; do
  # FR1 -> AT1: historical v2.4.3b best used a milder structural coupling.
  run_one "FR1_to_AT1" "france/30TXT/2017" "austria/33UVP/2017" "$seed" "off" "0.0" "0.00" "0.00" "0.00" "t000_si000_b000"
  run_one "FR1_to_AT1" "france/30TXT/2017" "austria/33UVP/2017" "$seed" "on"  "1.0" "0.03" "0.01" "0.10" "t003_si001_b010"

  # AT1 -> DK1: stable positive v2.4.3b signal in recent pure-code quickchecks.
  run_one "AT1_to_DK1" "austria/33UVP/2017" "denmark/32VNH/2017" "$seed" "off" "0.0" "0.00" "0.00" "0.00" "t000_si000_b000"
  run_one "AT1_to_DK1" "austria/33UVP/2017" "denmark/32VNH/2017" "$seed" "on"  "1.0" "0.05" "0.02" "0.20" "t005_si002_b020"
done

if [ "${#pids[@]}" -gt 0 ]; then
  wait_batch
fi

echo "Logs saved to: $LOG_DIR"
python "$ROOT_DIR/analysis/summarize_v243b_paired_lift.py" "$LOG_DIR" || true

exit "$failed"
