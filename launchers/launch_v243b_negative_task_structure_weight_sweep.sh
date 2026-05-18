#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BLOCK_SCRIPT="$ROOT_DIR/launchers/ideas/run_timematch_closed_set_sourcephasecompact_v255_allcheckpoints_block.sh"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v243b_negative_task_structure_weight_sweep_${BATCH_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"

# v2.4.3b backbone: keep the original reshaper and boundary-window residual structure loss.
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

# Explicitly disable mechanisms after v2.4.3b.
export SOURCE_PHASE_GRID_TRADE_OFF="${SOURCE_PHASE_GRID_TRADE_OFF:-0.0}"
export SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS="${SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS:-False}"
export SOURCE_STRUCTURE_ADAPTIVITY_MODE="${SOURCE_STRUCTURE_ADAPTIVITY_MODE:-none}"
export SOURCE_SKIP_TRAIN="${SOURCE_SKIP_TRAIN:-0}"
export SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-}"
export SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-model.pt}"

# Each item: name:trend:segment_inter:boundary_window
VARIANT_SPECS="${VARIANT_SPECS:-base:0.05:0.02:0.20,trend_light:0.02:0.02:0.20,transition_light:0.05:0.01:0.20,boundary_light:0.05:0.02:0.10,structure_light:0.03:0.01:0.10,no_trend_light:0.00:0.01:0.10,boundary_strong_inter_light:0.03:0.01:0.30}"

# Task aliases:
# FR1=france/30TXT, FR2=france/31TCJ, DK1=denmark/32VNH, AT1=austria/33UVP.
TASK_SPECS="${TASK_SPECS:-FR1_to_FR2|france/30TXT/2017|france/31TCJ/2017,FR1_to_DK1|france/30TXT/2017|denmark/32VNH/2017,FR1_to_AT1|france/30TXT/2017|austria/33UVP/2017,AT1_to_FR1|austria/33UVP/2017|france/30TXT/2017,AT1_to_FR2|austria/33UVP/2017|france/31TCJ/2017,AT1_to_DK1|austria/33UVP/2017|denmark/32VNH/2017}"

IFS=',' read -r -a GPU_ITEMS <<< "$GPU_IDS"
IFS=',' read -r -a VARIANT_ITEMS <<< "$VARIANT_SPECS"
IFS=',' read -r -a TASK_ITEMS <<< "$TASK_SPECS"

job_count=0
gpu_cursor=0

run_one() {
  local variant_name="$1"
  local trend_weight="$2"
  local segment_inter_weight="$3"
  local boundary_weight="$4"
  local task_name="$5"
  local source_dataset="$6"
  local target_dataset="$7"

  local gpu_id="${GPU_ITEMS[$((gpu_cursor % ${#GPU_ITEMS[@]}))]}"
  gpu_cursor=$((gpu_cursor + 1))

  local source_tile
  local target_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"

  local job_tag="v243b_${variant_name}_${task_name}_${BATCH_STAMP}"
  local log_file="$LOG_DIR/${job_tag}.log"

  echo "START|${job_tag}|gpu=${gpu_id}|trend=${trend_weight}|segment_inter=${segment_inter_weight}|boundary=${boundary_weight}"
  (
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$target_dataset" \
      VARIANT_NAME="$variant_name" \
      SOURCE_STRUCTURE_TREND_TRADE_OFF="$trend_weight" \
      SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="$segment_inter_weight" \
      SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="$boundary_weight" \
      SOURCE_MODEL_TAG="${job_tag}_${source_tile}_to_${target_tile}_source" \
      RESHAPER_TAG="${job_tag}_${source_tile}_to_${target_tile}" \
      TARGET_MODEL_TAG="${job_tag}_${source_tile}_to_${target_tile}" \
      bash "$BLOCK_SCRIPT"
  ) > "$log_file" 2>&1 &

  job_count=$((job_count + 1))
  if [ $((job_count % MAX_PARALLEL)) -eq 0 ]; then
    wait
  fi
}

echo "Logs saved under: $LOG_DIR"
echo "SOURCE_PRETRAIN_EPOCHS=$SOURCE_PRETRAIN_EPOCHS TIMEMATCH_EPOCHS=$TIMEMATCH_EPOCHS"
echo "VARIANT_SPECS=$VARIANT_SPECS"
echo "TASK_SPECS=$TASK_SPECS"

for variant_spec in "${VARIANT_ITEMS[@]}"; do
  IFS=':' read -r variant_name trend_weight segment_inter_weight boundary_weight <<< "$variant_spec"
  for task_spec in "${TASK_ITEMS[@]}"; do
    IFS='|' read -r task_name source_dataset target_dataset <<< "$task_spec"
    run_one "$variant_name" "$trend_weight" "$segment_inter_weight" "$boundary_weight" "$task_name" "$source_dataset" "$target_dataset"
  done
done

wait

echo "All v2.4.3b structure-weight sweep jobs finished."
echo "Logs saved under: $LOG_DIR"
