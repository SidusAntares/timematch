#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v243b_remote_sanity_4gpu_${BATCH_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/${RUN_TAG}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_SOURCE_ON_TARGET="${EVAL_SOURCE_ON_TARGET:-true}"

RESHAPER_KIND="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
DUAL_REL_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

mkdir -p "$LOG_DIR" "$OUT_ROOT" "$RUN_ROOT"
IFS=',' read -r -a GPU_ITEMS <<< "$GPU_IDS"

common_args() {
  local source_dataset="$1"
  local target_dataset="$2"
  local trend_weight="$3"
  local segment_inter_weight="$4"
  local boundary_weight="$5"

  COMMON_ARGS=(
    --data_root "$DATA_ROOT"
    --closed_set True
    --with_shift_aug False
    --source_feature_reshaper "$RESHAPER_KIND"
    --source_feature_reshaper_strength "$RESHAPER_STRENGTH"
    --source_feature_reshaper_kernel_size "$RESHAPER_KERNEL_SIZE"
    --source_feature_reshaper_reg_trade_off "$RESHAPER_REG_TRADE_OFF"
    --source_feature_dual_path True
    --source_feature_dual_cls_trade_off "$DUAL_CLS_TRADE_OFF"
    --source_feature_dual_relation_trade_off "$DUAL_REL_TRADE_OFF"
    --source_phase_partition_mode doy_gap
    --source_segment_partition_mode doy_gap
    --source_phase_count 5
    --source_segment_count 5
    --source_phase_gap_threshold 45
    --source_phase_min_points 3
    --source_phase_max_points 8
    --source_phase_max_span 120
    --source_phase_min_sample_points 2
    --source_structure_loss_version segment_boundary_window_residual
    --source_structure_intra_trade_off 1.0
    --source_structure_amplitude_trade_off 0.0
    --source_structure_interphase_trade_off 0.0
    --source_structure_shape_trade_off 0.0
    --source_structure_trend_trade_off "$trend_weight"
    --source_structure_season_trade_off 0.0
    --source_structure_segment_inter_trade_off "$segment_inter_weight"
    --source_structure_boundary_window_trade_off "$boundary_weight"
    --source_structure_boundary_window_size 2
    --num_workers "$NUM_WORKERS"
    --batch_size "$BATCH_SIZE"
    --source "$source_dataset"
    --target "$target_dataset"
  )
}

run_one() {
  local gpu_index="$1"
  local task_name="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local trend_weight="$5"
  local segment_inter_weight="$6"
  local boundary_weight="$7"

  local gpu_id="${GPU_ITEMS[$gpu_index]}"
  local variant_name="v243b_${task_name}_t${trend_weight}_si${segment_inter_weight}_b${boundary_weight}"
  variant_name="${variant_name//./}"
  local source_tile
  local target_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  local source_exp="${variant_name}_${source_tile}_source_${BATCH_STAMP}"
  local da_exp="${variant_name}_${source_tile}_to_${target_tile}_timematch_${BATCH_STAMP}"
  local source_out="${OUT_ROOT}/${source_exp}"
  local log_file="$LOG_DIR/${variant_name}_${source_tile}_to_${target_tile}.log"

  echo "START|${task_name}|gpu=${gpu_id}|trend=${trend_weight}|segment_inter=${segment_inter_weight}|boundary=${boundary_weight}|log=${log_file}"
  (
    common_args "$source_dataset" "$source_dataset" "$trend_weight" "$segment_inter_weight" "$boundary_weight"
    echo "V243B_SOURCE_START|${task_name}|${source_exp}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
      "${COMMON_ARGS[@]}" \
      --output_dir "$OUT_ROOT" \
      --tensorboard_log_dir "$RUN_ROOT" \
      --epochs "$SOURCE_PRETRAIN_EPOCHS" \
      -e "$source_exp" \
      sourcephasecompact

    if [ "$EVAL_SOURCE_ON_TARGET" = "true" ]; then
      common_args "$source_dataset" "$target_dataset" "$trend_weight" "$segment_inter_weight" "$boundary_weight"
      echo "V243B_SOURCE_TARGET_EVAL|${task_name}|${source_exp}|target=${target_dataset}"
      CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
        "${COMMON_ARGS[@]}" \
        --output_dir "$OUT_ROOT" \
        --tensorboard_log_dir "$RUN_ROOT" \
        -e "$source_exp" \
        --eval
    fi

    common_args "$source_dataset" "$target_dataset" "$trend_weight" "$segment_inter_weight" "$boundary_weight"
    echo "V243B_DA_START|${task_name}|${da_exp}|weights=${source_out}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
      "${COMMON_ARGS[@]}" \
      --output_dir "$OUT_ROOT" \
      --tensorboard_log_dir "$RUN_ROOT" \
      -e "$da_exp" \
      timematch \
      --weights "$source_out" \
      --epochs "$TIMEMATCH_EPOCHS"
    echo "DONE|${task_name}|${da_exp}"
  ) > "$log_file" 2>&1 &
}

if [ "${#GPU_ITEMS[@]}" -lt 4 ]; then
  echo "Expected four GPU ids in GPU_IDS, got: $GPU_IDS" >&2
  exit 1
fi

run_one 0 "FR1_to_AT1" "france/30TXT/2017" "austria/33UVP/2017" "0.03" "0.01" "0.10"
run_one 1 "FR2_to_DK1" "france/31TCJ/2017" "denmark/32VNH/2017" "0.05" "0.02" "0.20"
run_one 2 "AT1_to_DK1" "austria/33UVP/2017" "denmark/32VNH/2017" "0.05" "0.02" "0.20"
run_one 3 "FR1_to_DK1" "france/30TXT/2017" "denmark/32VNH/2017" "0.02" "0.02" "0.20"

wait

echo "All v2.4.3b remote sanity jobs finished."
echo "Logs saved under: $LOG_DIR"
