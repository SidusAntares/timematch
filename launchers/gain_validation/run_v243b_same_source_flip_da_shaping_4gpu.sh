#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v243b_same_source_flip_da_${BATCH_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/${RUN_TAG}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
A_SOURCE_ROOT="${A_SOURCE_ROOT:-/data/user/timematch/outputs/v243b_mech_A_exact_20260525_212349}"
A_SOURCE_STAMP="${A_SOURCE_STAMP:-20260525_212349}"
BCD_SOURCE_ROOT="${BCD_SOURCE_ROOT:-/data/user/timematch/outputs/v243b_mech_BCD_20260525_224906}"
BCD_SOURCE_STAMP="${BCD_SOURCE_STAMP:-20260525_224906}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"

RESHAPER_KIND="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
DUAL_REL_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

mkdir -p "$LOG_DIR" "$OUT_ROOT" "$RUN_ROOT"
IFS=',' read -r -a GPU_ITEMS <<< "$GPU_IDS"

if [ "${#GPU_ITEMS[@]}" -lt 4 ]; then
  echo "Expected four GPU ids in GPU_IDS, got: $GPU_IDS" >&2
  exit 1
fi

task_spec() {
  local task_name="$1"
  case "$task_name" in
    FR1_to_AT1) echo "france/30TXT/2017 austria/33UVP/2017 0.03 0.01 0.10" ;;
    FR2_to_DK1) echo "france/31TCJ/2017 denmark/32VNH/2017 0.05 0.02 0.20" ;;
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017 0.05 0.02 0.20" ;;
    FR1_to_DK1) echo "france/30TXT/2017 denmark/32VNH/2017 0.02 0.02 0.20" ;;
    *) echo "Unknown task: $task_name" >&2; exit 1 ;;
  esac
}

source_spec() {
  local mode="$1"
  local task_name="$2"
  local source_tile="$3"
  local trend_weight="$4"
  local segment_inter_weight="$5"
  local boundary_weight="$6"

  local source_exp source_root source_stamp loss_version partition_mode phase_count da_structure_weight
  case "$mode" in
    A_daoff)
      source_root="$A_SOURCE_ROOT"
      source_stamp="$A_SOURCE_STAMP"
      source_exp="v243b_${task_name}_t${trend_weight}_si${segment_inter_weight}_b${boundary_weight}"
      source_exp="${source_exp//./}_${source_tile}_source_${source_stamp}"
      loss_version="segment_boundary_window_residual"
      partition_mode="doy_gap"
      phase_count="5"
      da_structure_weight="0.0"
      ;;
    B_daon)
      source_root="$BCD_SOURCE_ROOT"
      source_stamp="$BCD_SOURCE_STAMP"
      source_exp="v243b_B_no_da_shaping_${task_name}_t${trend_weight}_si${segment_inter_weight}_b${boundary_weight}_da00"
      source_exp="${source_exp//./}_${source_tile}_source_${source_stamp}"
      loss_version="segment_boundary_window_residual"
      partition_mode="doy_gap"
      phase_count="5"
      da_structure_weight="1.0"
      ;;
    C_daoff)
      source_root="$BCD_SOURCE_ROOT"
      source_stamp="$BCD_SOURCE_STAMP"
      source_exp="v243b_C_no_boundary_${task_name}_t${trend_weight}_si${segment_inter_weight}_b00_da10"
      source_exp="${source_exp//./}_${source_tile}_source_${source_stamp}"
      loss_version="segment_boundary_window_residual"
      partition_mode="doy_gap"
      phase_count="5"
      boundary_weight="0.0"
      da_structure_weight="0.0"
      ;;
    D_daoff)
      source_root="$BCD_SOURCE_ROOT"
      source_stamp="$BCD_SOURCE_STAMP"
      source_exp="v243b_D_global_like_${task_name}_t${trend_weight}_si00_b00_da10"
      source_exp="${source_exp//./}_${source_tile}_source_${source_stamp}"
      loss_version="trend_residual"
      partition_mode="uniform"
      phase_count="1"
      segment_inter_weight="0.0"
      boundary_weight="0.0"
      da_structure_weight="0.0"
      ;;
    *)
      echo "Unknown flip mode: $mode" >&2
      exit 1
      ;;
  esac

  echo "$source_root $source_exp $loss_version $partition_mode $phase_count $trend_weight $segment_inter_weight $boundary_weight $da_structure_weight"
}

common_args() {
  local source_dataset="$1"
  local target_dataset="$2"
  local loss_version="$3"
  local partition_mode="$4"
  local phase_count="$5"
  local trend_weight="$6"
  local segment_inter_weight="$7"
  local boundary_weight="$8"

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
    --source_phase_partition_mode "$partition_mode"
    --source_segment_partition_mode "$partition_mode"
    --source_phase_count "$phase_count"
    --source_segment_count "$phase_count"
    --source_phase_gap_threshold 45
    --source_phase_min_points 3
    --source_phase_max_points 8
    --source_phase_max_span 120
    --source_phase_min_sample_points 2
    --source_structure_loss_version "$loss_version"
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
  local mode="$2"
  local task_name="$3"
  local source_dataset target_dataset trend_weight segment_inter_weight boundary_weight
  read -r source_dataset target_dataset trend_weight segment_inter_weight boundary_weight <<< "$(task_spec "$task_name")"

  local source_tile target_tile source_root source_exp loss_version partition_mode phase_count da_trend da_segment_inter da_boundary da_structure_weight
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  read -r source_root source_exp loss_version partition_mode phase_count da_trend da_segment_inter da_boundary da_structure_weight \
    <<< "$(source_spec "$mode" "$task_name" "$source_tile" "$trend_weight" "$segment_inter_weight" "$boundary_weight")"

  local gpu_id="${GPU_ITEMS[$gpu_index]}"
  local weights_dir="${source_root}/${source_exp}"
  local da_exp="v243b_${mode}_${task_name}_${source_tile}_to_${target_tile}_timematch_${BATCH_STAMP}"
  local log_file="$LOG_DIR/${da_exp}.log"

  if [ ! -f "${weights_dir}/fold_0/model.pt" ]; then
    echo "Missing source checkpoint for ${mode}/${task_name}: ${weights_dir}/fold_0/model.pt" >&2
    exit 1
  fi

  echo "START|${mode}|${task_name}|gpu=${gpu_id}|weights=${weights_dir}|loss=${loss_version}|partition=${partition_mode}|da_structure=${da_structure_weight}|log=${log_file}"
  (
    common_args "$source_dataset" "$target_dataset" "$loss_version" "$partition_mode" "$phase_count" "$da_trend" "$da_segment_inter" "$da_boundary"
    echo "SAME_SOURCE_FLIP_DA_START|${mode}|${task_name}|${da_exp}|weights=${weights_dir}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
      "${COMMON_ARGS[@]}" \
      --output_dir "$OUT_ROOT" \
      --tensorboard_log_dir "$RUN_ROOT" \
      -e "$da_exp" \
      timematch \
      --weights "$weights_dir" \
      --epochs "$TIMEMATCH_EPOCHS" \
      --steps_per_epoch "$TIMEMATCH_STEPS_PER_EPOCH" \
      --timematch_source_structure_trade_off "$da_structure_weight"
    echo "DONE|${mode}|${task_name}|${da_exp}"
  ) > "$log_file" 2>&1 &
}

TASKS=(FR1_to_AT1 FR2_to_DK1 AT1_to_DK1 FR1_to_DK1)
MODES=(A_daoff B_daon C_daoff D_daoff)

job_index=0
for mode in "${MODES[@]}"; do
  for task_name in "${TASKS[@]}"; do
    gpu_index=$((job_index % ${#GPU_ITEMS[@]}))
    run_one "$gpu_index" "$mode" "$task_name"
    job_index=$((job_index + 1))
    if [ $((job_index % ${#GPU_ITEMS[@]})) -eq 0 ]; then
      wait
    fi
  done
done

wait

echo "All v2.4.3b same-source DA-shaping flip jobs finished."
echo "Logs saved under: $LOG_DIR"
