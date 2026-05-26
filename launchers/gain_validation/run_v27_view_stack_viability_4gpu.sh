#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v27_view_stack_viability_${BATCH_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/${RUN_TAG}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
VIEW_MODES="${VIEW_MODES:-G_global,G_meaningful_local,G_random_local}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_SOURCE_ON_TARGET="${EVAL_SOURCE_ON_TARGET:-true}"
RANDOM_LOCAL_SEGMENT_COUNT="${RANDOM_LOCAL_SEGMENT_COUNT:-8}"

RESHAPER_KIND="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
DUAL_REL_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

mkdir -p "$LOG_DIR" "$OUT_ROOT" "$RUN_ROOT"
IFS=',' read -r -a GPU_ITEMS <<< "$GPU_IDS"
IFS=',' read -r -a MODE_ITEMS <<< "$VIEW_MODES"

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

view_spec() {
  local mode="$1"
  local base_trend="$2"
  local base_segment_inter="$3"
  local base_boundary="$4"

  case "$mode" in
    G_global)
      echo "v271_global uniform 1 $base_trend 0.0 0.0 0.0"
      ;;
    G_meaningful_local)
      echo "v271_global_segment_basis doy_gap 5 $base_trend $base_segment_inter $base_boundary 1.0"
      ;;
    G_random_local)
      echo "v271_global_segment_basis random_local $RANDOM_LOCAL_SEGMENT_COUNT $base_trend $base_segment_inter $base_boundary 1.0"
      ;;
    *)
      echo "Unknown view mode: $mode" >&2
      exit 1
      ;;
  esac
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
  local segment_basis_weight="$9"

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
    --source_structure_v271_segment_basis_trade_off "$segment_basis_weight"
    --num_workers "$NUM_WORKERS"
    --batch_size "$BATCH_SIZE"
    --source "$source_dataset"
    --target "$target_dataset"
  )
}

run_one() {
  local gpu_id="$1"
  local view_mode="$2"
  local task_name="$3"
  local source_dataset target_dataset base_trend base_segment_inter base_boundary
  read -r source_dataset target_dataset base_trend base_segment_inter base_boundary <<< "$(task_spec "$task_name")"

  local loss_version partition_mode phase_count trend_weight segment_inter_weight boundary_weight segment_basis_weight
  read -r loss_version partition_mode phase_count trend_weight segment_inter_weight boundary_weight segment_basis_weight \
    <<< "$(view_spec "$view_mode" "$base_trend" "$base_segment_inter" "$base_boundary")"

  local source_tile target_tile variant_name source_exp da_exp source_out log_file
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  variant_name="v27view_${view_mode}_${task_name}_t${trend_weight}_si${segment_inter_weight}_b${boundary_weight}_sb${segment_basis_weight}"
  variant_name="${variant_name//./}"
  source_exp="${variant_name}_${source_tile}_source_${BATCH_STAMP}"
  da_exp="${variant_name}_${source_tile}_to_${target_tile}_timematch_${BATCH_STAMP}"
  source_out="${OUT_ROOT}/${source_exp}"
  log_file="$LOG_DIR/${variant_name}_${source_tile}_to_${target_tile}.log"

  echo "START|${view_mode}|${task_name}|gpu=${gpu_id}|loss=${loss_version}|partition=${partition_mode}|phase_count=${phase_count}|trend=${trend_weight}|segment_inter=${segment_inter_weight}|boundary=${boundary_weight}|segment_basis=${segment_basis_weight}|da_structure=${TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF}|log=${log_file}"
  (
    common_args "$source_dataset" "$source_dataset" "$loss_version" "$partition_mode" "$phase_count" "$trend_weight" "$segment_inter_weight" "$boundary_weight" "$segment_basis_weight"
    echo "VIEW_SOURCE_START|${view_mode}|${task_name}|${source_exp}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
      "${COMMON_ARGS[@]}" \
      --output_dir "$OUT_ROOT" \
      --tensorboard_log_dir "$RUN_ROOT" \
      --epochs "$SOURCE_PRETRAIN_EPOCHS" \
      -e "$source_exp" \
      sourcephasecompact

    if [ "$EVAL_SOURCE_ON_TARGET" = "true" ]; then
      common_args "$source_dataset" "$target_dataset" "$loss_version" "$partition_mode" "$phase_count" "$trend_weight" "$segment_inter_weight" "$boundary_weight" "$segment_basis_weight"
      echo "VIEW_SOURCE_TARGET_EVAL|${view_mode}|${task_name}|${source_exp}|target=${target_dataset}"
      CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
        "${COMMON_ARGS[@]}" \
        --output_dir "$OUT_ROOT" \
        --tensorboard_log_dir "$RUN_ROOT" \
        -e "$source_exp" \
        --eval
    fi

    common_args "$source_dataset" "$target_dataset" "$loss_version" "$partition_mode" "$phase_count" "$trend_weight" "$segment_inter_weight" "$boundary_weight" "$segment_basis_weight"
    echo "VIEW_DA_START|${view_mode}|${task_name}|${da_exp}|weights=${source_out}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
      "${COMMON_ARGS[@]}" \
      --output_dir "$OUT_ROOT" \
      --tensorboard_log_dir "$RUN_ROOT" \
      -e "$da_exp" \
      timematch \
      --weights "$source_out" \
      --epochs "$TIMEMATCH_EPOCHS" \
      --steps_per_epoch "$TIMEMATCH_STEPS_PER_EPOCH" \
      --timematch_source_structure_trade_off "$TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF"
    echo "DONE|${view_mode}|${task_name}|${da_exp}"
  ) > "$log_file" 2>&1 &
}

TASKS=(FR1_to_AT1 FR2_to_DK1 AT1_to_DK1 FR1_to_DK1)
job_index=0
for view_mode in "${MODE_ITEMS[@]}"; do
  for task_name in "${TASKS[@]}"; do
    gpu_id="${GPU_ITEMS[$((job_index % ${#GPU_ITEMS[@]}))]}"
    run_one "$gpu_id" "$view_mode" "$task_name"
    job_index=$((job_index + 1))
    if [ $((job_index % ${#GPU_ITEMS[@]})) -eq 0 ]; then
      wait
    fi
  done
done

wait

echo "All v2.7 view-stack viability jobs finished."
echo "Logs saved under: $LOG_DIR"
