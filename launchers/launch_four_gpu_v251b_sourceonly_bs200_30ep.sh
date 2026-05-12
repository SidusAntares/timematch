#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v251b_sourceonly_bs200_30ep}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-200}"
SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-30}"

SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
SOURCE_FEATURE_RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

SOURCE_PHASE_PARTITION_MODE="${SOURCE_PHASE_PARTITION_MODE:-doy_gap}"
SOURCE_PHASE_COUNT="${SOURCE_PHASE_COUNT:-5}"
SOURCE_SEGMENT_PARTITION_MODE="${SOURCE_SEGMENT_PARTITION_MODE:-$SOURCE_PHASE_PARTITION_MODE}"
SOURCE_SEGMENT_COUNT="${SOURCE_SEGMENT_COUNT:-$SOURCE_PHASE_COUNT}"
SOURCE_PHASE_GAP_THRESHOLD="${SOURCE_PHASE_GAP_THRESHOLD:-45}"
SOURCE_PHASE_MIN_POINTS="${SOURCE_PHASE_MIN_POINTS:-3}"
SOURCE_PHASE_MAX_POINTS="${SOURCE_PHASE_MAX_POINTS:-8}"
SOURCE_PHASE_MAX_SPAN="${SOURCE_PHASE_MAX_SPAN:-120}"
SOURCE_PHASE_MIN_SAMPLE_POINTS="${SOURCE_PHASE_MIN_SAMPLE_POINTS:-2}"

SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
SOURCE_STRUCTURE_INTRA_TRADE_OFF="${SOURCE_STRUCTURE_INTRA_TRADE_OFF:-1.0}"
SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.05}"
SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.02}"
SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.20}"
SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"

run_one() {
  local gpu_id="$1"
  local run_suffix="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local source_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  local target_tile
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  local exp_tag="v251b_boundarywindow_s010_rel003_bs200_30ep_${source_tile}_to_${target_tile}_${run_suffix}"
  local source_model="pseltae_${source_tile}_closedset_noshift_sourcephasecompact_p5_${exp_tag}"
  local log_file="$LOG_DIR/gpu${gpu_id}_${source_tile}_to_${target_tile}_${run_suffix}.log"

  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set True \
      --with_shift_aug False \
      --epochs "$SOURCE_PRETRAIN_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --source_feature_reshaper "$SOURCE_FEATURE_RESHAPER" \
      --source_feature_reshaper_strength "$SOURCE_FEATURE_RESHAPER_STRENGTH" \
      --source_feature_reshaper_kernel_size "$SOURCE_FEATURE_RESHAPER_KERNEL_SIZE" \
      --source_feature_reshaper_reg_trade_off "$SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF" \
      --source_feature_dual_path True \
      --source_feature_dual_cls_trade_off "$SOURCE_FEATURE_DUAL_CLS_TRADE_OFF" \
      --source_feature_dual_relation_trade_off "$SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF" \
      --source_phase_partition_mode "$SOURCE_PHASE_PARTITION_MODE" \
      --source_segment_partition_mode "$SOURCE_SEGMENT_PARTITION_MODE" \
      --source_phase_count "$SOURCE_PHASE_COUNT" \
      --source_segment_count "$SOURCE_SEGMENT_COUNT" \
      --source_phase_gap_threshold "$SOURCE_PHASE_GAP_THRESHOLD" \
      --source_phase_min_points "$SOURCE_PHASE_MIN_POINTS" \
      --source_phase_max_points "$SOURCE_PHASE_MAX_POINTS" \
      --source_phase_max_span "$SOURCE_PHASE_MAX_SPAN" \
      --source_phase_min_sample_points "$SOURCE_PHASE_MIN_SAMPLE_POINTS" \
      --source_structure_loss_version "$SOURCE_STRUCTURE_LOSS_VERSION" \
      --source_structure_intra_trade_off "$SOURCE_STRUCTURE_INTRA_TRADE_OFF" \
      --source_structure_amplitude_trade_off "$SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF" \
      --source_structure_interphase_trade_off "$SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF" \
      --source_structure_shape_trade_off "$SOURCE_STRUCTURE_SHAPE_TRADE_OFF" \
      --source_structure_trend_trade_off "$SOURCE_STRUCTURE_TREND_TRADE_OFF" \
      --source_structure_season_trade_off "$SOURCE_STRUCTURE_SEASON_TRADE_OFF" \
      --source_structure_segment_inter_trade_off "$SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF" \
      --source_structure_boundary_window_trade_off "$SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF" \
      --source_structure_boundary_window_size "$SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE" \
      -e "$source_model" \
      --source "$source_dataset" \
      --target "$source_dataset" \
      sourcephasecompact

    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set True \
      --with_shift_aug False \
      --epochs "$SOURCE_PRETRAIN_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --source_feature_reshaper "$SOURCE_FEATURE_RESHAPER" \
      --source_feature_reshaper_strength "$SOURCE_FEATURE_RESHAPER_STRENGTH" \
      --source_feature_reshaper_kernel_size "$SOURCE_FEATURE_RESHAPER_KERNEL_SIZE" \
      --source_feature_reshaper_reg_trade_off "$SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF" \
      --source_feature_dual_path True \
      --source_feature_dual_cls_trade_off "$SOURCE_FEATURE_DUAL_CLS_TRADE_OFF" \
      --source_feature_dual_relation_trade_off "$SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF" \
      --source_phase_partition_mode "$SOURCE_PHASE_PARTITION_MODE" \
      --source_segment_partition_mode "$SOURCE_SEGMENT_PARTITION_MODE" \
      --source_phase_count "$SOURCE_PHASE_COUNT" \
      --source_segment_count "$SOURCE_SEGMENT_COUNT" \
      --source_phase_gap_threshold "$SOURCE_PHASE_GAP_THRESHOLD" \
      --source_phase_min_points "$SOURCE_PHASE_MIN_POINTS" \
      --source_phase_max_points "$SOURCE_PHASE_MAX_POINTS" \
      --source_phase_max_span "$SOURCE_PHASE_MAX_SPAN" \
      --source_phase_min_sample_points "$SOURCE_PHASE_MIN_SAMPLE_POINTS" \
      --source_structure_loss_version "$SOURCE_STRUCTURE_LOSS_VERSION" \
      --source_structure_intra_trade_off "$SOURCE_STRUCTURE_INTRA_TRADE_OFF" \
      --source_structure_amplitude_trade_off "$SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF" \
      --source_structure_interphase_trade_off "$SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF" \
      --source_structure_shape_trade_off "$SOURCE_STRUCTURE_SHAPE_TRADE_OFF" \
      --source_structure_trend_trade_off "$SOURCE_STRUCTURE_TREND_TRADE_OFF" \
      --source_structure_season_trade_off "$SOURCE_STRUCTURE_SEASON_TRADE_OFF" \
      --source_structure_segment_inter_trade_off "$SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF" \
      --source_structure_boundary_window_trade_off "$SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF" \
      --source_structure_boundary_window_size "$SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE" \
      -e "$source_model" \
      --source "$source_dataset" \
      --target "$target_dataset" \
      --eval
  ) > "$log_file" 2>&1 &
}

run_one 0 "runA" "france/31TCJ/2017" "denmark/32VNH/2017"
run_one 1 "runB" "france/31TCJ/2017" "denmark/32VNH/2017"
run_one 2 "runA" "denmark/32VNH/2017" "france/31TCJ/2017"
run_one 3 "runB" "denmark/32VNH/2017" "france/31TCJ/2017"

wait

echo "Logs saved to: $LOG_DIR"
