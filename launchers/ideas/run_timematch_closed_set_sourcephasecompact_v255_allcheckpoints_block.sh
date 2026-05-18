#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE="${SOURCE:?SOURCE is required}"
TARGETS_BLOCK="${TARGETS_BLOCK:?TARGETS_BLOCK is required}"
SOURCE_TILE="$(echo "$SOURCE" | cut -d'/' -f2)"

RESHAPER_KIND="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
DUAL_REL_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"
RESHAPER_TAG="${RESHAPER_TAG:-v255_allcheckpoints_s010_rel003}"
SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-v255_sourcebank_s010_rel003}"
TARGET_MODEL_TAG="${TARGET_MODEL_TAG:-$RESHAPER_TAG}"
VARIANT_NAME="${VARIANT_NAME:-baseline}"

SOURCE_PHASE_PARTITION_MODE="${SOURCE_PHASE_PARTITION_MODE:-doy_gap}"
SOURCE_PHASE_COUNT="${SOURCE_PHASE_COUNT:-5}"
SOURCE_SEGMENT_PARTITION_MODE="${SOURCE_SEGMENT_PARTITION_MODE:-$SOURCE_PHASE_PARTITION_MODE}"
SOURCE_SEGMENT_COUNT="${SOURCE_SEGMENT_COUNT:-$SOURCE_PHASE_COUNT}"
SOURCE_PHASE_GAP_THRESHOLD="${SOURCE_PHASE_GAP_THRESHOLD:-45}"
SOURCE_PHASE_MIN_POINTS="${SOURCE_PHASE_MIN_POINTS:-3}"
SOURCE_PHASE_MAX_POINTS="${SOURCE_PHASE_MAX_POINTS:-8}"
SOURCE_PHASE_MAX_SPAN="${SOURCE_PHASE_MAX_SPAN:-120}"
SOURCE_PHASE_MIN_SAMPLE_POINTS="${SOURCE_PHASE_MIN_SAMPLE_POINTS:-2}"
SOURCE_PHASE_GRID_COUNT="${SOURCE_PHASE_GRID_COUNT:-5}"
SOURCE_PHASE_GRID_TRADE_OFF="${SOURCE_PHASE_GRID_TRADE_OFF:-0.0}"
SOURCE_PHASE_GRID_KERNEL="${SOURCE_PHASE_GRID_KERNEL:-linear}"
SOURCE_PHASE_GRID_BANDWIDTH="${SOURCE_PHASE_GRID_BANDWIDTH:-0.0}"
SOURCE_PHASE_GRID_MIN_SUPPORT="${SOURCE_PHASE_GRID_MIN_SUPPORT:-0.20}"
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
SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF="${SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF:-0.35}"
SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF="${SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF:-0.05}"
SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS="${SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS:-False}"
SOURCE_STRUCTURE_ADAPTIVITY_MODE="${SOURCE_STRUCTURE_ADAPTIVITY_MODE:-none}"
SOURCE_STRUCTURE_RELIABILITY_ZETA="${SOURCE_STRUCTURE_RELIABILITY_ZETA:-0.90}"
SOURCE_STRUCTURE_RELIABILITY_STRENGTH="${SOURCE_STRUCTURE_RELIABILITY_STRENGTH:-0.35}"
SOURCE_STRUCTURE_RELIABILITY_MIN_FACTOR="${SOURCE_STRUCTURE_RELIABILITY_MIN_FACTOR:-0.70}"
SOURCE_STRUCTURE_RELIABILITY_MAX_FACTOR="${SOURCE_STRUCTURE_RELIABILITY_MAX_FACTOR:-1.20}"

SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"

SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-30,50,70,100}"
SOURCE_CHECKPOINT_DIRNAME="${SOURCE_CHECKPOINT_DIRNAME:-checkpoints}"
SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-checkpoints/epoch_30.pt,checkpoints/epoch_50.pt,checkpoints/epoch_70.pt,checkpoints/epoch_100.pt}"
SOURCE_SKIP_TRAIN="${SOURCE_SKIP_TRAIN:-0}"

SOURCE_MODEL="${SOURCE_MODEL:-pseltae_${SOURCE_TILE}_closedset_noshift_sourcephasecompact_p5_${SOURCE_MODEL_TAG}}"

cd "$ROOT_DIR"

echo "V255B_VARIANT|${VARIANT_NAME}|source=${SOURCE_TILE}|source_model=${SOURCE_MODEL}"
echo "V255B_WEIGHTS|${VARIANT_NAME}|trend=${SOURCE_STRUCTURE_TREND_TRADE_OFF}|segment_inter=${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF}|boundary_window=${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF}|reshaper_reg=${RESHAPER_REG_TRADE_OFF}|dual_relation=${DUAL_REL_TRADE_OFF}"
echo "V244_WEIGHTS|${VARIANT_NAME}|prototype_dynamics=${SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF}|intra=${SOURCE_STRUCTURE_INTRA_TRADE_OFF}"

if [ "$SOURCE_SKIP_TRAIN" != "1" ]; then
  python train.py \
    --data_root "$DATA_ROOT" \
    --closed_set True \
    --with_shift_aug False \
    --source_feature_reshaper "$RESHAPER_KIND" \
    --source_feature_reshaper_strength "$RESHAPER_STRENGTH" \
    --source_feature_reshaper_kernel_size "$RESHAPER_KERNEL_SIZE" \
    --source_feature_reshaper_reg_trade_off "$RESHAPER_REG_TRADE_OFF" \
    --source_feature_dual_path True \
    --source_feature_dual_cls_trade_off "$DUAL_CLS_TRADE_OFF" \
    --source_feature_dual_relation_trade_off "$DUAL_REL_TRADE_OFF" \
    --source_phase_partition_mode "$SOURCE_PHASE_PARTITION_MODE" \
    --source_segment_partition_mode "$SOURCE_SEGMENT_PARTITION_MODE" \
    --source_phase_count "$SOURCE_PHASE_COUNT" \
    --source_segment_count "$SOURCE_SEGMENT_COUNT" \
    --source_phase_gap_threshold "$SOURCE_PHASE_GAP_THRESHOLD" \
    --source_phase_min_points "$SOURCE_PHASE_MIN_POINTS" \
    --source_phase_max_points "$SOURCE_PHASE_MAX_POINTS" \
    --source_phase_max_span "$SOURCE_PHASE_MAX_SPAN" \
    --source_phase_min_sample_points "$SOURCE_PHASE_MIN_SAMPLE_POINTS" \
    --source_phase_grid_count "$SOURCE_PHASE_GRID_COUNT" \
    --source_phase_grid_trade_off "$SOURCE_PHASE_GRID_TRADE_OFF" \
    --source_phase_grid_kernel "$SOURCE_PHASE_GRID_KERNEL" \
    --source_phase_grid_bandwidth "$SOURCE_PHASE_GRID_BANDWIDTH" \
    --source_phase_grid_min_support "$SOURCE_PHASE_GRID_MIN_SUPPORT" \
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
    --source_structure_warp_invariant_trade_off "$SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF" \
    --source_structure_prototype_dynamics_trade_off "$SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF" \
    --source_structure_adaptive_weights "$SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS" \
    --source_structure_adaptivity_mode "$SOURCE_STRUCTURE_ADAPTIVITY_MODE" \
    --source_structure_reliability_zeta "$SOURCE_STRUCTURE_RELIABILITY_ZETA" \
    --source_structure_reliability_strength "$SOURCE_STRUCTURE_RELIABILITY_STRENGTH" \
    --source_structure_reliability_min_factor "$SOURCE_STRUCTURE_RELIABILITY_MIN_FACTOR" \
    --source_structure_reliability_max_factor "$SOURCE_STRUCTURE_RELIABILITY_MAX_FACTOR" \
    --source_checkpoint_epochs "$SOURCE_CHECKPOINT_EPOCHS" \
    --source_checkpoint_dirname "$SOURCE_CHECKPOINT_DIRNAME" \
    --epochs "$SOURCE_PRETRAIN_EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    -e "$SOURCE_MODEL" \
    --source "$SOURCE" \
    --target "$SOURCE" \
    sourcephasecompact
fi

mapfile -t SOURCE_CHECKPOINT_ITEMS < <(printf '%s\n' "$SOURCE_WEIGHTS_CHECKPOINTS" | tr ',' '\n' | sed '/^[[:space:]]*$/d')
if [ "${#SOURCE_CHECKPOINT_ITEMS[@]}" -eq 0 ]; then
  SOURCE_CHECKPOINT_ITEMS=("model.pt")
fi

while IFS= read -r TARGET; do
  if [ -z "$TARGET" ]; then
    continue
  fi

  TARGET_TILE="$(echo "$TARGET" | cut -d'/' -f2)"

  for WEIGHTS_CHECKPOINT in "${SOURCE_CHECKPOINT_ITEMS[@]}"; do
    CHECKPOINT_LABEL="$(basename "$WEIGHTS_CHECKPOINT" .pt)"
    CHECKPOINT_LABEL="${CHECKPOINT_LABEL//\//_}"
    FINAL_MODEL="timematch_${SOURCE_TILE}_to_${TARGET_TILE}_closedset_noshift_sourcephasecompact_p5_${TARGET_MODEL_TAG}_${VARIANT_NAME}_${CHECKPOINT_LABEL}"

    echo "V255B_DA_START|${VARIANT_NAME}|${SOURCE_TILE}|${TARGET_TILE}|${WEIGHTS_CHECKPOINT}|${FINAL_MODEL}"
    python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set True \
      --with_shift_aug False \
      --source_feature_reshaper "$RESHAPER_KIND" \
      --source_feature_reshaper_strength "$RESHAPER_STRENGTH" \
      --source_feature_reshaper_kernel_size "$RESHAPER_KERNEL_SIZE" \
      --source_feature_reshaper_reg_trade_off "$RESHAPER_REG_TRADE_OFF" \
      --source_feature_dual_path True \
      --source_feature_dual_cls_trade_off "$DUAL_CLS_TRADE_OFF" \
      --source_feature_dual_relation_trade_off "$DUAL_REL_TRADE_OFF" \
      --source_phase_partition_mode "$SOURCE_PHASE_PARTITION_MODE" \
      --source_segment_partition_mode "$SOURCE_SEGMENT_PARTITION_MODE" \
      --source_phase_count "$SOURCE_PHASE_COUNT" \
      --source_segment_count "$SOURCE_SEGMENT_COUNT" \
      --source_phase_gap_threshold "$SOURCE_PHASE_GAP_THRESHOLD" \
      --source_phase_min_points "$SOURCE_PHASE_MIN_POINTS" \
      --source_phase_max_points "$SOURCE_PHASE_MAX_POINTS" \
      --source_phase_max_span "$SOURCE_PHASE_MAX_SPAN" \
      --source_phase_min_sample_points "$SOURCE_PHASE_MIN_SAMPLE_POINTS" \
      --source_phase_grid_count "$SOURCE_PHASE_GRID_COUNT" \
      --source_phase_grid_trade_off "$SOURCE_PHASE_GRID_TRADE_OFF" \
      --source_phase_grid_kernel "$SOURCE_PHASE_GRID_KERNEL" \
      --source_phase_grid_bandwidth "$SOURCE_PHASE_GRID_BANDWIDTH" \
      --source_phase_grid_min_support "$SOURCE_PHASE_GRID_MIN_SUPPORT" \
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
      --source_structure_warp_invariant_trade_off "$SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF" \
      --source_structure_prototype_dynamics_trade_off "$SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF" \
      --source_structure_adaptive_weights "$SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS" \
      --source_structure_adaptivity_mode "$SOURCE_STRUCTURE_ADAPTIVITY_MODE" \
      --source_structure_reliability_zeta "$SOURCE_STRUCTURE_RELIABILITY_ZETA" \
      --source_structure_reliability_strength "$SOURCE_STRUCTURE_RELIABILITY_STRENGTH" \
      --source_structure_reliability_min_factor "$SOURCE_STRUCTURE_RELIABILITY_MIN_FACTOR" \
      --source_structure_reliability_max_factor "$SOURCE_STRUCTURE_RELIABILITY_MAX_FACTOR" \
      --num_workers "$NUM_WORKERS" \
      --batch_size "$BATCH_SIZE" \
      -e "$FINAL_MODEL" \
      --source "$SOURCE" \
      --target "$TARGET" \
      timematch \
      --epochs "$TIMEMATCH_EPOCHS" \
      --weights "outputs/$SOURCE_MODEL" \
      --weights_checkpoint "$WEIGHTS_CHECKPOINT"
    echo "V255B_DA_DONE|${VARIANT_NAME}|${SOURCE_TILE}|${TARGET_TILE}|${WEIGHTS_CHECKPOINT}|${FINAL_MODEL}"
  done
done <<< "$TARGETS_BLOCK"
