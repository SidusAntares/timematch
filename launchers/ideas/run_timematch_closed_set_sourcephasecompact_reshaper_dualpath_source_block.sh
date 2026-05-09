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
RESHAPER_TAG="${RESHAPER_TAG:-v223_current_s010_rel003}"
SOURCE_PHASE_PARTITION_MODE="${SOURCE_PHASE_PARTITION_MODE:-uniform}"
SOURCE_PHASE_COUNT="${SOURCE_PHASE_COUNT:-5}"
SOURCE_SEGMENT_PARTITION_MODE="${SOURCE_SEGMENT_PARTITION_MODE:-$SOURCE_PHASE_PARTITION_MODE}"
SOURCE_SEGMENT_COUNT="${SOURCE_SEGMENT_COUNT:-$SOURCE_PHASE_COUNT}"
SOURCE_PHASE_GAP_THRESHOLD="${SOURCE_PHASE_GAP_THRESHOLD:-45}"
SOURCE_PHASE_MIN_POINTS="${SOURCE_PHASE_MIN_POINTS:-3}"
SOURCE_PHASE_MAX_POINTS="${SOURCE_PHASE_MAX_POINTS:-8}"
SOURCE_PHASE_MAX_SPAN="${SOURCE_PHASE_MAX_SPAN:-120}"
SOURCE_PHASE_MIN_SAMPLE_POINTS="${SOURCE_PHASE_MIN_SAMPLE_POINTS:-2}"
SOURCE_SEGMENT_SEMANTIC_QUANTILE="${SOURCE_SEGMENT_SEMANTIC_QUANTILE:-0.75}"
SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS="${SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS:-128}"
SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF="${SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF:-0.50}"
SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF="${SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF:-0.25}"
SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF="${SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF:-0.25}"
SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE="${SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE:-2}"
SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF="${SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF:-0.50}"
SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS="${SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS:-3}"
SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK="${SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK:-1}"
SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE="${SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE:-1.15}"
SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF="${SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF:-0.35}"
SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-compactness}"
SOURCE_STRUCTURE_INTRA_TRADE_OFF="${SOURCE_STRUCTURE_INTRA_TRADE_OFF:-1.0}"
SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.25}"
SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.25}"
SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.15}"
SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.05}"
SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.02}"
SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.02}"
SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.02}"
SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"
SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-}"
SOURCE_CHECKPOINT_DIRNAME="${SOURCE_CHECKPOINT_DIRNAME:-checkpoints}"
SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-model.pt}"

SOURCE_MODEL="${SOURCE_MODEL:-pseltae_${SOURCE_TILE}_closedset_noshift_sourcephasecompact_p5_${RESHAPER_TAG}}"

cd "$ROOT_DIR"

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
  --source_segment_semantic_quantile "$SOURCE_SEGMENT_SEMANTIC_QUANTILE" \
  --source_segment_semantic_max_samples_per_class "$SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS" \
  --source_segment_semantic_curvature_trade_off "$SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF" \
  --source_segment_semantic_energy_trade_off "$SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF" \
  --source_segment_semantic_similarity_trade_off "$SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF" \
  --source_segment_semantic_max_extra_cuts_per_base "$SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE" \
  --source_segment_semantic_merge_boundary_trade_off "$SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF" \
  --source_segment_semantic_aggl_min_points "$SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS" \
  --source_segment_semantic_aggl_target_slack "$SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK" \
  --source_segment_semantic_aggl_merge_cost_tolerance "$SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE" \
  --source_segment_semantic_aggl_dynamics_trade_off "$SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF" \
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
  --source_checkpoint_epochs "$SOURCE_CHECKPOINT_EPOCHS" \
  --source_checkpoint_dirname "$SOURCE_CHECKPOINT_DIRNAME" \
  --epochs "$SOURCE_PRETRAIN_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$SOURCE" \
  sourcephasecompact

mapfile -t SOURCE_CHECKPOINT_ITEMS < <(printf '%s\n' "$SOURCE_WEIGHTS_CHECKPOINTS" | tr ',' '\n' | sed '/^[[:space:]]*$/d')
if [ "${#SOURCE_CHECKPOINT_ITEMS[@]}" -eq 0 ]; then
  SOURCE_CHECKPOINT_ITEMS=("model.pt")
fi

while IFS= read -r TARGET; do
  if [ -z "$TARGET" ]; then
    continue
  fi

  TARGET_TILE="$(echo "$TARGET" | cut -d'/' -f2)"
  BASE_TIMEMATCH_MODEL="timematch_${SOURCE_TILE}_to_${TARGET_TILE}_closedset_noshift_sourcephasecompact_p5_${RESHAPER_TAG}"

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
    --source_segment_semantic_quantile "$SOURCE_SEGMENT_SEMANTIC_QUANTILE" \
    --source_segment_semantic_max_samples_per_class "$SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS" \
    --source_segment_semantic_curvature_trade_off "$SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF" \
    --source_segment_semantic_energy_trade_off "$SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF" \
    --source_segment_semantic_similarity_trade_off "$SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF" \
    --source_segment_semantic_max_extra_cuts_per_base "$SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE" \
    --source_segment_semantic_merge_boundary_trade_off "$SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF" \
    --source_segment_semantic_aggl_min_points "$SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS" \
    --source_segment_semantic_aggl_target_slack "$SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK" \
    --source_segment_semantic_aggl_merge_cost_tolerance "$SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE" \
    --source_segment_semantic_aggl_dynamics_trade_off "$SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF" \
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
    --num_workers "$NUM_WORKERS" \
    -e "$SOURCE_MODEL" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --eval

  for WEIGHTS_CHECKPOINT in "${SOURCE_CHECKPOINT_ITEMS[@]}"; do
    CHECKPOINT_LABEL="$(basename "$WEIGHTS_CHECKPOINT" .pt)"
    CHECKPOINT_LABEL="${CHECKPOINT_LABEL//\//_}"
    TIMEMATCH_MODEL="$BASE_TIMEMATCH_MODEL"
    if [ "$WEIGHTS_CHECKPOINT" != "model.pt" ]; then
      TIMEMATCH_MODEL="${BASE_TIMEMATCH_MODEL}_${CHECKPOINT_LABEL}"
    fi

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
      --source_segment_semantic_quantile "$SOURCE_SEGMENT_SEMANTIC_QUANTILE" \
      --source_segment_semantic_max_samples_per_class "$SOURCE_SEGMENT_SEMANTIC_MAX_SAMPLES_PER_CLASS" \
      --source_segment_semantic_curvature_trade_off "$SOURCE_SEGMENT_SEMANTIC_CURVATURE_TRADE_OFF" \
      --source_segment_semantic_energy_trade_off "$SOURCE_SEGMENT_SEMANTIC_ENERGY_TRADE_OFF" \
      --source_segment_semantic_similarity_trade_off "$SOURCE_SEGMENT_SEMANTIC_SIMILARITY_TRADE_OFF" \
      --source_segment_semantic_max_extra_cuts_per_base "$SOURCE_SEGMENT_SEMANTIC_MAX_EXTRA_CUTS_PER_BASE" \
      --source_segment_semantic_merge_boundary_trade_off "$SOURCE_SEGMENT_SEMANTIC_MERGE_BOUNDARY_TRADE_OFF" \
      --source_segment_semantic_aggl_min_points "$SOURCE_SEGMENT_SEMANTIC_AGGL_MIN_POINTS" \
      --source_segment_semantic_aggl_target_slack "$SOURCE_SEGMENT_SEMANTIC_AGGL_TARGET_SLACK" \
      --source_segment_semantic_aggl_merge_cost_tolerance "$SOURCE_SEGMENT_SEMANTIC_AGGL_MERGE_COST_TOLERANCE" \
      --source_segment_semantic_aggl_dynamics_trade_off "$SOURCE_SEGMENT_SEMANTIC_AGGL_DYNAMICS_TRADE_OFF" \
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
      --num_workers "$NUM_WORKERS" \
      -e "$TIMEMATCH_MODEL" \
      --source "$SOURCE" \
      --target "$TARGET" \
      timematch \
      --epochs "$TIMEMATCH_EPOCHS" \
      --weights "outputs/$SOURCE_MODEL" \
      --weights_checkpoint "$WEIGHTS_CHECKPOINT"
  done
done <<< "$TARGETS_BLOCK"
