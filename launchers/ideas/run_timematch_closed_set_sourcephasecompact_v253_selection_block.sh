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
RESHAPER_TAG="${RESHAPER_TAG:-v253_selection_s010_rel003}"
SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-v250_boundarywindow_s010_rel003_ckptbank}"
TARGET_MODEL_TAG="${TARGET_MODEL_TAG:-$RESHAPER_TAG}"

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

SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"

SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-30,50,70,100}"
SOURCE_CHECKPOINT_DIRNAME="${SOURCE_CHECKPOINT_DIRNAME:-checkpoints}"
SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-checkpoints/epoch_30.pt,checkpoints/epoch_50.pt,checkpoints/epoch_70.pt,checkpoints/epoch_100.pt}"
SOURCE_SKIP_TRAIN="${SOURCE_SKIP_TRAIN:-1}"
SELECTION_ONLY="${SELECTION_ONLY:-0}"

SELECTION_WARMUP_EPOCHS="${SELECTION_WARMUP_EPOCHS:-5}"
SELECTION_METRIC_BATCHES="${SELECTION_METRIC_BATCHES:-200}"
SELECTION_COVERAGE_WEIGHT="${SELECTION_COVERAGE_WEIGHT:-0.25}"
SELECTION_CONFIDENCE_WEIGHT="${SELECTION_CONFIDENCE_WEIGHT:-0.15}"
SELECTION_AGREEMENT_WEIGHT="${SELECTION_AGREEMENT_WEIGHT:-0.15}"
SELECTION_ENTROPY_WEIGHT="${SELECTION_ENTROPY_WEIGHT:-0.10}"
SELECTION_CLASS_BALANCE_WEIGHT="${SELECTION_CLASS_BALANCE_WEIGHT:-0.45}"
SELECTION_SOURCE_PRIOR_WEIGHT="${SELECTION_SOURCE_PRIOR_WEIGHT:-0.25}"
SELECTION_SHIFT_STABILITY_WEIGHT="${SELECTION_SHIFT_STABILITY_WEIGHT:-0.20}"
SELECTION_SCORE_MODE="${SELECTION_SCORE_MODE:-temporal_perturbation}"
SELECTION_TIME_MASK_P="${SELECTION_TIME_MASK_P:-0.15}"
SELECTION_TEMPORAL_JITTER="${SELECTION_TEMPORAL_JITTER:-3}"
SELECTION_VALUE_NOISE_STD="${SELECTION_VALUE_NOISE_STD:-0.03}"
SELECTION_PERTURBATION_WEIGHT="${SELECTION_PERTURBATION_WEIGHT:-1.0}"
SELECTION_COLLAPSE_PENALTY_WEIGHT="${SELECTION_COLLAPSE_PENALTY_WEIGHT:-0.35}"
SELECTION_TRAJECTORY_ALPHA="${SELECTION_TRAJECTORY_ALPHA:-0.30}"
SELECTION_LATE_GAIN_THRESHOLD="${SELECTION_LATE_GAIN_THRESHOLD:-0.20}"
SELECTION_LATE_REJECT_THRESHOLD="${SELECTION_LATE_REJECT_THRESHOLD:-0.80}"
SELECTION_MARGIN_TIEBREAK="${SELECTION_MARGIN_TIEBREAK:-0.01}"
SELECTION_BLEND_ROBUST_WEIGHT="${SELECTION_BLEND_ROBUST_WEIGHT:-0.70}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-max_selection_score}"
SELECTION_ROBUST_TIEBREAK_MARGIN="${SELECTION_ROBUST_TIEBREAK_MARGIN:-0.01}"

SOURCE_MODEL="${SOURCE_MODEL:-pseltae_${SOURCE_TILE}_closedset_noshift_sourcephasecompact_p5_${SOURCE_MODEL_TAG}}"

cd "$ROOT_DIR"

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
  BASE_TIMEMATCH_MODEL="timematch_${SOURCE_TILE}_to_${TARGET_TILE}_closedset_noshift_sourcephasecompact_p5_${TARGET_MODEL_TAG}"
  SELECTION_DIR="outputs/${BASE_TIMEMATCH_MODEL}_selection"
  mkdir -p "$SELECTION_DIR"

  for WEIGHTS_CHECKPOINT in "${SOURCE_CHECKPOINT_ITEMS[@]}"; do
    CHECKPOINT_LABEL="$(basename "$WEIGHTS_CHECKPOINT" .pt)"
    CHECKPOINT_LABEL="${CHECKPOINT_LABEL//\//_}"
    WARMUP_MODEL="${BASE_TIMEMATCH_MODEL}_warmup_${CHECKPOINT_LABEL}"
    METRICS_JSON="$SELECTION_DIR/${CHECKPOINT_LABEL}_metrics.json"

    python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set True \
      --with_shift_aug False \
      --skip_final_test True \
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
      --batch_size "$BATCH_SIZE" \
      -e "$WARMUP_MODEL" \
      --source "$SOURCE" \
      --target "$TARGET" \
      timematch \
      --epochs "$SELECTION_WARMUP_EPOCHS" \
      --weights "outputs/$SOURCE_MODEL" \
      --weights_checkpoint "$WEIGHTS_CHECKPOINT" \
      --disable_validation_in_timematch True \
      --selection_metrics_out "$METRICS_JSON" \
      --selection_metric_batches "$SELECTION_METRIC_BATCHES" \
      --selection_score_coverage_weight "$SELECTION_COVERAGE_WEIGHT" \
      --selection_score_confidence_weight "$SELECTION_CONFIDENCE_WEIGHT" \
      --selection_score_agreement_weight "$SELECTION_AGREEMENT_WEIGHT" \
      --selection_score_entropy_weight "$SELECTION_ENTROPY_WEIGHT" \
      --selection_score_class_balance_weight "$SELECTION_CLASS_BALANCE_WEIGHT" \
      --selection_score_source_prior_weight "$SELECTION_SOURCE_PRIOR_WEIGHT" \
      --selection_score_shift_stability_weight "$SELECTION_SHIFT_STABILITY_WEIGHT" \
      --selection_score_mode "$SELECTION_SCORE_MODE" \
      --selection_time_mask_p "$SELECTION_TIME_MASK_P" \
      --selection_temporal_jitter "$SELECTION_TEMPORAL_JITTER" \
      --selection_value_noise_std "$SELECTION_VALUE_NOISE_STD" \
      --selection_perturbation_weight "$SELECTION_PERTURBATION_WEIGHT" \
      --selection_collapse_penalty_weight "$SELECTION_COLLAPSE_PENALTY_WEIGHT" \
      --selection_trajectory_alpha "$SELECTION_TRAJECTORY_ALPHA" \
      --selection_late_gain_threshold "$SELECTION_LATE_GAIN_THRESHOLD" \
      --selection_late_reject_threshold "$SELECTION_LATE_REJECT_THRESHOLD" \
      --selection_margin_tiebreak "$SELECTION_MARGIN_TIEBREAK" \
      --selection_blend_robust_weight "$SELECTION_BLEND_ROBUST_WEIGHT"
  done

  SELECTION_SUMMARY_JSON="$SELECTION_DIR/selection_summary.json"
  python analysis/select_best_checkpoint_from_metrics.py \
    --metrics_glob "$SELECTION_DIR/*_metrics.json" \
    --output_json "$SELECTION_SUMMARY_JSON" \
    --strategy "$SELECTION_STRATEGY" \
    --robust_tiebreak_margin "$SELECTION_ROBUST_TIEBREAK_MARGIN"

  BEST_WEIGHTS_CHECKPOINT="$(python -c "import json; print(json.load(open(r'$SELECTION_SUMMARY_JSON', 'r', encoding='utf-8'))['best_weights_checkpoint'])")"
  echo "Selected checkpoint for ${SOURCE_TILE} -> ${TARGET_TILE}: ${BEST_WEIGHTS_CHECKPOINT}"
  echo "SELECTION_RESULT|${SOURCE_TILE}|${TARGET_TILE}|${BEST_WEIGHTS_CHECKPOINT}|${SELECTION_SUMMARY_JSON}"

  if [ "$SELECTION_ONLY" = "1" ]; then
    echo "[SELECTION-ONLY MODE] Selected ${BEST_WEIGHTS_CHECKPOINT} for ${SOURCE_TILE} -> ${TARGET_TILE}; skipping final TimeMatch."
    continue
  fi

  FINAL_MODEL="${BASE_TIMEMATCH_MODEL}_selected"
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
    --batch_size "$BATCH_SIZE" \
    -e "$FINAL_MODEL" \
    --source "$SOURCE" \
    --target "$TARGET" \
    timematch \
    --epochs "$TIMEMATCH_EPOCHS" \
    --weights "outputs/$SOURCE_MODEL" \
    --weights_checkpoint "$BEST_WEIGHTS_CHECKPOINT"
done <<< "$TARGETS_BLOCK"
