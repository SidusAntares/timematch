#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
MASTER_TAG="${MASTER_TAG:-v253efg_selection_only_12tasks}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-$ROOT_DIR/logs/${MASTER_TAG}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$MASTER_LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-v250_boundarywindow_s010_rel003_ckptbank}"
export SOURCE_SKIP_TRAIN="${SOURCE_SKIP_TRAIN:-1}"
export SELECTION_ONLY="${SELECTION_ONLY:-1}"

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
export SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.05}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.02}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.20}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"

export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export SELECTION_WARMUP_EPOCHS="${SELECTION_WARMUP_EPOCHS:-5}"
export SELECTION_METRIC_BATCHES="${SELECTION_METRIC_BATCHES:-200}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-checkpoints/epoch_30.pt,checkpoints/epoch_50.pt,checkpoints/epoch_70.pt,checkpoints/epoch_100.pt}"

export SELECTION_TIME_MASK_P="${SELECTION_TIME_MASK_P:-0.15}"
export SELECTION_TEMPORAL_JITTER="${SELECTION_TEMPORAL_JITTER:-3}"
export SELECTION_VALUE_NOISE_STD="${SELECTION_VALUE_NOISE_STD:-0.03}"
export SELECTION_PERTURBATION_WEIGHT="${SELECTION_PERTURBATION_WEIGHT:-1.0}"
export SELECTION_COLLAPSE_PENALTY_WEIGHT="${SELECTION_COLLAPSE_PENALTY_WEIGHT:-0.35}"
export SELECTION_TRAJECTORY_ALPHA="${SELECTION_TRAJECTORY_ALPHA:-0.30}"
export SELECTION_LATE_GAIN_THRESHOLD="${SELECTION_LATE_GAIN_THRESHOLD:-0.20}"
export SELECTION_LATE_REJECT_THRESHOLD="${SELECTION_LATE_REJECT_THRESHOLD:-0.80}"
export SELECTION_MARGIN_TIEBREAK="${SELECTION_MARGIN_TIEBREAK:-0.01}"

run_variant_source_block() {
  local gpu_id="$1"
  local source_dataset="$2"
  local targets_block="$3"
  local variant_tag="$4"
  local source_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  local log_file="$VARIANT_LOG_DIR/gpu${gpu_id}_${source_tile}_${variant_tag}_tasks.log"

  (
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$targets_block" \
      bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_v253_selection_block.sh"
  ) > "$log_file" 2>&1 &
}

run_variant() {
  local variant_tag="$1"
  local score_mode="$2"
  export RUN_TAG="$variant_tag"
  export RESHAPER_TAG="${variant_tag}_s010_rel003"
  export TARGET_MODEL_TAG="$RESHAPER_TAG"
  export SELECTION_SCORE_MODE="$score_mode"
  VARIANT_LOG_DIR="$MASTER_LOG_DIR/$variant_tag"
  export VARIANT_LOG_DIR
  mkdir -p "$VARIANT_LOG_DIR"

  echo "===== Starting ${variant_tag} (${score_mode}) ====="
  run_variant_source_block 0 "france/30TXT/2017" $'france/31TCJ/2017\ndenmark/32VNH/2017\naustria/33UVP/2017' "$variant_tag"
  run_variant_source_block 1 "france/31TCJ/2017" $'france/30TXT/2017\ndenmark/32VNH/2017\naustria/33UVP/2017' "$variant_tag"
  run_variant_source_block 2 "denmark/32VNH/2017" $'france/30TXT/2017\nfrance/31TCJ/2017\naustria/33UVP/2017' "$variant_tag"
  run_variant_source_block 3 "austria/33UVP/2017" $'france/30TXT/2017\nfrance/31TCJ/2017\ndenmark/32VNH/2017' "$variant_tag"
  wait
  echo "===== Finished ${variant_tag}; logs: ${VARIANT_LOG_DIR} ====="
}

VARIANT_PLAN="${VARIANT_PLAN:-e}"
IFS=',' read -r -a VARIANTS <<< "$VARIANT_PLAN"
for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    e)
      run_variant "v253e_pureperturb_selection_only_12tasks" "pure_perturbation"
      ;;
    f)
      run_variant "v253f_pureperturb_late_reject_selection_only_12tasks" "pure_perturbation_late_reject"
      ;;
    g)
      run_variant "v253g_pureperturb_margin_tiebreak_selection_only_12tasks" "pure_perturbation_margin_tiebreak"
      ;;
    *)
      echo "Unknown variant '${variant}'. Use VARIANT_PLAN=e, f, g, or comma-separated combinations such as e,f,g." >&2
      exit 1
      ;;
  esac
done

echo "Requested v2.5.3 selection-only runs finished: ${VARIANT_PLAN}"
echo "Master logs saved to: $MASTER_LOG_DIR"
