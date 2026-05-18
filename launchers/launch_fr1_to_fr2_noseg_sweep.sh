#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BLOCK_SCRIPT="$ROOT_DIR/launchers/ideas/run_timematch_closed_set_sourcephasecompact_v255_allcheckpoints_block.sh"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-fr1_fr2_noseg_sweep_${BATCH_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"

# Backbone (defaults)
export SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
export SOURCE_FEATURE_RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
export SOURCE_FEATURE_DUAL_PATH="${SOURCE_FEATURE_DUAL_PATH:-True}"

# Single segment (uniform + count=1 = whole time series as one segment)
export SOURCE_PHASE_PARTITION_MODE="${SOURCE_PHASE_PARTITION_MODE:-uniform}"
export SOURCE_PHASE_COUNT="${SOURCE_PHASE_COUNT:-1}"
export SOURCE_SEGMENT_PARTITION_MODE="${SOURCE_SEGMENT_PARTITION_MODE:-uniform}"
export SOURCE_SEGMENT_COUNT="${SOURCE_SEGMENT_COUNT:-1}"
export SOURCE_PHASE_GAP_THRESHOLD="${SOURCE_PHASE_GAP_THRESHOLD:-45}"
export SOURCE_PHASE_MIN_POINTS="${SOURCE_PHASE_MIN_POINTS:-3}"
export SOURCE_PHASE_MAX_POINTS="${SOURCE_PHASE_MAX_POINTS:-8}"
export SOURCE_PHASE_MAX_SPAN="${SOURCE_PHASE_MAX_SPAN:-120}"
export SOURCE_PHASE_MIN_SAMPLE_POINTS="${SOURCE_PHASE_MIN_SAMPLE_POINTS:-2}"

# Structure loss: only intra compactness is meaningful with 1 segment
export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF="${SOURCE_STRUCTURE_WARP_INVARIANT_TRADE_OFF:-0.00}"

# Disable adaptive weights & grid
export SOURCE_PHASE_GRID_TRADE_OFF="${SOURCE_PHASE_GRID_TRADE_OFF:-0.0}"
export SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS="${SOURCE_STRUCTURE_ADAPTIVE_WEIGHTS:-False}"
export SOURCE_STRUCTURE_ADAPTIVITY_MODE="${SOURCE_STRUCTURE_ADAPTIVITY_MODE:-none}"
export SOURCE_SKIP_TRAIN="${SOURCE_SKIP_TRAIN:-0}"
export SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-}"
export SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-model.pt}"

# ─── 24 variants ──────────────────────────────────────────
# Format: name|intra_trade_off|reshaper_reg|dual_relation
#   intra_sweep: 12 values covering 0.00 → 5.00
#   reg_sweep:   4 combos of intra × reshaper_reg
#   rel_sweep:   4 combos of intra × dual_relation
#   combo:       2 combos of reg+rel together
#   extreme:     2 extreme intra values
VARIANT_SPECS="${VARIANT_SPECS:-\
intra_0p00|0.00|0.05|0.03,\
intra_0p01|0.01|0.05|0.03,\
intra_0p05|0.05|0.05|0.03,\
intra_0p10|0.10|0.05|0.03,\
intra_0p25|0.25|0.05|0.03,\
intra_0p50|0.50|0.05|0.03,\
intra_0p75|0.75|0.05|0.03,\
intra_1p00|1.00|0.05|0.03,\
intra_1p50|1.50|0.05|0.03,\
intra_2p00|2.00|0.05|0.03,\
intra_3p00|3.00|0.05|0.03,\
intra_5p00|5.00|0.05|0.03,\
intra0p5_reg0p01|0.50|0.01|0.03,\
intra0p5_reg0p10|0.50|0.10|0.03,\
intra1p0_reg0p01|1.00|0.01|0.03,\
intra1p0_reg0p10|1.00|0.10|0.03,\
intra0p5_rel0p01|0.50|0.05|0.01,\
intra0p5_rel0p10|0.50|0.05|0.10,\
intra1p0_rel0p01|1.00|0.05|0.01,\
intra1p0_rel0p10|1.00|0.05|0.10,\
intra0p5_reg0p01_rel0p01|0.50|0.01|0.01,\
intra1p0_reg0p125_rel0p05|1.00|0.125|0.05,\
intra_0p001|0.001|0.05|0.03,\
intra_10p0|10.00|0.05|0.03\
}"

# Single task: FR1 → FR2
TASK_SPECS="FR1_to_FR2|france/30TXT/2017|france/31TCJ/2017"

# ─── Parallel dispatch ────────────────────────────────────
IFS=',' read -r -a GPU_ITEMS <<< "$GPU_IDS"
IFS=',' read -r -a VARIANT_ITEMS <<< "$VARIANT_SPECS"
IFS=',' read -r -a TASK_ITEMS <<< "$TASK_SPECS"

job_count=0
gpu_cursor=0

run_one() {
  local variant_name="$1"
  local intra_weight="$2"
  local reshaper_reg="$3"
  local dual_rel="$4"
  local task_name="$5"
  local source_dataset="$6"
  local target_dataset="$7"

  local gpu_id="${GPU_ITEMS[$((gpu_cursor % ${#GPU_ITEMS[@]}))]}"
  gpu_cursor=$((gpu_cursor + 1))

  local source_tile
  local target_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"

  local job_tag="noseg_${variant_name}_${BATCH_STAMP}"
  local log_file="$LOG_DIR/${job_tag}.log"

  echo "START|${job_tag}|gpu=${gpu_id}|intra=${intra_weight}|reg=${reshaper_reg}|rel=${dual_rel}"
  (
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$target_dataset" \
      VARIANT_NAME="${variant_name}" \
      SOURCE_STRUCTURE_INTRA_TRADE_OFF="$intra_weight" \
      SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$reshaper_reg" \
      SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="$dual_rel" \
      SOURCE_MODEL_TAG="${job_tag}_source" \
      RESHAPER_TAG="${job_tag}" \
      TARGET_MODEL_TAG="${job_tag}" \
      bash "$BLOCK_SCRIPT"
  ) > "$log_file" 2>&1 &

  job_count=$((job_count + 1))
  if [ $((job_count % MAX_PARALLEL)) -eq 0 ]; then
    wait
  fi
}

echo "======================================================"
echo "FR1->FR2 不分段 sweep | $(date)"
echo "Logs: $LOG_DIR"
echo "GPUs: $GPU_IDS"
echo "SOURCE_PRETRAIN_EPOCHS=$SOURCE_PRETRAIN_EPOCHS  TIMEMATCH_EPOCHS=$TIMEMATCH_EPOCHS"
echo "Variant count: 24"
echo "======================================================"

for variant_spec in "${VARIANT_ITEMS[@]}"; do
  IFS='|' read -r variant_name intra_weight reshaper_reg dual_rel <<< "$variant_spec"
  for task_spec in "${TASK_ITEMS[@]}"; do
    IFS='|' read -r task_name source_dataset target_dataset <<< "$task_spec"
    run_one "$variant_name" "$intra_weight" "$reshaper_reg" "$dual_rel" \
            "$task_name" "$source_dataset" "$target_dataset"
  done
done

wait

echo "======================================================"
echo "All FR1->FR2 不分段 sweep jobs finished | $(date)"
echo "Logs: $LOG_DIR"
echo "======================================================"
