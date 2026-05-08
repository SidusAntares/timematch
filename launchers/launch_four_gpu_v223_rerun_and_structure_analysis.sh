#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v223}"
ANALYSIS_SUBDIR="${ANALYSIS_SUBDIR:-${RUN_TAG}_analysis}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_rerun_and_structure_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
DEVICE="${DEVICE:-cuda}"
RESHAPER_TAG="${RESHAPER_TAG:-v223_current_s010_rel003}"
SOURCE_EXPERIMENT_SUFFIX="_sourcephasecompact_p5_${RESHAPER_TAG}"
SKIP_STRUCTURE_ANALYSIS="${SKIP_STRUCTURE_ANALYSIS:-0}"

export DATA_ROOT
export OUTPUTS_ROOT
export DEVICE
export SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
export SOURCE_FEATURE_RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"
export RESHAPER_TAG

run_source_block() {
  local gpu_id="$1"
  local source_dataset="$2"
  local targets_block="$3"
  local source_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  local log_file="$LOG_DIR/gpu${gpu_id}_${source_tile}_${RUN_TAG}_tasks.log"

  (
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$targets_block" \
      bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_source_block.sh"
  ) > "$log_file" 2>&1 &
}

run_source_block 0 "france/30TXT/2017" $'france/31TCJ/2017\ndenmark/32VNH/2017\naustria/33UVP/2017'
run_source_block 1 "france/31TCJ/2017" $'france/30TXT/2017\ndenmark/32VNH/2017\naustria/33UVP/2017'
run_source_block 2 "denmark/32VNH/2017" $'france/30TXT/2017\nfrance/31TCJ/2017\naustria/33UVP/2017'
run_source_block 3 "austria/33UVP/2017" $'france/30TXT/2017\nfrance/31TCJ/2017\ndenmark/32VNH/2017'

wait

if [ "$SKIP_STRUCTURE_ANALYSIS" != "1" ]; then
  CUDA_VISIBLE_DEVICES=0 python "$ROOT_DIR/analysis/recompute_transfer_metrics.py" \
    --data_root "$DATA_ROOT" \
    --outputs_root "$OUTPUTS_ROOT" \
    --output_csv "$ROOT_DIR/result/${ANALYSIS_SUBDIR}/transfer_metrics_${RUN_TAG}.csv" \
    --phase_output_csv "$ROOT_DIR/result/${ANALYSIS_SUBDIR}/phase_metrics_${RUN_TAG}.csv" \
    --closed_set True \
    --device "$DEVICE" \
    --phase_partition_mode structure \
    --phase_count 5 \
    --max_feature_samples 2048 \
    --temporal_grid_size 30 \
    --max_acf_lag 10 \
    --num_workers 8 \
    --batch_size 128 \
    --seed 111 \
    --experiment_suffix "$SOURCE_EXPERIMENT_SUFFIX" \
    > "$LOG_DIR/gpu0_${RUN_TAG}_transfer_metrics.log" 2>&1

  CUDA_VISIBLE_DEVICES=0 python "$ROOT_DIR/analysis/analyze_encoded_structure_dimensions.py" \
    --data_root "$DATA_ROOT" \
    --outputs_root "$OUTPUTS_ROOT" \
    --transfer_csv "$ROOT_DIR/result/${ANALYSIS_SUBDIR}/transfer_metrics_${RUN_TAG}.csv" \
    --output_dir "$ROOT_DIR/result/${ANALYSIS_SUBDIR}/encoded_structure_dimension_analysis" \
    --source_experiment_suffix "$SOURCE_EXPERIMENT_SUFFIX" \
    --closed_set True \
    --device "$DEVICE" \
    --batch_size 128 \
    --num_workers 8 \
    --seed 111 \
    --temporal_grid_size 30 \
    --max_acf_lag 10 \
    --phase_count 5 \
    > "$LOG_DIR/gpu0_${RUN_TAG}_dimension_analysis.log" 2>&1
fi

echo "Logs saved to: $LOG_DIR"
echo "Training logs:"
echo "  $LOG_DIR/gpu0_30TXT_${RUN_TAG}_tasks.log"
echo "  $LOG_DIR/gpu1_31TCJ_${RUN_TAG}_tasks.log"
echo "  $LOG_DIR/gpu2_32VNH_${RUN_TAG}_tasks.log"
echo "  $LOG_DIR/gpu3_33UVP_${RUN_TAG}_tasks.log"
if [ "$SKIP_STRUCTURE_ANALYSIS" != "1" ]; then
  echo "Analysis logs:"
  echo "  $LOG_DIR/gpu0_${RUN_TAG}_transfer_metrics.log"
  echo "  $LOG_DIR/gpu0_${RUN_TAG}_dimension_analysis.log"
else
  echo "Structure analysis skipped for RUN_TAG=$RUN_TAG"
fi
