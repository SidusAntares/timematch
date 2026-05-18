#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

EPOCH_LIST="${EPOCH_LIST:-50,70}"
REPEAT_COUNT="${REPEAT_COUNT:-5}"
SEED_START="${SEED_START:-2601}"
BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MASTER_RUN_TAG="${MASTER_RUN_TAG:-v243b_boundarywindow_repeats_50_70_5x_${BATCH_STAMP}}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-$ROOT_DIR/logs/${MASTER_RUN_TAG}}"

export SKIP_STRUCTURE_ANALYSIS="${SKIP_STRUCTURE_ANALYSIS:-1}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"

mkdir -p "$MASTER_LOG_DIR"

unset SOURCE_MODEL

IFS=',' read -r -a EPOCH_ITEMS <<< "$EPOCH_LIST"

echo "Master log dir: ${MASTER_LOG_DIR}"

ALL_TASK_SPECS="FR1_to_FR2|france/30TXT/2017|france/31TCJ/2017,FR1_to_DK1|france/30TXT/2017|denmark/32VNH/2017,FR1_to_AT1|france/30TXT/2017|austria/33UVP/2017,FR2_to_FR1|france/31TCJ/2017|france/30TXT/2017,FR2_to_DK1|france/31TCJ/2017|denmark/32VNH/2017,FR2_to_AT1|france/31TCJ/2017|austria/33UVP/2017,DK1_to_FR1|denmark/32VNH/2017|france/30TXT/2017,DK1_to_FR2|denmark/32VNH/2017|france/31TCJ/2017,DK1_to_AT1|denmark/32VNH/2017|austria/33UVP/2017,AT1_to_FR1|austria/33UVP/2017|france/30TXT/2017,AT1_to_FR2|austria/33UVP/2017|france/31TCJ/2017,AT1_to_DK1|austria/33UVP/2017|denmark/32VNH/2017"

run_index=0
for source_epochs in "${EPOCH_ITEMS[@]}"; do
  source_epochs="$(echo "$source_epochs" | xargs)"
  if [ -z "$source_epochs" ]; then
    continue
  fi

  for repeat_id in $(seq 1 "$REPEAT_COUNT"); do
    run_index=$((run_index + 1))
    train_seed=$((SEED_START + run_index - 1))
    repeat_label="e${source_epochs}_r${repeat_id}_s${train_seed}_${BATCH_STAMP}"

    export SOURCE_PRETRAIN_EPOCHS="$source_epochs"
    export TRAIN_SEED="$train_seed"
    export RUN_TAG="v243b_boundarywindow_${repeat_label}"
    export ANALYSIS_SUBDIR="${RUN_TAG}_analysis"
    export RESHAPER_TAG="v243b_boundarywindow_s010_rel003_${repeat_label}"
    export LOG_DIR="$MASTER_LOG_DIR/${RUN_TAG}"
    export TASK_SPECS="$ALL_TASK_SPECS"
    export VARIANT_SPECS="base:0.05:0.02:0.20"

    echo "===== Starting v2.4.3b repeat: source_epochs=${source_epochs}, repeat=${repeat_id}, seed=${train_seed} ====="
    echo "RUN_TAG=${RUN_TAG}"
    echo "RESHAPER_TAG=${RESHAPER_TAG}"
    echo "LOG_DIR=${LOG_DIR}"
    bash "$ROOT_DIR/launchers/gain_validation/run_v243b_negative_task_structure_weight_sweep.sh"
    echo "===== Finished v2.4.3b repeat: source_epochs=${source_epochs}, repeat=${repeat_id}, seed=${train_seed} ====="
  done
done

echo "All v2.4.3b repeat runs finished."
echo "Batch stamp: ${BATCH_STAMP}"
echo "Master log dir: ${MASTER_LOG_DIR}"
