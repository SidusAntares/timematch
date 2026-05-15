#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EPOCH_LIST="${EPOCH_LIST:-50,70}"
REPEAT_COUNT="${REPEAT_COUNT:-5}"
SEED_START="${SEED_START:-2601}"
BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export SKIP_STRUCTURE_ANALYSIS="${SKIP_STRUCTURE_ANALYSIS:-1}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

unset SOURCE_MODEL

IFS=',' read -r -a EPOCH_ITEMS <<< "$EPOCH_LIST"

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
    export LOG_DIR="$ROOT_DIR/logs/${RUN_TAG}"

    echo "===== Starting v2.4.3b repeat: source_epochs=${source_epochs}, repeat=${repeat_id}, seed=${train_seed} ====="
    echo "RUN_TAG=${RUN_TAG}"
    echo "RESHAPER_TAG=${RESHAPER_TAG}"
    bash "$ROOT_DIR/launchers/launch_four_gpu_v243b_boundarywindow_v222_analysis_50ep.sh"
    echo "===== Finished v2.4.3b repeat: source_epochs=${source_epochs}, repeat=${repeat_id}, seed=${train_seed} ====="
  done
done

echo "All v2.4.3b repeat runs finished."
echo "Batch stamp: ${BATCH_STAMP}"
