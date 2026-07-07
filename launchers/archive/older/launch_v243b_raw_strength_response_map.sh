#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This wrapper records a response curve, not a per-task winner.
# Keep w0.25 to preserve the lower-bound signal from the first intervention.
export RUN_TAG="${RUN_TAG:-v243b_raw_strength_response_map}"
export TASKS="${TASKS:-FR2_to_FR1,FR1_to_AT1,FR2_to_DK1,DK1_to_FR1,AT1_to_FR2}"
export SEEDS="${SEEDS:-1 2 3}"
export RAW_WEIGHTS="${RAW_WEIGHTS:-0.25 0.5 1.0 2.0}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export GPUS="${GPUS:-0 1 2 3}"

echo "RUN_TAG=$RUN_TAG"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "RAW_WEIGHTS=$RAW_WEIGHTS"
echo "NOTE=This run outputs strength curves and pseudolabel divergence, not best-weight selection."

bash "$SCRIPT_DIR/launch_v243b_raw_strength_causal_intervention.sh"
