#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export RUN_TAG="${RUN_TAG:-v250_checkpointbank_v243b_12tasks}"
export ANALYSIS_SUBDIR="${ANALYSIS_SUBDIR:-${RUN_TAG}_analysis}"
export RESHAPER_TAG="${RESHAPER_TAG:-v250_boundarywindow_s010_rel003_ckptbank}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-30,50,70,100}"
export SOURCE_CHECKPOINT_DIRNAME="${SOURCE_CHECKPOINT_DIRNAME:-checkpoints}"
export SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-checkpoints/epoch_30.pt,checkpoints/epoch_50.pt,checkpoints/epoch_70.pt,checkpoints/epoch_100.pt}"

bash "$ROOT_DIR/launchers/launch_four_gpu_v25b_checkpointbank_v243b_analysis.sh"
