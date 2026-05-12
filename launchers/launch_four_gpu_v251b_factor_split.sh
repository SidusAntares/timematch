#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BLOCK_SCRIPT="$ROOT_DIR/launchers/launch_four_gpu_v251b_sourceonly_bs200_30ep.sh"

RUN_TAG="${RUN_TAG:-v251b_factor_split}"
MODE="${MODE:-30ep_bs128}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

case "$MODE" in
  30ep_bs128)
    export SOURCE_PRETRAIN_EPOCHS=30
    export BATCH_SIZE=128
    ;;
  30ep_bs200)
    export SOURCE_PRETRAIN_EPOCHS=30
    export BATCH_SIZE=200
    ;;
  50ep_bs200)
    export SOURCE_PRETRAIN_EPOCHS=50
    export BATCH_SIZE=200
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    echo "Use one of: 30ep_bs128, 30ep_bs200, 50ep_bs200" >&2
    exit 1
    ;;
esac

export NUM_WORKERS="${NUM_WORKERS:-8}"
export RUN_TAG="${RUN_TAG}_${MODE}"
export LOG_DIR

echo "Started factor-split experiment"
echo "  MODE=$MODE"
echo "  SOURCE_PRETRAIN_EPOCHS=$SOURCE_PRETRAIN_EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  Logs: $LOG_DIR"

bash "$BLOCK_SCRIPT"
