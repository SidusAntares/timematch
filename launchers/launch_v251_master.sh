#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-full12}"

run_quickcheck() {
  bash "$SCRIPT_DIR/launch_four_gpu_v251_selection_quickcheck.sh"
}

run_full12() {
  bash "$SCRIPT_DIR/launch_four_gpu_v251_selection_analysis.sh"
}

case "$MODE" in
  quickcheck)
    echo "[v2.5.1] Running target-aware checkpoint-selection quickcheck..."
    run_quickcheck
    ;;
  full12)
    echo "[v2.5.1] Running target-aware checkpoint-selection full 12-task run..."
    run_full12
    ;;
  quickcheck_then_full)
    echo "[v2.5.1] Running quickcheck, then full 12-task run..."
    run_quickcheck
    export SOURCE_SKIP_TRAIN=1
    run_full12
    ;;
  reuse_quickcheck)
    echo "[v2.5.1] Reusing existing source checkpoints for quickcheck..."
    export SOURCE_SKIP_TRAIN=1
    export SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-v250_boundarywindow_s010_rel003_ckptbank}"
    run_quickcheck
    ;;
  reuse_full12)
    echo "[v2.5.1] Reusing existing source checkpoints for full 12-task run..."
    export SOURCE_SKIP_TRAIN=1
    export SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-v250_boundarywindow_s010_rel003_ckptbank}"
    run_full12
    ;;
  reuse_quickcheck_then_full)
    echo "[v2.5.1] Reusing existing source checkpoints for quickcheck, then full 12-task run..."
    export SOURCE_SKIP_TRAIN=1
    export SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-v250_boundarywindow_s010_rel003_ckptbank}"
    run_quickcheck
    run_full12
    ;;
  *)
    echo "Unknown MODE=$MODE"
    echo "Valid modes: quickcheck, full12, quickcheck_then_full, reuse_quickcheck, reuse_full12, reuse_quickcheck_then_full"
    exit 1
    ;;
esac
