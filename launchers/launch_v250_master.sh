#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-full12}"

case "$MODE" in
  quickcheck)
    echo "[v2.5.0] Running checkpoint-bank quickcheck..."
    bash "$SCRIPT_DIR/launch_four_gpu_v250_checkpointbank_v243b_quickcheck.sh"
    ;;
  full12)
    echo "[v2.5.0] Running checkpoint-bank full 12-task sweep..."
    bash "$SCRIPT_DIR/launch_four_gpu_v250_checkpointbank_v243b_analysis.sh"
    ;;
  quickcheck_then_full)
    echo "[v2.5.0] Running checkpoint-bank quickcheck, then full 12-task sweep..."
    bash "$SCRIPT_DIR/launch_four_gpu_v250_checkpointbank_v243b_quickcheck.sh"
    bash "$SCRIPT_DIR/launch_four_gpu_v250_checkpointbank_v243b_analysis.sh"
    ;;
  *)
    echo "Unknown MODE=$MODE"
    echo "Valid modes: quickcheck, full12, quickcheck_then_full"
    exit 1
    ;;
esac
