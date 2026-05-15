#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${MODE:-selection_then_reshaper_then_structure}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-$ROOT_DIR/logs/dir123_gtw_direction_chain_$(date +%Y%m%d_%H%M%S)}"
START_DELAY_MINUTES="${START_DELAY_MINUTES:-0}"

mkdir -p "$CHAIN_LOG_DIR"

if [ "$START_DELAY_MINUTES" != "0" ]; then
  echo "Delaying dir123 chain start by ${START_DELAY_MINUTES} minute(s)..."
  sleep "$((START_DELAY_MINUTES * 60))"
fi

run_stage() {
  local name="$1"
  local script="$2"
  echo "===== Starting ${name} ====="
  bash "$script" > "$CHAIN_LOG_DIR/${name}.out" 2>&1
  echo "===== Finished ${name} ====="
}

case "$MODE" in
  selection_only)
    run_stage "dir3_gtw_selector" "$SCRIPT_DIR/launch_dir3_gtw_selector_quickcheck.sh"
    ;;
  structure_only)
    run_stage "dir2_warp_invariant_structure" "$SCRIPT_DIR/launch_dir2_warp_invariant_structure_quickcheck.sh"
    ;;
  reshaper_only)
    run_stage "dir1_monotonic_warp_reshaper" "$SCRIPT_DIR/launch_dir1_monotonic_warp_reshaper_quickcheck.sh"
    ;;
  selection_then_reshaper)
    run_stage "dir3_gtw_selector" "$SCRIPT_DIR/launch_dir3_gtw_selector_quickcheck.sh"
    run_stage "dir1_monotonic_warp_reshaper" "$SCRIPT_DIR/launch_dir1_monotonic_warp_reshaper_quickcheck.sh"
    ;;
  selection_then_structure)
    run_stage "dir3_gtw_selector" "$SCRIPT_DIR/launch_dir3_gtw_selector_quickcheck.sh"
    run_stage "dir2_warp_invariant_structure" "$SCRIPT_DIR/launch_dir2_warp_invariant_structure_quickcheck.sh"
    ;;
  selection_then_reshaper_then_structure)
    run_stage "dir3_gtw_selector" "$SCRIPT_DIR/launch_dir3_gtw_selector_quickcheck.sh"
    run_stage "dir1_monotonic_warp_reshaper" "$SCRIPT_DIR/launch_dir1_monotonic_warp_reshaper_quickcheck.sh"
    run_stage "dir2_warp_invariant_structure" "$SCRIPT_DIR/launch_dir2_warp_invariant_structure_quickcheck.sh"
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    echo "Allowed: selection_only, structure_only, reshaper_only, selection_then_reshaper, selection_then_structure, selection_then_reshaper_then_structure" >&2
    exit 2
    ;;
esac

echo "Chain logs saved to: $CHAIN_LOG_DIR"
