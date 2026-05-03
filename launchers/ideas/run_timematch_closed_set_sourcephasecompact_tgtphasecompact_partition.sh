#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
PARTITION="${PARTITION:-A}"

run_task() {
  local source="$1"
  local target="$2"
  SOURCE="$source" TARGET="$target" DATA_ROOT="$DATA_ROOT" \
    TARGET_STRUCT_TRADE_OFF="${TARGET_STRUCT_TRADE_OFF:-0.05}" \
    TARGET_STRUCT_WARMUP_EPOCHS="${TARGET_STRUCT_WARMUP_EPOCHS:-2}" \
    bash "$SCRIPT_DIR/run_timematch_closed_set_sourcephasecompact_tgtphasecompact.sh"
}

if [[ "$PARTITION" == "A" ]]; then
  run_task "france/30TXT/2017" "france/31TCJ/2017"
  run_task "france/30TXT/2017" "denmark/32VNH/2017"
  run_task "france/30TXT/2017" "austria/33UVP/2017"

  run_task "france/31TCJ/2017" "france/30TXT/2017"
  run_task "france/31TCJ/2017" "denmark/32VNH/2017"
  run_task "france/31TCJ/2017" "austria/33UVP/2017"
elif [[ "$PARTITION" == "B" ]]; then
  run_task "denmark/32VNH/2017" "france/30TXT/2017"
  run_task "denmark/32VNH/2017" "france/31TCJ/2017"
  run_task "denmark/32VNH/2017" "austria/33UVP/2017"

  run_task "austria/33UVP/2017" "france/30TXT/2017"
  run_task "austria/33UVP/2017" "france/31TCJ/2017"
  run_task "austria/33UVP/2017" "denmark/32VNH/2017"
else
  echo "[ERROR] Unknown PARTITION=$PARTITION. Use A or B."
  exit 1
fi
