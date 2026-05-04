#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
PARTITION="${PARTITION:-A}"

run_task() {
  local source="$1"
  local target="$2"
  SOURCE="$source" TARGET="$target" DATA_ROOT="$DATA_ROOT" \
    SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-none}" \
    SOURCE_FEATURE_RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}" \
    SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}" \
    SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}" \
    RESHAPER_TAG="${RESHAPER_TAG:-${SOURCE_FEATURE_RESHAPER:-none}_s${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}_k${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}_r${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}}" \
    bash "$SCRIPT_DIR/run_timematch_closed_set_sourcephasecompact_reshaper.sh"
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
