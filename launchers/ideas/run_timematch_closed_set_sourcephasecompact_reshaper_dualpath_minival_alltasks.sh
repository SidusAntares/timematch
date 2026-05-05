#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS=(
  "france/30TXT/2017 france/31TCJ/2017"
  "denmark/32VNH/2017 france/31TCJ/2017"
  "france/31TCJ/2017 france/30TXT/2017"
  "france/30TXT/2017 denmark/32VNH/2017"
)

for task in "${TASKS[@]}"; do
  read -r SOURCE TARGET <<< "$task"
  SOURCE="$SOURCE" TARGET="$TARGET" bash "$SCRIPT_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath.sh"
done
