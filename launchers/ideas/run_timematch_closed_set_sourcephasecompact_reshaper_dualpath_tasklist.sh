#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${TASKS_BLOCK:-}" ]; then
  echo "[ERROR] TASKS_BLOCK is empty. Expected newline-separated 'SOURCE TARGET' pairs." >&2
  exit 1
fi

while IFS= read -r task; do
  if [ -z "$task" ]; then
    continue
  fi
  read -r SOURCE TARGET <<< "$task"
  SOURCE="$SOURCE" TARGET="$TARGET" bash "$SCRIPT_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath.sh"
done <<< "$TASKS_BLOCK"
