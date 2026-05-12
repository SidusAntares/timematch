#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REPEAT_TAGS="${REPEAT_TAGS:-runA,runB}"
IFS=',' read -r -a REPEAT_ITEMS <<< "$REPEAT_TAGS"

for repeat_tag in "${REPEAT_ITEMS[@]}"; do
  repeat_tag="$(echo "$repeat_tag" | xargs)"
  if [ -z "$repeat_tag" ]; then
    continue
  fi

  echo "===== Starting v2.5.2 repeat: ${repeat_tag} ====="
  RUN_SUFFIX="$repeat_tag" \
  RUN_TAG="v252_component_controller_${repeat_tag}" \
  bash "$ROOT_DIR/launchers/launch_four_gpu_v252_component_controller_analysis.sh"
  echo "===== Finished v2.5.2 repeat: ${repeat_tag} ====="
done
