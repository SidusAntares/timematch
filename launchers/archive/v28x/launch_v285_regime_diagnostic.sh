#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TAG="${RUN_TAG:-v285_regime_diagnostic_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/$RUN_TAG}"

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR" || exit 2

python analysis/v285_regime_diagnostic.py \
  --output_dir "$OUTPUT_DIR" \
  --input logs/v281_a_group_full_20260622_121128/full12 \
  --input logs/v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121 \
  --input logs/v283_umsc_full_20260624_155911 \
  --input logs/v284_elastic_full12_20260626_181827_20260626_181827
