#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TAG="${RUN_TAG:-v286_regime_state_routing_diagnostic_$(date +%Y%m%d_%H%M%S)}"
V285_DIR="${V285_DIR:-$ROOT_DIR/logs/v285_regime_diagnostic_20260627_201553}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/$RUN_TAG}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR" || exit 2

python analysis/v286_regime_state_routing_diagnostic.py \
  --v285_dir "$V285_DIR" \
  --output_dir "$OUTPUT_DIR"
