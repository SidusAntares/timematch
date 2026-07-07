#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v287_stable_descriptor_extraction_$(date +%Y%m%d_%H%M%S)}"
V285_DIR="${V285_DIR:-$ROOT_DIR/logs/v285_regime_diagnostic_20260627_201553}"
V286_DIR="${V286_DIR:-$ROOT_DIR/logs/v286_regime_state_routing_diagnostic_20260627_225007}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/$RUN_TAG}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
MAX_RAW_SAMPLES="${MAX_RAW_SAMPLES:-512}"
MAX_MMD_SAMPLES="${MAX_MMD_SAMPLES:-128}"
MAX_SHIFT="${MAX_SHIFT:-3}"
SEED="${SEED:-111}"
CLOSED_SET="${CLOSED_SET:-True}"
ALLOW_IDENTITY_RULES="${ALLOW_IDENTITY_RULES:-False}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR" || exit 2

python analysis/v287_stable_descriptor_extraction.py \
  --v285_dir "$V285_DIR" \
  --v286_dir "$V286_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --data_root "$DATA_ROOT" \
  --max_raw_samples "$MAX_RAW_SAMPLES" \
  --max_mmd_samples "$MAX_MMD_SAMPLES" \
  --max_shift "$MAX_SHIFT" \
  --seed "$SEED" \
  --closed_set "$CLOSED_SET" \
  --allow_identity_rules "$ALLOW_IDENTITY_RULES"
