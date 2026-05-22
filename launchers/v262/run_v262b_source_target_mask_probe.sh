#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

export STAMP
export RUN_TAG="${RUN_TAG:-v262b_source_target_mask_probe_${STAMP}}"
export WINDOWS="${WINDOWS:-source_target_mask}"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
export DA_EPOCHS="${DA_EPOCHS:-20}"
export ADAPT_MAX_FACTOR="${ADAPT_MAX_FACTOR:-1.00}"
export WINDOW_MIN_WEIGHT="${WINDOW_MIN_WEIGHT:-0.20}"

echo "v2.6.2b source-target reliability mask probe"
echo "RUN_TAG=${RUN_TAG}"
echo "WINDOWS=${WINDOWS}"
echo "SOURCE_EPOCHS=${SOURCE_EPOCHS} DA_EPOCHS=${DA_EPOCHS}"
echo "WINDOW_MIN_WEIGHT=${WINDOW_MIN_WEIGHT} ADAPT_MAX_FACTOR=${ADAPT_MAX_FACTOR}"

exec bash "${SCRIPT_DIR}/run_v262a_soft_window_probe.sh"
