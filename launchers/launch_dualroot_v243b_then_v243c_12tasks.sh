#!/bin/bash

set -euo pipefail

ROOT_B="${ROOT_B:-/data/user/timematch}"
ROOT_C="${ROOT_C:-/data/user/ti}"

RUN_TAG_B="${RUN_TAG_B:-v243b_boundarywindow_v222}"
RUN_TAG_C="${RUN_TAG_C:-v243c_boundarykeypoint_v222}"

echo "[1/2] Running v2.4.3b 12-task suite in: $ROOT_B"
(
  cd "$ROOT_B"
  RUN_TAG="$RUN_TAG_B" \
  SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}" \
  TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  bash "launchers/launch_four_gpu_v243_boundarywindow_v222_analysis.sh"
)

echo "[2/2] Running v2.4.3c 12-task suite in: $ROOT_C"
(
  cd "$ROOT_C"
  RUN_TAG="$RUN_TAG_C" \
  SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}" \
  TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  bash "launchers/launch_four_gpu_v243c_boundarykeypoint_v222_analysis.sh"
)

echo "Dual-root sequential run finished."
