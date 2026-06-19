#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MASTER_TAG="${MASTER_TAG:-v277_lowfreq_dct_probe_$(date +%Y%m%d_%H%M%S)}"
export TASKS="${TASKS:-FR2_to_FR1,FR2_to_DK1,AT1_to_DK1}"
export SEEDS="${SEEDS:-1 2 3}"
export CONFIGS="${CONFIGS:-plain,v275_raw_w1,v277_dct_k2_w1,v277_dct_k4_w1,v277_dct_k8_w1,v276_smoothed_timepoint_w1}"
export WEIGHTS="${WEIGHTS:-0.5 1.0}"
export GPUS="${GPUS:-0 1 2 3}"
export CLOSED_SET="${CLOSED_SET:-True}"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
export DA_EPOCHS="${DA_EPOCHS:-20}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"

bash "$SCRIPT_DIR/launch_v276_center_lambda_probe.sh"
