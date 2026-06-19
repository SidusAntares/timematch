#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-v276_fr2_center_probe}"
export TASKS="${TASKS:-FR2_to_FR1}"
export SEEDS="${SEEDS:-1 2 3}"
export CONFIGS="${CONFIGS:-v275_raw_w1,v276_timepoint_w1,v276_smoothed_timepoint_w1,v276_trimmed_w1}"
export CLOSED_SET="${CLOSED_SET:-True}"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
export DA_EPOCHS="${DA_EPOCHS:-20}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
export V275_WEIGHT="${V275_WEIGHT:-1.0}"

bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
