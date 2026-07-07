#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-v275_closedset_baseline_v275_12tasks_3seeds}"
export TASKS="${TASKS:-FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1,FR2_to_DK1,FR2_to_AT1,DK1_to_FR1,DK1_to_FR2,DK1_to_AT1,AT1_to_FR1,AT1_to_FR2,AT1_to_DK1}"
export SEEDS="${SEEDS:-1 2 3}"
export CONFIGS="${CONFIGS:-plain,v275_raw_w1}"
export CLOSED_SET="${CLOSED_SET:-True}"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
export DA_EPOCHS="${DA_EPOCHS:-20}"
export V275_WEIGHT="${V275_WEIGHT:-1.0}"

bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
