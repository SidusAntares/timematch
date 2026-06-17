#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-timematch_openset_baseline_12tasks}"
export TASKS="${TASKS:-FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1,FR2_to_DK1,FR2_to_AT1,DK1_to_FR1,DK1_to_FR2,DK1_to_AT1,AT1_to_FR1,AT1_to_FR2,AT1_to_DK1}"
export SEEDS="${SEEDS:-1}"
export CONFIGS="${CONFIGS:-plain}"
export CLOSED_SET="${CLOSED_SET:-False}"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
export DA_EPOCHS="${DA_EPOCHS:-20}"

bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
