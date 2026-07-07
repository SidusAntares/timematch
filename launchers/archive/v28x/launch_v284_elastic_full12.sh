#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-v284_elastic_full12_$(date +%Y%m%d_%H%M%S)}"
export TASKS="${TASKS:-FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1,FR2_to_DK1,FR2_to_AT1,DK1_to_FR1,DK1_to_FR2,DK1_to_AT1,AT1_to_FR1,AT1_to_FR2,AT1_to_DK1}"
export SEEDS="${SEEDS:-1 2 3}"
export CONFIGS="${CONFIGS:-v284_elastic_k3_r0_w1,v284_elastic_k3_r1_w1,v284_elastic_k3_r2_w1}"
export CLOSED_SET="${CLOSED_SET:-True}"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
export DA_EPOCHS="${DA_EPOCHS:-20}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export GPUS="${GPUS:-0 1 2 3}"

bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
