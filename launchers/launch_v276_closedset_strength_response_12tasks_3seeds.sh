#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export RUN_TAG="${RUN_TAG:-v276_closedset_strength_response_12tasks_3seeds}"
export TASKS="${TASKS:-FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1,FR2_to_DK1,FR2_to_AT1,DK1_to_FR1,DK1_to_FR2,DK1_to_AT1,AT1_to_FR1,AT1_to_FR2,AT1_to_DK1}"
export SEEDS="${SEEDS:-1 2 3}"
export RAW_WEIGHTS="${RAW_WEIGHTS:-0.25 0.50 0.75}"
export INCLUDE_PLAIN="${INCLUDE_PLAIN:-False}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-100}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
export BASELINE_ROWS_TSV="${BASELINE_ROWS_TSV:-$ROOT_DIR/logs/v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121/raw_strength_rows.tsv}"

bash "$SCRIPT_DIR/launch_v275_clean_raw_compactness_baseline.sh"
