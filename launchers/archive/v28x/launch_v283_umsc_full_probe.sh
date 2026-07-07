#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v283_umsc_full_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$RUN_TAG}"
GPUS="${GPUS:-0 1 2 3}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SEEDS="${SEEDS:-1 2 3}"
TASKS="${TASKS:-FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1,FR2_to_DK1,FR2_to_AT1,DK1_to_FR1,DK1_to_FR2,DK1_to_AT1,AT1_to_FR1,AT1_to_FR2,AT1_to_DK1}"
CONFIGS="${CONFIGS:-v283a_umsc_075l3_025linf_w1,v283a_umsc_050l3_050linf_w1,v283b_umsc_060l3_020l5_020linf_w1}"
RUN_NORMDIAG="${RUN_NORMDIAG:-False}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "$LOG_DIR"

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "GPUS=$GPUS"
echo "DATA_ROOT=$DATA_ROOT"
echo "CLOSED_SET=$CLOSED_SET"
echo "SOURCE_EPOCHS=$SOURCE_EPOCHS"
echo "DA_EPOCHS=$DA_EPOCHS"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "SEEDS=$SEEDS"
echo "TASKS=$TASKS"
echo "CONFIGS=$CONFIGS"
echo "RUN_NORMDIAG=$RUN_NORMDIAG"

env \
  RUN_TAG="$RUN_TAG" \
  LOG_DIR="$LOG_DIR" \
  GPUS="$GPUS" \
  DATA_ROOT="$DATA_ROOT" \
  CLOSED_SET="$CLOSED_SET" \
  SOURCE_EPOCHS="$SOURCE_EPOCHS" \
  DA_EPOCHS="$DA_EPOCHS" \
  STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
  NUM_WORKERS="$NUM_WORKERS" \
  SEEDS="$SEEDS" \
  TASKS="$TASKS" \
  CONFIGS="$CONFIGS" \
  DRY_RUN="$DRY_RUN" \
  bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
status="$?"

case "$(echo "$RUN_NORMDIAG" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    if [ "$status" -eq 0 ] && [ -f "$LOG_DIR/raw_strength_rows.tsv" ]; then
      env \
        RUN_TAG="${RUN_TAG}_normdiag" \
        LOG_DIR="$LOG_DIR/normdiag" \
        DATA_ROOT="$DATA_ROOT" \
        GPUS="$GPUS" \
        RUN_SPECS="$LOG_DIR/raw_strength_rows.tsv::$LOG_DIR::" \
        TASKS="" \
        CONFIGS="$CONFIGS" \
        bash "$SCRIPT_DIR/launch_v276_stiffness_diagnostic_existing.sh"
      status="$?"
    fi
    ;;
esac

exit "$status"
