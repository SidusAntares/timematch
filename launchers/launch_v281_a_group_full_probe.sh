#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MASTER_TAG="${MASTER_TAG:-v281_a_group_full_$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-$ROOT_DIR/logs/$MASTER_TAG}"
GPUS="${GPUS:-0 1 2 3}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"

FULL_TASKS="${FULL_TASKS:-FR1_to_FR2,FR1_to_DK1,FR1_to_AT1,FR2_to_FR1,FR2_to_DK1,FR2_to_AT1,DK1_to_FR1,DK1_to_FR2,DK1_to_AT1,AT1_to_FR1,AT1_to_FR2,AT1_to_DK1}"
FULL_CONFIGS="${FULL_CONFIGS:-plain,v276_timepoint_w0p5,v276_timepoint_w1,v276_smooth_k3_w1,v276_smooth_k3_w1_detach}"

KERNEL_TASKS="${KERNEL_TASKS:-DK1_to_AT1,DK1_to_FR1,FR1_to_FR2,FR2_to_FR1}"
KERNEL_CONFIGS="${KERNEL_CONFIGS:-v276_smooth_k1_w1,v276_smooth_k5_w1,v276_smooth_k7_w1}"

RUN_NORMDIAG="${RUN_NORMDIAG:-True}"
DIAG_NUM_WORKERS="${DIAG_NUM_WORKERS:-8}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-128}"
DIAG_OUTPUTS_ROOT="${DIAG_OUTPUTS_ROOT:-outputs}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "$MASTER_LOG_DIR"

SUMMARY="$MASTER_LOG_DIR/controller_summary.tsv"
printf "stage\tstatus\tlog_dir\n" > "$SUMMARY"

echo "MASTER_TAG=$MASTER_TAG"
echo "MASTER_LOG_DIR=$MASTER_LOG_DIR"
echo "GPUS=$GPUS"
echo "DATA_ROOT=$DATA_ROOT"
echo "CLOSED_SET=$CLOSED_SET"
echo "SOURCE_EPOCHS=$SOURCE_EPOCHS"
echo "DA_EPOCHS=$DA_EPOCHS"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "FULL_TASKS=$FULL_TASKS"
echo "FULL_CONFIGS=$FULL_CONFIGS"
echo "KERNEL_TASKS=$KERNEL_TASKS"
echo "KERNEL_CONFIGS=$KERNEL_CONFIGS"
echo "RUN_NORMDIAG=$RUN_NORMDIAG"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "DRY_RUN=True"
    env \
      RUN_TAG="${MASTER_TAG}_full12" \
      LOG_DIR="$MASTER_LOG_DIR/full12" \
      DATA_ROOT="$DATA_ROOT" \
      CLOSED_SET="$CLOSED_SET" \
      SOURCE_EPOCHS="$SOURCE_EPOCHS" \
      DA_EPOCHS="$DA_EPOCHS" \
      STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
      NUM_WORKERS="$NUM_WORKERS" \
      GPUS="$GPUS" \
      TASKS="$FULL_TASKS" \
      CONFIGS="$FULL_CONFIGS" \
      DRY_RUN=True \
      bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
    env \
      RUN_TAG="${MASTER_TAG}_kernel4" \
      LOG_DIR="$MASTER_LOG_DIR/kernel4" \
      DATA_ROOT="$DATA_ROOT" \
      CLOSED_SET="$CLOSED_SET" \
      SOURCE_EPOCHS="$SOURCE_EPOCHS" \
      DA_EPOCHS="$DA_EPOCHS" \
      STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
      NUM_WORKERS="$NUM_WORKERS" \
      GPUS="$GPUS" \
      TASKS="$KERNEL_TASKS" \
      CONFIGS="$KERNEL_CONFIGS" \
      DRY_RUN=True \
      bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
    exit 0
    ;;
esac

failed=0

FULL_LOG_DIR="$MASTER_LOG_DIR/full12"
echo "START_STAGE|stage=full12|log_dir=$FULL_LOG_DIR"
env \
  RUN_TAG="${MASTER_TAG}_full12" \
  LOG_DIR="$FULL_LOG_DIR" \
  DATA_ROOT="$DATA_ROOT" \
  CLOSED_SET="$CLOSED_SET" \
  SOURCE_EPOCHS="$SOURCE_EPOCHS" \
  DA_EPOCHS="$DA_EPOCHS" \
  STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
  NUM_WORKERS="$NUM_WORKERS" \
  GPUS="$GPUS" \
  TASKS="$FULL_TASKS" \
  CONFIGS="$FULL_CONFIGS" \
  bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
status="$?"
printf "full12\t%s\t%s\n" "$status" "$FULL_LOG_DIR" >> "$SUMMARY"
if [ "$status" -ne 0 ]; then
  failed=1
fi
echo "END_STAGE|stage=full12|status=$status"

KERNEL_LOG_DIR="$MASTER_LOG_DIR/kernel4"
echo "START_STAGE|stage=kernel4|log_dir=$KERNEL_LOG_DIR"
env \
  RUN_TAG="${MASTER_TAG}_kernel4" \
  LOG_DIR="$KERNEL_LOG_DIR" \
  DATA_ROOT="$DATA_ROOT" \
  CLOSED_SET="$CLOSED_SET" \
  SOURCE_EPOCHS="$SOURCE_EPOCHS" \
  DA_EPOCHS="$DA_EPOCHS" \
  STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
  NUM_WORKERS="$NUM_WORKERS" \
  GPUS="$GPUS" \
  TASKS="$KERNEL_TASKS" \
  CONFIGS="$KERNEL_CONFIGS" \
  bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
status="$?"
printf "kernel4\t%s\t%s\n" "$status" "$KERNEL_LOG_DIR" >> "$SUMMARY"
if [ "$status" -ne 0 ]; then
  failed=1
fi
echo "END_STAGE|stage=kernel4|status=$status"

case "$(echo "$RUN_NORMDIAG" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    DIAG_LOG_DIR="$MASTER_LOG_DIR/normdiag"
    RUN_SPECS_VALUE=""
    if [ -f "$FULL_LOG_DIR/raw_strength_rows.tsv" ]; then
      RUN_SPECS_VALUE="$FULL_LOG_DIR/raw_strength_rows.tsv::$FULL_LOG_DIR::"
    fi
    if [ -f "$KERNEL_LOG_DIR/raw_strength_rows.tsv" ]; then
      RUN_SPECS_VALUE="$RUN_SPECS_VALUE $KERNEL_LOG_DIR/raw_strength_rows.tsv::$KERNEL_LOG_DIR::"
    fi
    if [ -n "$RUN_SPECS_VALUE" ]; then
      echo "START_STAGE|stage=normdiag|log_dir=$DIAG_LOG_DIR"
      env \
        RUN_TAG="${MASTER_TAG}_normdiag" \
        LOG_DIR="$DIAG_LOG_DIR" \
        DATA_ROOT="$DATA_ROOT" \
        OUTPUTS_ROOT="$DIAG_OUTPUTS_ROOT" \
        GPUS="$GPUS" \
        RUN_SPECS="$RUN_SPECS_VALUE" \
        TASKS="" \
        CONFIGS="plain,v276_timepoint_w0p5,v276_timepoint_w1,v276_smooth_k3_w1,v276_smooth_k3_w1_detach,v276_smooth_k1_w1,v276_smooth_k5_w1,v276_smooth_k7_w1" \
        NUM_WORKERS="$DIAG_NUM_WORKERS" \
        BATCH_SIZE="$DIAG_BATCH_SIZE" \
        bash "$SCRIPT_DIR/launch_v276_stiffness_diagnostic_existing.sh"
      status="$?"
      printf "normdiag\t%s\t%s\n" "$status" "$DIAG_LOG_DIR" >> "$SUMMARY"
      if [ "$status" -ne 0 ]; then
        failed=1
      fi
      echo "END_STAGE|stage=normdiag|status=$status"
    else
      echo "SKIP_STAGE|stage=normdiag|reason=no_raw_strength_rows"
      printf "normdiag\tskipped\t%s\n" "$DIAG_LOG_DIR" >> "$SUMMARY"
    fi
    ;;
esac

echo "SUMMARY=$SUMMARY"
exit "$failed"
