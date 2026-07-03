#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MASTER_TAG="${MASTER_TAG:-v303_control_and_intraclass_diagnostic_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$MASTER_TAG}"
CONTROL_RUN_TAG="${CONTROL_RUN_TAG:-${MASTER_TAG}_control_runs}"
CONTROL_LOG_DIR="${CONTROL_LOG_DIR:-$LOG_DIR/control_runs}"

GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR1_to_FR2,AT1_to_DK1,FR2_to_AT1,DK1_to_AT1,FR2_to_FR1,AT1_to_FR2}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-plain,v276_smooth_k3_w1,v275_raw_w1,v303_time_permuted_smooth_k3_w1}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-60}"
QUEUE_SCHEDULE="${QUEUE_SCHEDULE:-interleave_configs}"

DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-128}"
DIAG_NUM_WORKERS="${DIAG_NUM_WORKERS:-8}"
DIAG_DEVICE="${DIAG_DEVICE:-cuda}"
K="${K:-6}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-10000}"
DRY_RUN="${DRY_RUN:-False}"
SKIP_CONTROL_RUNS="${SKIP_CONTROL_RUNS:-False}"

mkdir -p "$LOG_DIR" "$CONTROL_LOG_DIR"

START_SECONDS="$(date +%s)"

echo "MASTER_TAG=$MASTER_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "CONTROL_RUN_TAG=$CONTROL_RUN_TAG"
echo "CONTROL_LOG_DIR=$CONTROL_LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CONFIGS=$CONFIGS"
echo "GPUS=$GPUS"
echo "DATA_ROOT=$DATA_ROOT"
echo "CLOSED_SET=$CLOSED_SET"
echo "SOURCE_EPOCHS=$SOURCE_EPOCHS"
echo "DA_EPOCHS=$DA_EPOCHS"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "DATA_LOADER_TIMEOUT=$DATA_LOADER_TIMEOUT"
echo "QUEUE_SCHEDULE=$QUEUE_SCHEDULE"
echo "K=$K"
echo "SKIP_CONTROL_RUNS=$SKIP_CONTROL_RUNS"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "DRY_RUN=True"
    env \
      RUN_TAG="$CONTROL_RUN_TAG" \
      LOG_DIR="$CONTROL_LOG_DIR" \
      GPUS="$GPUS" \
      TASKS="$TASKS" \
      SEEDS="$SEEDS" \
      CONFIGS="$CONFIGS" \
      DATA_ROOT="$DATA_ROOT" \
      CLOSED_SET="$CLOSED_SET" \
      SOURCE_EPOCHS="$SOURCE_EPOCHS" \
      DA_EPOCHS="$DA_EPOCHS" \
      STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
      NUM_WORKERS="$NUM_WORKERS" \
      DATA_LOADER_TIMEOUT="$DATA_LOADER_TIMEOUT" \
      QUEUE_SCHEDULE="$QUEUE_SCHEDULE" \
      DRY_RUN=True \
      bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh"
    exit 0
    ;;
esac

failed=0

case "$(echo "$SKIP_CONTROL_RUNS" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "SKIP_CONTROL_RUNS=True"
    ;;
  *)
    env \
      RUN_TAG="$CONTROL_RUN_TAG" \
      LOG_DIR="$CONTROL_LOG_DIR" \
      GPUS="$GPUS" \
      TASKS="$TASKS" \
      SEEDS="$SEEDS" \
      CONFIGS="$CONFIGS" \
      DATA_ROOT="$DATA_ROOT" \
      CLOSED_SET="$CLOSED_SET" \
      SOURCE_EPOCHS="$SOURCE_EPOCHS" \
      DA_EPOCHS="$DA_EPOCHS" \
      STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
      NUM_WORKERS="$NUM_WORKERS" \
      DATA_LOADER_TIMEOUT="$DATA_LOADER_TIMEOUT" \
      QUEUE_SCHEDULE="$QUEUE_SCHEDULE" \
      bash "$SCRIPT_DIR/launch_v275_clean_baseline_4task_probe.sh" || failed=1
    ;;
esac

cd "$ROOT_DIR" || exit 2

python analysis/v303_control_and_intraclass_diagnostic.py \
  --log_dir "$CONTROL_LOG_DIR" \
  --output_dir "$LOG_DIR" \
  --source_run_tag "$CONTROL_RUN_TAG" \
  --data_root "$DATA_ROOT" \
  --outputs_root "$ROOT_DIR/outputs" \
  --closed_set "$CLOSED_SET" \
  --batch_size "$DIAG_BATCH_SIZE" \
  --num_workers "$DIAG_NUM_WORKERS" \
  --device "$DIAG_DEVICE" \
  --K "$K" \
  --bootstrap_repeats "$BOOTSTRAP_REPEATS" \
  > "$LOG_DIR/diagnostic.log" 2>&1 || failed=1

END_SECONDS="$(date +%s)"
RUNTIME_SECONDS=$((END_SECONDS - START_SECONDS))
{
  echo "master_tag	total_runtime_seconds	status"
  echo "$MASTER_TAG	$RUNTIME_SECONDS	$failed"
} > "$LOG_DIR/runtime.tsv"

echo "Logs saved to: $LOG_DIR"
echo "Control run logs: $CONTROL_LOG_DIR"
echo "Total runtime seconds: $RUNTIME_SECONDS"

exit "$failed"
