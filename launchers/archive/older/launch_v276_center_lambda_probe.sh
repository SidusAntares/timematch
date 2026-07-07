#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MASTER_TAG="${MASTER_TAG:-v276_center_lambda_probe_$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-$ROOT_DIR/logs/$MASTER_TAG}"

TASKS="${TASKS:-FR2_to_FR1,FR2_to_DK1,AT1_to_DK1}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-plain,v275_raw_w1,v276_timepoint_w1,v276_smoothed_timepoint_w1}"
WEIGHTS="${WEIGHTS:-0.5 1.0 2.0}"
REPEAT_PLAIN="${REPEAT_PLAIN:-False}"
GPUS="${GPUS:-0 1 2 3}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"

mkdir -p "$MASTER_LOG_DIR"

SUMMARY="$MASTER_LOG_DIR/controller_summary.tsv"
printf "weight\tstatus\tlog_dir\n" > "$SUMMARY"

weight_tag() {
  echo "$1" | sed 's/\./p/g'
}

echo "MASTER_TAG=$MASTER_TAG"
echo "MASTER_LOG_DIR=$MASTER_LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CONFIGS=$CONFIGS"
echo "WEIGHTS=$WEIGHTS"
echo "REPEAT_PLAIN=$REPEAT_PLAIN"
echo "GPUS=$GPUS"
echo "CLOSED_SET=$CLOSED_SET"
echo "SOURCE_EPOCHS=$SOURCE_EPOCHS"
echo "DA_EPOCHS=$DA_EPOCHS"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"

weight_index=0
for weight in $WEIGHTS; do
  tag="$(weight_tag "$weight")"
  sub_log_dir="$MASTER_LOG_DIR/w${tag}"
  run_configs="$CONFIGS"
  if [ "$weight_index" -gt 0 ] && [ "$(echo "$REPEAT_PLAIN" | tr '[:upper:]' '[:lower:]')" != "true" ]; then
    run_configs="$(echo "$CONFIGS" | sed 's/^plain,//; s/,plain,/,/g; s/,plain$//')"
  fi
  mkdir -p "$sub_log_dir"
  echo "START_WEIGHT|weight=$weight|configs=$run_configs|log_dir=$sub_log_dir"
  env \
    RUN_TAG="${MASTER_TAG}_w${tag}" \
    LOG_DIR="$sub_log_dir" \
    TASKS="$TASKS" \
    SEEDS="$SEEDS" \
    CONFIGS="$run_configs" \
    GPUS="$GPUS" \
    CLOSED_SET="$CLOSED_SET" \
    SOURCE_EPOCHS="$SOURCE_EPOCHS" \
    DA_EPOCHS="$DA_EPOCHS" \
    STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
    V275_WEIGHT="$weight" \
    bash "$SCRIPT_DIR/launch_v276_fr2_center_probe.sh"
  status="$?"
  printf "%s\t%s\t%s\n" "$weight" "$status" "$sub_log_dir" >> "$SUMMARY"
  echo "END_WEIGHT|weight=$weight|status=$status"
  weight_index=$((weight_index + 1))
done

echo "SUMMARY=$SUMMARY"
