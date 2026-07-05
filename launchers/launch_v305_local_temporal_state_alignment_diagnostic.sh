#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v305_local_temporal_state_alignment_diagnostic_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/$RUN_TAG}"

V303_DIR="${V303_DIR:-$ROOT_DIR/logs/v303_control_and_intraclass_diagnostic_20260702_120442}"
V304_DIR="${V304_DIR:-$ROOT_DIR/logs/v304_mechanism_reanalysis_20260703_190144}"
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-v303_control_and_intraclass_diagnostic_20260702_120442_control_runs}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT_DIR/outputs}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
RADIUS="${RADIUS:-2}"
TAU="${TAU:-0.1}"
MAX_SHIFT="${MAX_SHIFT:-8}"
MIN_SAMPLES_PER_CLASS_TIME="${MIN_SAMPLES_PER_CLASS_TIME:-5}"
MAX_SOURCE_SAMPLES="${MAX_SOURCE_SAMPLES:-0}"
MAX_TARGET_SAMPLES="${MAX_TARGET_SAMPLES:-0}"
RANDOM_REPEATS="${RANDOM_REPEATS:-50}"
SHUFFLE_REPEATS="${SHUFFLE_REPEATS:-50}"
WRITE_COST_MATRIX="${WRITE_COST_MATRIX:-False}"
AUDIT_SUMMARY_MODE="${AUDIT_SUMMARY_MODE:-False}"
AUDIT_SHUFFLE_REPEATS="${AUDIT_SHUFFLE_REPEATS:-100}"
AUDIT_BLOCK_SIZE="${AUDIT_BLOCK_SIZE:-3}"
AUDIT_BOOTSTRAP_REPEATS="${AUDIT_BOOTSTRAP_REPEATS:-1000}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"

mkdir -p "$OUTPUT_DIR"

echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "V303_DIR=$V303_DIR"
echo "V304_DIR=$V304_DIR"
echo "SOURCE_RUN_TAG=$SOURCE_RUN_TAG"
echo "OUTPUTS_ROOT=$OUTPUTS_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "CLOSED_SET=$CLOSED_SET"
echo "DEVICE=$DEVICE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "WINDOW_SIZE=$WINDOW_SIZE"
echo "RADIUS=$RADIUS"
echo "TAU=$TAU"
echo "MAX_SHIFT=$MAX_SHIFT"
echo "MIN_SAMPLES_PER_CLASS_TIME=$MIN_SAMPLES_PER_CLASS_TIME"
echo "MAX_SOURCE_SAMPLES=$MAX_SOURCE_SAMPLES"
echo "MAX_TARGET_SAMPLES=$MAX_TARGET_SAMPLES"
echo "RANDOM_REPEATS=$RANDOM_REPEATS"
echo "SHUFFLE_REPEATS=$SHUFFLE_REPEATS"
echo "WRITE_COST_MATRIX=$WRITE_COST_MATRIX"
echo "AUDIT_SUMMARY_MODE=$AUDIT_SUMMARY_MODE"
echo "AUDIT_SHUFFLE_REPEATS=$AUDIT_SHUFFLE_REPEATS"
echo "AUDIT_BLOCK_SIZE=$AUDIT_BLOCK_SIZE"
echo "AUDIT_BOOTSTRAP_REPEATS=$AUDIT_BOOTSTRAP_REPEATS"
echo "LIMIT_ROWS=$LIMIT_ROWS"

START_SECONDS="$(date +%s)"
cd "$ROOT_DIR" || exit 2

python analysis/summarize_v305_local_temporal_state_alignment_diagnostic.py \
  --v303_dir "$V303_DIR" \
  --v304_dir "$V304_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --source_run_tag "$SOURCE_RUN_TAG" \
  --outputs_root "$OUTPUTS_ROOT" \
  --data_root "$DATA_ROOT" \
  --closed_set "$CLOSED_SET" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --window_size "$WINDOW_SIZE" \
  --radius "$RADIUS" \
  --tau "$TAU" \
  --max_shift "$MAX_SHIFT" \
  --min_samples_per_class_time "$MIN_SAMPLES_PER_CLASS_TIME" \
  --max_source_samples "$MAX_SOURCE_SAMPLES" \
  --max_target_samples "$MAX_TARGET_SAMPLES" \
  --random_repeats "$RANDOM_REPEATS" \
  --shuffle_repeats "$SHUFFLE_REPEATS" \
  --write_cost_matrix "$WRITE_COST_MATRIX" \
  --audit_summary_mode "$AUDIT_SUMMARY_MODE" \
  --audit_shuffle_repeats "$AUDIT_SHUFFLE_REPEATS" \
  --audit_block_size "$AUDIT_BLOCK_SIZE" \
  --audit_bootstrap_repeats "$AUDIT_BOOTSTRAP_REPEATS" \
  --limit_rows "$LIMIT_ROWS" \
  > "$OUTPUT_DIR/diagnostic.log" 2>&1

STATUS="$?"
END_SECONDS="$(date +%s)"
RUNTIME_SECONDS=$((END_SECONDS - START_SECONDS))
{
  echo "run_tag	total_runtime_seconds	status"
  echo "$RUN_TAG	$RUNTIME_SECONDS	$STATUS"
} > "$OUTPUT_DIR/runtime.tsv"

echo "Logs saved to: $OUTPUT_DIR"
echo "Total runtime seconds: $RUNTIME_SECONDS"
exit "$STATUS"
