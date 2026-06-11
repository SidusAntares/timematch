#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_LOG_DIR="${SOURCE_LOG_DIR:?SOURCE_LOG_DIR is required, e.g. logs/v243b_stage_timing_fix_20260610_224858_20260610_224858}"
RUN_TAG="${RUN_TAG:-v243b_paired_geometry_chain}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

TASKS="${TASKS:-FR2_to_FR1,DK1_to_FR1}"
SEEDS="${SEEDS:-1,2,3}"
BASE_CONFIG="${BASE_CONFIG:-plain}"
SHAPED_CONFIG="${SHAPED_CONFIG:-raw_global_w1_source_only}"
SOURCE_TAG_PREFIX="${SOURCE_TAG_PREFIX:-v243b_stage}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"

mkdir -p "$LOG_DIR"

JOBS_TSV="$SOURCE_LOG_DIR/jobs.tsv"
SUMMARY_TSV="$SOURCE_LOG_DIR/summary.tsv"

if [ ! -f "$JOBS_TSV" ]; then
  echo "ERROR: jobs.tsv not found: $JOBS_TSV" >&2
  exit 2
fi

if [ ! -f "$SUMMARY_TSV" ]; then
  echo "summary.tsv not found; trying to summarize SOURCE_LOG_DIR first: $SOURCE_LOG_DIR"
  python "$ROOT_DIR/analysis/summarize_v243b_stage_timing_counterfactual.py" "$SOURCE_LOG_DIR"
fi

if [ ! -f "$SUMMARY_TSV" ]; then
  echo "ERROR: summary.tsv not found after summarization: $SUMMARY_TSV" >&2
  exit 2
fi

echo "RUN_TAG=$RUN_TAG"
echo "SOURCE_LOG_DIR=$SOURCE_LOG_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "BASE_CONFIG=$BASE_CONFIG"
echo "SHAPED_CONFIG=$SHAPED_CONFIG"
echo "SOURCE_TAG_PREFIX=$SOURCE_TAG_PREFIX"
echo "MAX_BATCHES=$MAX_BATCHES"

python "$ROOT_DIR/analysis/v243b_paired_geometry_chain_diagnostic.py" \
  --jobs_tsv "$JOBS_TSV" \
  --summary_tsv "$SUMMARY_TSV" \
  --output_dir "$LOG_DIR" \
  --data_root "$DATA_ROOT" \
  --outputs_root "$OUTPUTS_ROOT" \
  --source_tag_prefix "$SOURCE_TAG_PREFIX" \
  --tasks "$TASKS" \
  --seeds "$SEEDS" \
  --base_config "$BASE_CONFIG" \
  --shaped_config "$SHAPED_CONFIG" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_batches "$MAX_BATCHES" \
  > "$LOG_DIR/paired_geometry_chain.log" 2>&1

status="$?"
if [ "$status" -eq 0 ]; then
  echo "Paired geometry chain diagnostic saved to: $LOG_DIR"
else
  echo "FAIL: paired geometry chain diagnostic failed, see $LOG_DIR/paired_geometry_chain.log" >&2
fi
exit "$status"
