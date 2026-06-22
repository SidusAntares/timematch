#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v276_stiffness_diagnostic_existing_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$RUN_TAG}"
GPUS="${GPUS:-0 1 2 3}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
TASKS="${TASKS:-}"
CONFIGS="${CONFIGS:-plain,raw_global_w0p50_source_only,raw_global_w0p75_source_only,raw_w1,smooth_w0p5,smooth_w1p0}"
MAX_SOURCE_BATCHES="${MAX_SOURCE_BATCHES:-0}"
MAX_TARGET_BATCHES="${MAX_TARGET_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DRY_RUN="${DRY_RUN:-False}"

RUN_SPECS_DEFAULT="logs/v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121/raw_strength_rows.tsv::logs/v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121:: logs/v276_closedset_strength_response_12tasks_3seeds_20260617_112641/raw_strength_rows.tsv::logs/v276_closedset_strength_response_12tasks_3seeds_20260617_112641:: logs/v276_smoothed_lambda12_half_20260619_215420/w0p5/raw_strength_rows.tsv::logs/v276_smoothed_lambda12_half_20260619_215420/w0p5::smooth_w0p5 logs/v276_smoothed_lambda12_half_20260619_215420/w1p0/raw_strength_rows.tsv::logs/v276_smoothed_lambda12_half_20260619_215420/w1p0::smooth_w1p0"
RUN_SPECS="${RUN_SPECS:-$RUN_SPECS_DEFAULT}"

mkdir -p "$LOG_DIR"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "GPUS=$GPUS"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUTS_ROOT=$OUTPUTS_ROOT"
echo "TASKS=$TASKS"
echo "CONFIGS=$CONFIGS"
echo "MAX_SOURCE_BATCHES=$MAX_SOURCE_BATCHES"
echo "MAX_TARGET_BATCHES=$MAX_TARGET_BATCHES"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "RUN_SPECS=$RUN_SPECS"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "DRY_RUN=True"
    exit 0
    ;;
esac

pids=()
shard_paths=()
num_shards="${#GPU_IDS[@]}"

for shard_index in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$shard_index]}"
  shard_dir="$LOG_DIR/shard_${shard_index}"
  mkdir -p "$shard_dir"
  shard_paths+=("$shard_dir/stiffness_diagnostic_rows.tsv")
  (
    cd "$ROOT_DIR" || exit 2
    cmd=(python analysis/v276_stiffness_diagnostic_existing.py
      --data_root "$DATA_ROOT"
      --outputs_root "$OUTPUTS_ROOT"
      --output_dir "$shard_dir"
      --tasks "$TASKS"
      --configs "$CONFIGS"
      --max_source_batches "$MAX_SOURCE_BATCHES"
      --max_target_batches "$MAX_TARGET_BATCHES"
      --num_workers "$NUM_WORKERS"
      --batch_size "$BATCH_SIZE"
      --device cuda
      --num_shards "$num_shards"
      --shard_index "$shard_index")
    for spec in $RUN_SPECS; do
      cmd+=(--run "$spec")
    done
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
  ) > "$LOG_DIR/shard_${shard_index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "ERROR: at least one diagnostic shard failed. See $LOG_DIR/shard_*.log" >&2
  exit "$failed"
fi

IFS=,
merged_inputs="${shard_paths[*]}"
unset IFS

(
  cd "$ROOT_DIR" || exit 2
  python analysis/v276_stiffness_diagnostic_existing.py \
    --output_dir "$LOG_DIR" \
    --merge_detail_tsvs "$merged_inputs"
) > "$LOG_DIR/merge.log" 2>&1 || exit "$?"

echo "Diagnostics saved to: $LOG_DIR"
echo "Rows: $LOG_DIR/stiffness_diagnostic_rows.tsv"
echo "Config summary: $LOG_DIR/stiffness_summary_by_config.tsv"
echo "Delta summary: $LOG_DIR/stiffness_delta_vs_plain_summary.tsv"
