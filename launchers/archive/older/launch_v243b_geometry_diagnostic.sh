#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_LOG_DIR="${SOURCE_LOG_DIR:?SOURCE_LOG_DIR is required, e.g. logs/v243b_cf_remote12_20260608_224435_20260608_224435}"
RUN_TAG="${RUN_TAG:-v243b_geometry_diag}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"

TASKS="${TASKS:-AT1_to_DK1,FR1_to_AT1,FR2_to_DK1,DK1_to_FR1}"
SEEDS="${SEEDS:-1,2,3}"
CONFIGS="${CONFIGS:-plain,raw_global_w1}"
MAX_BATCHES="${MAX_BATCHES:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"

mkdir -p "$LOG_DIR"

JOBS_TSV="$SOURCE_LOG_DIR/jobs.tsv"
CONTRAST_TSV="$SOURCE_LOG_DIR/counterfactual_contrasts.tsv"
if [ ! -f "$JOBS_TSV" ]; then
  echo "ERROR: jobs.tsv not found: $JOBS_TSV" >&2
  exit 2
fi
if [ ! -f "$CONTRAST_TSV" ]; then
  echo "ERROR: counterfactual_contrasts.tsv not found: $CONTRAST_TSV" >&2
  exit 2
fi

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

for gpu in "${GPU_IDS[@]}"; do
  : > "$LOG_DIR/jobs_gpu${gpu}.tsv"
done

job_index=0
while IFS= read -r line; do
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  printf "%s\n" "$line" >> "$LOG_DIR/jobs_gpu${gpu}.tsv"
  job_index=$((job_index + 1))
done < "$JOBS_TSV"

echo "RUN_TAG=$RUN_TAG"
echo "SOURCE_LOG_DIR=$SOURCE_LOG_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CONFIGS=$CONFIGS"
echo "MAX_BATCHES=$MAX_BATCHES"

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/jobs_gpu${gpu}.tsv"
  local out="$LOG_DIR/geometry_rows_gpu${gpu}.tsv"
  local worker_log="$LOG_DIR/gpu${gpu}_geometry_extract.log"
  echo "START|gpu=$gpu|queue=$queue|out=$out|log=$worker_log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT_DIR/analysis/v243b_representation_geometry_diagnostic.py" extract \
      --jobs_tsv "$queue" \
      --output_tsv "$out" \
      --data_root "$DATA_ROOT" \
      --outputs_root "$OUTPUTS_ROOT" \
      --tasks "$TASKS" \
      --seeds "$SEEDS" \
      --configs "$CONFIGS" \
      --device cuda \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --max_batches "$MAX_BATCHES"
  ) > "$worker_log" 2>&1
  local status="$?"
  if [ "$status" -eq 0 ]; then
    echo "DONE|gpu=$gpu"
  else
    echo "FAIL|gpu=$gpu|status=$status"
  fi
  return "$status"
}

pids=()
failed=0
for gpu in "${GPU_IDS[@]}"; do
  run_worker "$gpu" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

python "$ROOT_DIR/analysis/v243b_representation_geometry_diagnostic.py" merge \
  --metric_dir "$LOG_DIR" \
  --contrast_tsv "$CONTRAST_TSV" \
  --output_dir "$LOG_DIR" || failed=1

echo "Geometry diagnostic saved to: $LOG_DIR"
exit "$failed"
