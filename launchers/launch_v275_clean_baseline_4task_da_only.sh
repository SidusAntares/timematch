#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_RUN_TAG="${SOURCE_RUN_TAG:?SOURCE_RUN_TAG is required, e.g. v275_clean_baseline_4task_probe_20260614_163733}"
RUN_TAG="${RUN_TAG:-${SOURCE_RUN_TAG}_da_only}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR1_to_FR2,FR1_to_AT1,FR2_to_DK1,AT1_to_DK1}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-plain,v275_raw_w1}"
DRY_RUN="${DRY_RUN:-False}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"

mkdir -p "$LOG_DIR"

JOBS="$LOG_DIR/jobs.tsv"
SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
: > "$JOBS"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

task_spec() {
  case "$1" in
    FR1_to_FR2) echo "france/30TXT/2017 france/31TCJ/2017 4" ;;
    FR1_to_DK1) echo "france/30TXT/2017 denmark/32VNH/2017 4" ;;
    FR1_to_AT1) echo "france/30TXT/2017 austria/33UVP/2017 4" ;;
    FR2_to_FR1) echo "france/31TCJ/2017 france/30TXT/2017 3" ;;
    FR2_to_DK1) echo "france/31TCJ/2017 denmark/32VNH/2017 3" ;;
    FR2_to_AT1) echo "france/31TCJ/2017 austria/33UVP/2017 3" ;;
    DK1_to_FR1) echo "denmark/32VNH/2017 france/30TXT/2017 2" ;;
    DK1_to_FR2) echo "denmark/32VNH/2017 france/31TCJ/2017 2" ;;
    DK1_to_AT1) echo "denmark/32VNH/2017 austria/33UVP/2017 2" ;;
    AT1_to_FR1) echo "austria/33UVP/2017 france/30TXT/2017 2" ;;
    AT1_to_FR2) echo "austria/33UVP/2017 france/31TCJ/2017 2" ;;
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017 2" ;;
    *)
      echo "ERROR unknown task: $1" >&2
      return 1
      ;;
  esac
}

add_job() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" >> "$JOBS"
}

IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
IFS=',' read -r -a CONFIG_NAMES <<< "$CONFIGS"

missing=0
for seed in $SEEDS; do
  for task in "${TASK_NAMES[@]}"; do
    task="$(echo "$task" | xargs)"
    spec="$(task_spec "$task")" || exit 2
    read -r source_dataset target_dataset est_weight <<< "$spec"
    source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
    case "$(echo "$CLOSED_SET" | tr '[:upper:]' '[:lower:]')" in
      1|true|yes|y|on) set_tag="closedset" ;;
      *) set_tag="openset" ;;
    esac
    for config in "${CONFIG_NAMES[@]}"; do
      config="$(echo "$config" | xargs)"
      case "$config" in
        plain|v275_raw_w1) ;;
        *)
          echo "ERROR unknown config: $config" >&2
          exit 2
          ;;
      esac
      source_model="pseltae_${source_tile}_${set_tag}_noshift_${SOURCE_RUN_TAG}_${task}_seed${seed}_${config}_source"
      if [ ! -f "$ROOT_DIR/outputs/$source_model/fold_0/model.pt" ]; then
        echo "MISSING_SOURCE|task=$task|seed=$seed|config=$config|path=$ROOT_DIR/outputs/$source_model/fold_0/model.pt" >&2
        missing=1
        continue
      fi
      add_job "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$est_weight"
    done
  done
done

if [ "$missing" -ne 0 ]; then
  echo "ERROR: at least one required source checkpoint is missing. No DA jobs launched." >&2
  exit 2
fi

for gpu in "${GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${gpu}.tsv"
done

sort -t $'\t' -k6,6nr "$JOBS" > "$SORTED_JOBS"
job_index=0
while IFS= read -r line; do
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  printf "%s\n" "$line" >> "$LOG_DIR/queue_gpu${gpu}.tsv"
  job_index=$((job_index + 1))
done < "$SORTED_JOBS"

run_job() {
  local gpu="$1"
  local task="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local seed="$5"
  local config="$6"
  local source_tile target_tile source_model timematch_model set_tag

  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  case "$(echo "$CLOSED_SET" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) set_tag="closedset" ;;
    *) set_tag="openset" ;;
  esac
  source_model="pseltae_${source_tile}_${set_tag}_noshift_${SOURCE_RUN_TAG}_${task}_seed${seed}_${config}_source"
  timematch_model="timematch_${source_tile}_to_${target_tile}_${set_tag}_noshift_${RUN_TAG}_${task}_seed${seed}_${config}"

  cd "$ROOT_DIR" || return 2
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    --data_root "$DATA_ROOT" \
    --closed_set "$CLOSED_SET" \
    --with_shift_aug False \
    --num_workers "$NUM_WORKERS" \
    --seed "$seed" \
    -e "$timematch_model" \
    --source "$source_dataset" \
    --target "$target_dataset" \
    timematch \
    --epochs "$DA_EPOCHS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --weights "outputs/$source_model"
}

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/queue_gpu${gpu}.tsv"
  local worker_failed=0
  local task source_dataset target_dataset seed config est_weight log_file status

  while IFS=$'\t' read -r task source_dataset target_dataset seed config est_weight; do
    [ -z "$task" ] && continue
    log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|log=$log_file"
    run_job "$gpu" "$task" "$source_dataset" "$target_dataset" "$seed" "$config" > "$log_file" 2>&1
    status="$?"
    if [ "$status" -eq 0 ]; then
      echo "DONE|gpu=$gpu|task=$task|seed=$seed|config=$config"
    else
      echo "FAIL|gpu=$gpu|task=$task|seed=$seed|config=$config|status=$status"
      worker_failed=1
    fi
  done < "$queue"
  return "$worker_failed"
}

echo "SOURCE_RUN_TAG=$SOURCE_RUN_TAG"
echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CONFIGS=$CONFIGS"
echo "CLOSED_SET=$CLOSED_SET"
echo "DA_EPOCHS=$DA_EPOCHS"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "JOBS=$(wc -l < "$JOBS")"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "DRY_RUN=True"
    for gpu in "${GPU_IDS[@]}"; do
      echo "Queue gpu${gpu}: $LOG_DIR/queue_gpu${gpu}.tsv ($(wc -l < "$LOG_DIR/queue_gpu${gpu}.tsv") jobs)"
    done
    exit 0
    ;;
esac

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

python "$ROOT_DIR/analysis/summarize_v275_da_only.py" "$LOG_DIR" || failed=1

echo "Logs saved to: $LOG_DIR"
exit "$failed"
