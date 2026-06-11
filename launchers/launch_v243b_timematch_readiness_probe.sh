#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v243b_timematch_readiness}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR2_to_FR1,DK1_to_FR1}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-plain,raw_global_w1_source_only}"
SOURCE_TAG_PREFIX="${SOURCE_TAG_PREFIX:-v243b_stage}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"

mkdir -p "$LOG_DIR"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

JOBS="$LOG_DIR/jobs.tsv"
: > "$JOBS"

task_spec() {
  case "$1" in
    FR1_to_FR2) echo "france/30TXT/2017 france/31TCJ/2017 3" ;;
    FR1_to_DK1) echo "france/30TXT/2017 denmark/32VNH/2017 3" ;;
    FR1_to_AT1) echo "france/30TXT/2017 austria/33UVP/2017 3" ;;
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

source_model_name() {
  local source_dataset="$1"
  local task="$2"
  local seed="$3"
  local config="$4"
  local source_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  echo "pseltae_${source_tile}_closedset_noshift_sourcephasecompact_p5_${SOURCE_TAG_PREFIX}_${task}_seed${seed}_${config}"
}

output_model_dir() {
  local model_name="$1"
  case "$OUTPUTS_ROOT" in
    /*) echo "$OUTPUTS_ROOT/$model_name" ;;
    *) echo "$ROOT_DIR/$OUTPUTS_ROOT/$model_name" ;;
  esac
}

timematch_model_name() {
  local source_dataset="$1"
  local target_dataset="$2"
  local task="$3"
  local seed="$4"
  local config="$5"
  local source_tile target_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  echo "timematch_${source_tile}_to_${target_tile}_closedset_readiness_${RUN_TAG}_${task}_seed${seed}_${config}"
}

IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
IFS=',' read -r -a CONFIG_NAMES <<< "$CONFIGS"
for seed in $SEEDS; do
  for task in "${TASK_NAMES[@]}"; do
    task="$(echo "$task" | xargs)"
    spec="$(task_spec "$task")" || exit 2
    read -r source_dataset target_dataset est_weight <<< "$spec"
    for config in "${CONFIG_NAMES[@]}"; do
      config="$(echo "$config" | xargs)"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$est_weight" >> "$JOBS"
    done
  done
done

for gpu in "${GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${gpu}.tsv"
done

SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
sort -t $'\t' -k6,6nr "$JOBS" > "$SORTED_JOBS"
job_index=0
while IFS= read -r line; do
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  printf "%s\n" "$line" >> "$LOG_DIR/queue_gpu${gpu}.tsv"
  job_index=$((job_index + 1))
done < "$SORTED_JOBS"

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/queue_gpu${gpu}.tsv"
  local worker_failed=0
  local task source_dataset target_dataset seed config est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config est_weight; do
    [ -z "$task" ] && continue
    local source_model timematch_model log_file
    source_model="$(source_model_name "$source_dataset" "$task" "$seed" "$config")"
    timematch_model="$(timematch_model_name "$source_dataset" "$target_dataset" "$task" "$seed" "$config")"
    log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"

    local source_model_dir
    source_model_dir="$(output_model_dir "$source_model")"
    if [ ! -f "$source_model_dir/fold_0/model.pt" ]; then
      echo "MISS|gpu=$gpu|task=$task|seed=$seed|config=$config|source=$source_model" | tee "$log_file"
      worker_failed=1
      continue
    fi

    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|source=$source_model|timematch=$timematch_model|log=$log_file"
    (
      common_args=(
        --data_root "$DATA_ROOT"
        --output_dir "$OUTPUTS_ROOT"
        --closed_set True
        --with_shift_aug False
        --source_feature_reshaper none
        --source_feature_dual_path False
        --source_phase_partition_mode uniform
        --source_segment_partition_mode uniform
        --source_phase_count 1
        --source_segment_count 1
        --source_structure_loss_version segment_boundary_window_residual
        --source_structure_feature_target raw
        --source_structure_intra_trade_off 0.0
        --source_structure_trend_trade_off 0.0
        --source_structure_segment_inter_trade_off 0.0
        --source_structure_boundary_window_trade_off 0.0
        --num_workers "$NUM_WORKERS"
        --seed "$seed"
        --source "$source_dataset"
        --target "$target_dataset"
      )

      CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT_DIR/train.py" \
        "${common_args[@]}" \
        -e "$source_model" \
        --eval

      CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT_DIR/train.py" \
        "${common_args[@]}" \
        -e "$timematch_model" \
        timematch \
        --epochs "$TIMEMATCH_EPOCHS" \
        --steps_per_epoch "$STEPS_PER_EPOCH" \
        --weights "$source_model_dir"
    ) > "$log_file" 2>&1

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

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CONFIGS=$CONFIGS"
echo "SOURCE_TAG_PREFIX=$SOURCE_TAG_PREFIX"
echo "TIMEMATCH_EPOCHS=$TIMEMATCH_EPOCHS"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"

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

python "$ROOT_DIR/analysis/summarize_v243b_timematch_readiness.py" \
  "$LOG_DIR" plain raw_global_w1_source_only || failed=1

echo "Logs saved to: $LOG_DIR"
exit "$failed"
