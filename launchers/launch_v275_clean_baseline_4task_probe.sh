#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v275_clean_baseline_4task_probe}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR1_to_FR2,FR1_to_AT1,FR2_to_DK1,AT1_to_DK1}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-plain,v275_raw_w1}"
DRY_RUN="${DRY_RUN:-False}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-50}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
V275_WEIGHT="${V275_WEIGHT:-1.0}"

mkdir -p "$LOG_DIR"

JOBS="$LOG_DIR/jobs.tsv"
SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
MANIFEST="$LOG_DIR/config_manifest.tsv"
: > "$JOBS"
printf "config\tsource_train\tstructure_loss\tstructure_target\tstructure_weight\tda_structure\tpurpose\n" > "$MANIFEST"

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

add_manifest() {
  local config="$1"
  if grep -F -q "${config}"$'\t' "$MANIFEST"; then
    return
  fi
  case "$config" in
    plain)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "train_supervised" "none" "none" "0.0" "off" \
        "Clean TimeMatch baseline: original supervised source pretrain, closed-set, no shift augmentation."
      ;;
    v275_raw_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v275_raw_global_compactness" "raw" "$V275_WEIGHT" "off" \
        "v2.7.5: source-stage raw encoder compactness only; TimeMatch DA-stage structure off."
      ;;
  esac >> "$MANIFEST"
}

add_job() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$JOBS"
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
      case "$config" in
        plain|v275_raw_w1) ;;
        *)
          echo "ERROR unknown config: $config" >&2
          exit 2
          ;;
      esac
      add_manifest "$config"
      add_job "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$est_weight" "$RUN_TAG"
    done
  done
done

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
  local tag source_model timematch_model source_tile target_tile
  local set_tag

  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  case "$(echo "$CLOSED_SET" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) set_tag="closedset" ;;
    *) set_tag="openset" ;;
  esac
  tag="${RUN_TAG}_${task}_seed${seed}_${config}"
  source_model="pseltae_${source_tile}_${set_tag}_noshift_${tag}_source"
  timematch_model="timematch_${source_tile}_to_${target_tile}_${set_tag}_noshift_${tag}"

  cd "$ROOT_DIR" || return 2

  if [ "$config" = "plain" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set "$CLOSED_SET" \
      --with_shift_aug False \
      --epochs "$SOURCE_EPOCHS" \
      --num_workers "$NUM_WORKERS" \
      --seed "$seed" \
      -e "$source_model" \
      --source "$source_dataset" \
      --target "$source_dataset" || return "$?"
  else
    CUDA_VISIBLE_DEVICES="$gpu" python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set "$CLOSED_SET" \
      --with_shift_aug False \
      --source_phase_partition_mode uniform \
      --source_segment_partition_mode uniform \
      --source_phase_count 1 \
      --source_segment_count 1 \
      --source_structure_loss_version v275_raw_global_compactness \
      --source_structure_feature_target raw \
      --source_structure_detach_features False \
      --source_structure_intra_trade_off "$V275_WEIGHT" \
      --source_structure_amplitude_trade_off 0.0 \
      --source_structure_interphase_trade_off 0.0 \
      --source_structure_shape_trade_off 0.0 \
      --source_structure_trend_trade_off 0.0 \
      --source_structure_season_trade_off 0.0 \
      --source_structure_segment_inter_trade_off 0.0 \
      --source_structure_boundary_window_trade_off 0.0 \
      --source_structure_compact_distance mse \
      --epochs "$SOURCE_EPOCHS" \
      --num_workers "$NUM_WORKERS" \
      --seed "$seed" \
      -e "$source_model" \
      --source "$source_dataset" \
      --target "$source_dataset" \
      sourcephasecompact || return "$?"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    --data_root "$DATA_ROOT" \
    --closed_set "$CLOSED_SET" \
    --with_shift_aug False \
    --num_workers "$NUM_WORKERS" \
    --seed "$seed" \
    -e "$source_model" \
    --source "$source_dataset" \
    --target "$target_dataset" \
    --eval || return "$?"

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
  local task source_dataset target_dataset seed config est_weight run_tag log_file status

  while IFS=$'\t' read -r task source_dataset target_dataset seed config est_weight run_tag; do
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

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CONFIGS=$CONFIGS"
echo "CLOSED_SET=$CLOSED_SET"
echo "SOURCE_EPOCHS=$SOURCE_EPOCHS"
echo "DA_EPOCHS=$DA_EPOCHS"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "V275_WEIGHT=$V275_WEIGHT"
echo "JOBS=$(wc -l < "$JOBS")"
echo "MANIFEST=$MANIFEST"

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

python "$ROOT_DIR/analysis/summarize_v243b_raw_strength_intervention.py" "$LOG_DIR" || failed=1

echo "Logs saved to: $LOG_DIR"
exit "$failed"
