#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_RUN_TAG="${SOURCE_RUN_TAG:?SOURCE_RUN_TAG is required; this launcher is DA-only and will not train source models}"
RUN_TAG="${RUN_TAG:-v31_adaptive_stage_shift_contrast_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR1_to_FR2,AT1_to_DK1,FR2_to_AT1,FR2_to_FR1}"
SEEDS="${SEEDS:-1}"
SOURCE_CONFIGS="${SOURCE_CONFIGS:-plain}"
STAGE_TRADE_OFFS="${STAGE_TRADE_OFFS:-0.05}"
DRY_RUN="${DRY_RUN:-False}"
RUN_UNIT_TEST="${RUN_UNIT_TEST:-True}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
STAGE_COUNT="${STAGE_COUNT:-6}"
STAGE_MIN_LEN="${STAGE_MIN_LEN:-2}"
STAGE_TIME_RADIUS="${STAGE_TIME_RADIUS:-30.0}"
STAGE_TIME_TEMPERATURE="${STAGE_TIME_TEMPERATURE:-10.0}"
STAGE_CONTRAST_TEMPERATURE="${STAGE_CONTRAST_TEMPERATURE:-0.1}"
STAGE_PSEUDO_THRESHOLD="${STAGE_PSEUDO_THRESHOLD:-}"

mkdir -p "$LOG_DIR"

JOBS="$LOG_DIR/jobs.tsv"
SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
STATUS="$LOG_DIR/status.tsv"
: > "$JOBS"
printf "event\tgpu\ttask\tseed\tsource_config\tstage_trade_off\tstatus\tlog\n" > "$STATUS"

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

source_model_name() {
  local source_dataset="$1"
  local task="$2"
  local seed="$3"
  local source_config="$4"
  local source_tile set_tag
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  case "$(echo "$CLOSED_SET" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) set_tag="closedset" ;;
    *) set_tag="openset" ;;
  esac
  echo "pseltae_${source_tile}_${set_tag}_noshift_${SOURCE_RUN_TAG}_${task}_seed${seed}_${source_config}_source"
}

IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
IFS=',' read -r -a SOURCE_CONFIG_NAMES <<< "$SOURCE_CONFIGS"
IFS=',' read -r -a STAGE_TRADE_OFF_VALUES <<< "$STAGE_TRADE_OFFS"

missing=0
for seed in $SEEDS; do
  for task in "${TASK_NAMES[@]}"; do
    task="$(echo "$task" | xargs)"
    spec="$(task_spec "$task")" || exit 2
    read -r source_dataset target_dataset est_weight <<< "$spec"
    for source_config in "${SOURCE_CONFIG_NAMES[@]}"; do
      source_config="$(echo "$source_config" | xargs)"
      source_model="$(source_model_name "$source_dataset" "$task" "$seed" "$source_config")"
      source_path="$ROOT_DIR/outputs/$source_model/fold_0/model.pt"
      if [ ! -f "$source_path" ]; then
        echo "MISSING_SOURCE|task=$task|seed=$seed|source_config=$source_config|path=$source_path" >&2
        missing=1
        continue
      fi
      for trade_off in "${STAGE_TRADE_OFF_VALUES[@]}"; do
        trade_off="$(echo "$trade_off" | xargs)"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$task" "$source_dataset" "$target_dataset" "$seed" "$source_config" "$trade_off" "$est_weight" >> "$JOBS"
      done
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

sort -t $'\t' -k7,7nr "$JOBS" > "$SORTED_JOBS"
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
  local source_config="$6"
  local trade_off="$7"
  local source_tile target_tile source_model timematch_model stage_log set_tag pseudo_args=()

  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  source_model="$(source_model_name "$source_dataset" "$task" "$seed" "$source_config")"
  case "$(echo "$CLOSED_SET" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) set_tag="closedset" ;;
    *) set_tag="openset" ;;
  esac
  trade_tag="$(echo "$trade_off" | sed 's/\./p/g')"
  timematch_model="timematch_${source_tile}_to_${target_tile}_${set_tag}_v31_${RUN_TAG}_${task}_seed${seed}_${source_config}_stage${trade_tag}"
  stage_log="$LOG_DIR/stage_${task}_seed${seed}_${source_config}_stage${trade_tag}.tsv"
  if [ -n "$STAGE_PSEUDO_THRESHOLD" ]; then
    pseudo_args=(--stage_contrast_pseudo_threshold "$STAGE_PSEUDO_THRESHOLD")
  fi

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
    --weights "outputs/$source_model" \
    --stage_contrast_trade_off "$trade_off" \
    --stage_contrast_stage_count "$STAGE_COUNT" \
    --stage_partition_mode feature_change_dp \
    --stage_min_len "$STAGE_MIN_LEN" \
    --stage_time_radius "$STAGE_TIME_RADIUS" \
    --stage_time_temperature "$STAGE_TIME_TEMPERATURE" \
    --stage_contrast_temperature "$STAGE_CONTRAST_TEMPERATURE" \
    --stage_contrast_feature_kind spatial \
    --stage_contrast_log_path "$stage_log" \
    "${pseudo_args[@]}"
}

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/queue_gpu${gpu}.tsv"
  local worker_failed=0
  local task source_dataset target_dataset seed source_config trade_off est_weight log_file status

  while IFS=$'\t' read -r task source_dataset target_dataset seed source_config trade_off est_weight; do
    [ -z "$task" ] && continue
    trade_tag="$(echo "$trade_off" | sed 's/\./p/g')"
    log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${source_config}_stage${trade_tag}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|source_config=$source_config|stage_trade_off=$trade_off|log=$log_file"
    printf "START\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$gpu" "$task" "$seed" "$source_config" "$trade_off" "running" "$log_file" >> "$STATUS"
    run_job "$gpu" "$task" "$source_dataset" "$target_dataset" "$seed" "$source_config" "$trade_off" > "$log_file" 2>&1
    status="$?"
    if [ "$status" -eq 0 ]; then
      echo "DONE|gpu=$gpu|task=$task|seed=$seed|source_config=$source_config|stage_trade_off=$trade_off"
      printf "DONE\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$gpu" "$task" "$seed" "$source_config" "$trade_off" "ok" "$log_file" >> "$STATUS"
    else
      echo "FAIL|gpu=$gpu|task=$task|seed=$seed|source_config=$source_config|stage_trade_off=$trade_off|status=$status"
      printf "FAIL\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$gpu" "$task" "$seed" "$source_config" "$trade_off" "$status" "$log_file" >> "$STATUS"
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
echo "SOURCE_CONFIGS=$SOURCE_CONFIGS"
echo "STAGE_TRADE_OFFS=$STAGE_TRADE_OFFS"
echo "GPUS=$GPUS"
echo "JOBS=$(wc -l < "$JOBS")"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "DRY_RUN=True"
    exit 0
    ;;
esac

failed=0
case "$(echo "$RUN_UNIT_TEST" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    python "$ROOT_DIR/analysis/test_v31_adaptive_stage_contrast.py" > "$LOG_DIR/unit_test.log" 2>&1
    if [ "$?" -ne 0 ]; then
      echo "FAIL_UNIT_TEST|log=$LOG_DIR/unit_test.log"
      exit 1
    fi
    echo "DONE_UNIT_TEST|log=$LOG_DIR/unit_test.log"
    ;;
esac

pids=()
for gpu in "${GPU_IDS[@]}"; do
  run_worker "$gpu" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ -f "$ROOT_DIR/analysis/summarize_v275_da_only.py" ]; then
  python "$ROOT_DIR/analysis/summarize_v275_da_only.py" "$LOG_DIR" || failed=1
fi

echo "Logs saved to: $LOG_DIR"
exit "$failed"
