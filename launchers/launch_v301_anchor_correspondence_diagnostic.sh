#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v301_anchor_correspondence_diagnostic_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$RUN_TAG}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR1_to_FR2,AT1_to_DK1,FR2_to_AT1,DK1_to_AT1,FR2_to_FR1,AT1_to_FR2}"
SEEDS="${SEEDS:-1 2 3}"
CHECKPOINT_CONFIGS="${CHECKPOINT_CONFIGS:-plain,smooth_k3}"
DRY_RUN="${DRY_RUN:-False}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_PIXELS="${NUM_PIXELS:-64}"
K="${K:-6}"
MAX_TEMPORAL_SHIFT="${MAX_TEMPORAL_SHIFT:-60}"
CV_FOLDS="${CV_FOLDS:-5}"
MAX_SOURCE_SAMPLES="${MAX_SOURCE_SAMPLES:-0}"
MAX_TARGET_SAMPLES="${MAX_TARGET_SAMPLES:-0}"
MAX_CODEBOOK_TIMEPOINTS="${MAX_CODEBOOK_TIMEPOINTS:-120000}"
MAX_SILHOUETTE_TIMEPOINTS="${MAX_SILHOUETTE_TIMEPOINTS:-5000}"
MAX_STABILITY_POINTS="${MAX_STABILITY_POINTS:-12000}"

# Important: this is the source experiment RUN_TAG embedded in output names,
# not the timestamped logs directory name.
PLAIN_SOURCE_RUN_TAG="${PLAIN_SOURCE_RUN_TAG:-v275_closedset_baseline_v275_12tasks_3seeds}"
PLAIN_CONFIG_NAME="${PLAIN_CONFIG_NAME:-plain}"
SMOOTH_SOURCE_RUN_TAG="${SMOOTH_SOURCE_RUN_TAG:-v281_a_group_full_20260622_121128_full12}"
SMOOTH_CONFIG_NAME="${SMOOTH_CONFIG_NAME:-v276_smooth_k3_w1}"
V300_LOG_DIR="${V300_LOG_DIR:-$ROOT_DIR/logs/v300_global_shift_assumption_diagnostic_20260630_093953}"

mkdir -p "$LOG_DIR" "$LOG_DIR/runs"

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

set_tag() {
  case "$(echo "$CLOSED_SET" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) echo "closedset" ;;
    *) echo "openset" ;;
  esac
}

source_model_name() {
  local source_dataset="$1"
  local task="$2"
  local seed="$3"
  local checkpoint_config="$4"
  local source_tile run_tag config_name
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  if [ "$checkpoint_config" = "plain" ]; then
    run_tag="$PLAIN_SOURCE_RUN_TAG"
    config_name="$PLAIN_CONFIG_NAME"
  else
    run_tag="$SMOOTH_SOURCE_RUN_TAG"
    config_name="$SMOOTH_CONFIG_NAME"
  fi
  echo "pseltae_${source_tile}_$(set_tag)_noshift_${run_tag}_${task}_seed${seed}_${config_name}_source"
}

write_jobs() {
  local jobs="$LOG_DIR/jobs.tsv"
  : > "$jobs"
  IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
  IFS=',' read -r -a CONFIG_NAMES <<< "$CHECKPOINT_CONFIGS"
  for seed in $SEEDS; do
    for task in "${TASK_NAMES[@]}"; do
      task="$(echo "$task" | xargs)"
      spec="$(task_spec "$task")" || exit 2
      read -r source_dataset target_dataset est_weight <<< "$spec"
      for checkpoint_config in "${CONFIG_NAMES[@]}"; do
        checkpoint_config="$(echo "$checkpoint_config" | xargs)"
        case "$checkpoint_config" in
          plain|smooth_k3) ;;
          *)
            echo "ERROR unknown checkpoint_config: $checkpoint_config" >&2
            exit 2
            ;;
        esac
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$task" "$source_dataset" "$target_dataset" "$seed" "$checkpoint_config" "$est_weight" >> "$jobs"
      done
    done
  done
  sort -t $'\t' -k6,6nr "$jobs" > "$LOG_DIR/jobs_sorted.tsv"
}

split_queues() {
  local job_index=0 gpu
  for gpu in "${GPU_IDS[@]}"; do
    : > "$LOG_DIR/queue_gpu${gpu}.tsv"
  done
  while IFS= read -r line; do
    gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
    printf "%s\n" "$line" >> "$LOG_DIR/queue_gpu${gpu}.tsv"
    job_index=$((job_index + 1))
  done < "$LOG_DIR/jobs_sorted.tsv"
}

run_job() {
  local gpu="$1" task="$2" source_dataset="$3" target_dataset="$4" seed="$5" checkpoint_config="$6"
  local source_model run_dir checkpoint_path
  source_model="$(source_model_name "$source_dataset" "$task" "$seed" "$checkpoint_config")"
  checkpoint_path="$ROOT_DIR/outputs/$source_model/fold_0/model.pt"
  run_dir="$LOG_DIR/runs/${task}/seed${seed}_${checkpoint_config}"
  mkdir -p "$run_dir"
  cd "$ROOT_DIR" || return 2
  if [ ! -f "$checkpoint_path" ]; then
    printf "task\tsource\ttarget\tseed\tcheckpoint_config\tK\tstatus\terror\n" > "$run_dir/failed_runs.tsv"
    printf "%s\t%s\t%s\t%s\t%s\t%s\tmissing_checkpoint\t%s\n" \
      "$task" "$source_dataset" "$target_dataset" "$seed" "$checkpoint_config" "$K" "$checkpoint_path" >> "$run_dir/failed_runs.tsv"
    echo "MISSING_CHECKPOINT|task=$task|seed=$seed|config=$checkpoint_config|path=$checkpoint_path"
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python analysis/v301_anchor_correspondence_diagnostic.py \
    --mode run \
    --data_root "$DATA_ROOT" \
    --closed_set "$CLOSED_SET" \
    --source "$source_dataset" \
    --target "$target_dataset" \
    --task "$task" \
    --seed "$seed" \
    --checkpoint_config "$checkpoint_config" \
    --checkpoint_path "$checkpoint_path" \
    --output_dir "$run_dir" \
    --v300_log_dir "$V300_LOG_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_pixels "$NUM_PIXELS" \
    --K "$K" \
    --max_temporal_shift "$MAX_TEMPORAL_SHIFT" \
    --cv_folds "$CV_FOLDS" \
    --max_source_samples "$MAX_SOURCE_SAMPLES" \
    --max_target_samples "$MAX_TARGET_SAMPLES" \
    --max_codebook_timepoints "$MAX_CODEBOOK_TIMEPOINTS" \
    --max_silhouette_timepoints "$MAX_SILHOUETTE_TIMEPOINTS" \
    --max_stability_points "$MAX_STABILITY_POINTS"
}

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/queue_gpu${gpu}.tsv"
  local task source_dataset target_dataset seed checkpoint_config est_weight log_file status failed=0
  while IFS=$'\t' read -r task source_dataset target_dataset seed checkpoint_config est_weight; do
    [ -z "$task" ] && continue
    log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${checkpoint_config}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$checkpoint_config|log=$log_file"
    run_job "$gpu" "$task" "$source_dataset" "$target_dataset" "$seed" "$checkpoint_config" > "$log_file" 2>&1
    status="$?"
    if [ "$status" -eq 0 ]; then
      echo "DONE|gpu=$gpu|task=$task|seed=$seed|config=$checkpoint_config"
    else
      echo "FAIL|gpu=$gpu|task=$task|seed=$seed|config=$checkpoint_config|status=$status"
      failed=1
    fi
  done < "$queue"
  return "$failed"
}

write_jobs
split_queues

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CHECKPOINT_CONFIGS=$CHECKPOINT_CONFIGS"
echo "PLAIN_SOURCE_RUN_TAG=$PLAIN_SOURCE_RUN_TAG"
echo "SMOOTH_SOURCE_RUN_TAG=$SMOOTH_SOURCE_RUN_TAG"
echo "V300_LOG_DIR=$V300_LOG_DIR"
echo "JOBS=$(wc -l < "$LOG_DIR/jobs.tsv")"

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

python "$ROOT_DIR/analysis/v301_anchor_correspondence_diagnostic.py" \
  --mode aggregate \
  --input_root "$LOG_DIR/runs" \
  --output_dir "$LOG_DIR" || failed=1

echo "Logs saved to: $LOG_DIR"
exit "$failed"
