#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v300_global_shift_assumption_diagnostic_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$RUN_TAG}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR1_to_FR2,AT1_to_DK1,FR2_to_AT1,DK1_to_AT1,FR2_to_FR1,AT1_to_FR2}"
SEEDS="${SEEDS:-1 2 3}"
DA_CONFIGS="${DA_CONFIGS:-original_timematch,no_shift,fixed_initial_shift,oracle_scalar_shift_diagnostic,topk_shift_ensemble_diagnostic}"
DRY_RUN="${DRY_RUN:-False}"
REUSE_SOURCE="${REUSE_SOURCE:-True}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-$RUN_TAG}"
SOURCE_CONFIG="${SOURCE_CONFIG:-smooth_const}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
MAX_TEMPORAL_SHIFT="${MAX_TEMPORAL_SHIFT:-60}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.9}"
TOPK_SHIFTS="${TOPK_SHIFTS:-3}"

mkdir -p "$LOG_DIR" "$LOG_DIR/offline" "$LOG_DIR/trajectory"

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
  local source_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  echo "pseltae_${source_tile}_$(set_tag)_noshift_${SOURCE_RUN_TAG}_${task}_seed${seed}_${SOURCE_CONFIG}_source"
}

write_phase_jobs() {
  local phase="$1"
  local jobs_file="$LOG_DIR/${phase}_jobs.tsv"
  : > "$jobs_file"
  IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
  IFS=',' read -r -a CONFIG_NAMES <<< "$DA_CONFIGS"
  for seed in $SEEDS; do
    for task in "${TASK_NAMES[@]}"; do
      task="$(echo "$task" | xargs)"
      spec="$(task_spec "$task")" || exit 2
      read -r source_dataset target_dataset est_weight <<< "$spec"
      if [ "$phase" = "da" ]; then
        for config in "${CONFIG_NAMES[@]}"; do
          config="$(echo "$config" | xargs)"
          printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$est_weight" >> "$jobs_file"
        done
      else
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$task" "$source_dataset" "$target_dataset" "$seed" "$phase" "$est_weight" >> "$jobs_file"
      fi
    done
  done
  sort -t $'\t' -k6,6nr "$jobs_file" > "$LOG_DIR/${phase}_jobs_sorted.tsv"
}

split_phase_queue() {
  local phase="$1"
  local sorted="$LOG_DIR/${phase}_jobs_sorted.tsv"
  local job_index=0
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    : > "$LOG_DIR/${phase}_queue_gpu${gpu}.tsv"
  done
  while IFS= read -r line; do
    gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
    printf "%s\n" "$line" >> "$LOG_DIR/${phase}_queue_gpu${gpu}.tsv"
    job_index=$((job_index + 1))
  done < "$sorted"
}

run_source_job() {
  local gpu="$1" task="$2" source_dataset="$3" target_dataset="$4" seed="$5"
  local model_name checkpoint
  model_name="$(source_model_name "$source_dataset" "$task" "$seed")"
  checkpoint="$ROOT_DIR/outputs/$model_name/fold_0/model.pt"
  cd "$ROOT_DIR" || return 2

  if [ -f "$checkpoint" ] && [ "$(echo "$REUSE_SOURCE" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
    echo "REUSE_SOURCE|task=$task|seed=$seed|model=$model_name"
    CUDA_VISIBLE_DEVICES="$gpu" python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set "$CLOSED_SET" \
      --with_shift_aug False \
      --num_workers "$NUM_WORKERS" \
      --seed "$seed" \
      -e "$model_name" \
      --source "$source_dataset" \
      --target "$source_dataset" \
      --eval || return "$?"
  else
    CUDA_VISIBLE_DEVICES="$gpu" python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set "$CLOSED_SET" \
      --with_shift_aug False \
      --source_phase_partition_mode uniform \
      --source_segment_partition_mode uniform \
      --source_phase_count 1 \
      --source_segment_count 1 \
      --source_structure_loss_version v276_raw_smoothed_timepoint_compactness \
      --source_structure_feature_target raw \
      --source_structure_detach_features False \
      --source_structure_intra_trade_off 1.0 \
      --source_structure_time_smooth_kernel_size 3 \
      --source_structure_lambda_schedule constant \
      --source_structure_lambda_base 1.0 \
      --source_structure_lambda_final 1.0 \
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
      -e "$model_name" \
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
    -e "$model_name" \
    --source "$source_dataset" \
    --target "$target_dataset" \
    --eval
}

run_offline_job() {
  local gpu="$1" task="$2" source_dataset="$3" target_dataset="$4" seed="$5"
  local model_name out_dir
  model_name="$(source_model_name "$source_dataset" "$task" "$seed")"
  out_dir="$LOG_DIR/offline/${task}_seed${seed}"
  cd "$ROOT_DIR" || return 2
  CUDA_VISIBLE_DEVICES="$gpu" python analysis/v300_global_shift_assumption_diagnostic.py \
    --data_root "$DATA_ROOT" \
    --closed_set "$CLOSED_SET" \
    --source "$source_dataset" \
    --target "$target_dataset" \
    --task "$task" \
    --seed "$seed" \
    --source_model "$model_name" \
    --output_dir "$out_dir" \
    --num_workers "$NUM_WORKERS" \
    --sample_size "$SAMPLE_SIZE" \
    --max_temporal_shift "$MAX_TEMPORAL_SHIFT" \
    --pseudo_threshold "$PSEUDO_THRESHOLD"
}

run_da_job() {
  local gpu="$1" task="$2" source_dataset="$3" target_dataset="$4" seed="$5" config="$6"
  local source_tile target_tile model_name timematch_model trajectory_path
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"
  model_name="$(source_model_name "$source_dataset" "$task" "$seed")"
  timematch_model="timematch_${source_tile}_to_${target_tile}_$(set_tag)_noshift_${RUN_TAG}_${task}_seed${seed}_${config}"
  trajectory_path="$LOG_DIR/trajectory/${task}_seed${seed}_${config}.tsv"
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
    --weights "outputs/$model_name" \
    --pseudo_threshold "$PSEUDO_THRESHOLD" \
    --max_temporal_shift "$MAX_TEMPORAL_SHIFT" \
    --sample_size "$SAMPLE_SIZE" \
    --timematch_shift_policy "$config" \
    --timematch_topk_shifts "$TOPK_SHIFTS" \
    --timematch_diagnostic_task "$task" \
    --timematch_diagnostic_log_path "$trajectory_path"
}

run_phase_worker() {
  local phase="$1" gpu="$2"
  local queue="$LOG_DIR/${phase}_queue_gpu${gpu}.tsv"
  local task source_dataset target_dataset seed config est_weight log_file status failed=0
  while IFS=$'\t' read -r task source_dataset target_dataset seed config est_weight; do
    [ -z "$task" ] && continue
    if [ "$phase" = "source" ]; then
      log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_source.log"
      echo "START_SOURCE|gpu=$gpu|task=$task|seed=$seed|log=$log_file"
      run_source_job "$gpu" "$task" "$source_dataset" "$target_dataset" "$seed" > "$log_file" 2>&1
    elif [ "$phase" = "offline" ]; then
      log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_offline.log"
      echo "START_OFFLINE|gpu=$gpu|task=$task|seed=$seed|log=$log_file"
      run_offline_job "$gpu" "$task" "$source_dataset" "$target_dataset" "$seed" > "$log_file" 2>&1
    else
      log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"
      echo "START_DA|gpu=$gpu|task=$task|seed=$seed|config=$config|log=$log_file"
      run_da_job "$gpu" "$task" "$source_dataset" "$target_dataset" "$seed" "$config" > "$log_file" 2>&1
    fi
    status="$?"
    if [ "$status" -eq 0 ]; then
      echo "DONE|phase=$phase|gpu=$gpu|task=$task|seed=$seed|config=$config"
    else
      echo "FAIL|phase=$phase|gpu=$gpu|task=$task|seed=$seed|config=$config|status=$status"
      failed=1
    fi
  done < "$queue"
  return "$failed"
}

run_phase() {
  local phase="$1"
  local pids=()
  local failed=0
  write_phase_jobs "$phase"
  split_phase_queue "$phase"
  for gpu in "${GPU_IDS[@]}"; do
    run_phase_worker "$phase" "$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "DA_CONFIGS=$DA_CONFIGS"
echo "SOURCE_RUN_TAG=$SOURCE_RUN_TAG"
echo "SOURCE_CONFIG=$SOURCE_CONFIG"
echo "REUSE_SOURCE=$REUSE_SOURCE"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    write_phase_jobs source
    write_phase_jobs offline
    write_phase_jobs da
    echo "DRY_RUN=True"
    echo "source jobs: $(wc -l < "$LOG_DIR/source_jobs.tsv")"
    echo "offline jobs: $(wc -l < "$LOG_DIR/offline_jobs.tsv")"
    echo "da jobs: $(wc -l < "$LOG_DIR/da_jobs.tsv")"
    exit 0
    ;;
esac

failed=0
run_phase source || failed=1
run_phase offline || failed=1
run_phase da || failed=1

python "$ROOT_DIR/analysis/summarize_v300_global_shift_assumption_diagnostic.py" "$LOG_DIR" || failed=1

echo "Logs saved to: $LOG_DIR"
exit "$failed"
