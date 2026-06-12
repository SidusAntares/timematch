#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-v243b_readiness_checkpoint_causal}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR2_to_FR1,AT1_to_FR2}"
SEEDS="${SEEDS:-1 2 3}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1,3,5,10,20,35,50}"
DRY_RUN="${DRY_RUN:-False}"
REUSE_SOURCE_CHECKPOINTS="${REUSE_SOURCE_CHECKPOINTS:-False}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-1}"
TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"

SOURCE_TAG_PREFIX="${SOURCE_TAG_PREFIX:-v243b_ckpt_causal}"
BASE_CONFIG="${BASE_CONFIG:-plain}"
SHAPED_CONFIG="${SHAPED_CONFIG:-raw_global_w1_source_only}"
CONFIGS="${CONFIGS:-${BASE_CONFIG},${SHAPED_CONFIG}}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

JOBS="$LOG_DIR/source_jobs.tsv"
MANIFEST="$LOG_DIR/config_manifest.tsv"
: > "$JOBS"
printf "config\tcompact_weight\tpurpose\n" > "$MANIFEST"

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

truthy() {
  case "$(echo "${1:-False}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
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

compact_weight_for_config() {
  case "$1" in
    "$BASE_CONFIG") echo "0.0" ;;
    "$SHAPED_CONFIG") echo "1.0" ;;
    *)
      echo "ERROR unknown config: $1" >&2
      return 1
      ;;
  esac
}

manifest_once() {
  local config="$1"
  local weight="$2"
  local purpose="$3"
  if grep -F -q "${config}"$'\t' "$MANIFEST"; then
    return
  fi
  printf "%s\t%s\t%s\n" "$config" "$weight" "$purpose" >> "$MANIFEST"
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
      compact_weight="$(compact_weight_for_config "$config")" || exit 2
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$compact_weight" "$est_weight" >> "$JOBS"
      if [ "$config" = "$BASE_CONFIG" ]; then
        manifest_once "$config" "$compact_weight" "Plain source checkpoint trajectory without source compactness."
      else
        manifest_once "$config" "$compact_weight" "Source-stage raw global compactness checkpoint trajectory."
      fi
    done
  done
done

for gpu in "${GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${gpu}.tsv"
done

SORTED_JOBS="$LOG_DIR/source_jobs_sorted.tsv"
sort -t $'\t' -k7,7nr "$JOBS" > "$SORTED_JOBS"
job_index=0
while IFS= read -r line; do
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  printf "%s\n" "$line" >> "$LOG_DIR/queue_gpu${gpu}.tsv"
  job_index=$((job_index + 1))
done < "$SORTED_JOBS"

run_timematch_epoch_probe() {
  local gpu="$1"
  local task="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local seed="$5"
  local config="$6"
  local source_model="$7"
  local epoch_text="$8"

  local epoch_num epoch_tag checkpoint alias_model alias_dir source_tile target_tile timematch_model log_file
  epoch_num="$(printf "%d" "$epoch_text")"
  epoch_tag="$(printf "%03d" "$epoch_num")"
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"

  checkpoint="$(output_model_dir "$source_model")/fold_0/source_epoch_checkpoints/epoch_${epoch_tag}.pt"
  alias_model="${source_model}_epoch${epoch_tag}"
  alias_dir="$(output_model_dir "$alias_model")"
  timematch_model="timematch_${source_tile}_to_${target_tile}_readiness_ckpt_${RUN_TAG}_${task}_seed${seed}_${config}_epoch${epoch_tag}"
  log_file="$LOG_DIR/timematch_gpu${gpu}_${task}_seed${seed}_${config}_epoch${epoch_tag}.log"

  if [ ! -f "$checkpoint" ]; then
    echo "MISSING_CHECKPOINT|gpu=$gpu|task=$task|seed=$seed|config=$config|epoch=$epoch_tag|path=$checkpoint"
    return 1
  fi

  mkdir -p "$alias_dir/fold_0"
  cp "$checkpoint" "$alias_dir/fold_0/model.pt"

  echo "START_TIMEMATCH|gpu=$gpu|task=$task|seed=$seed|config=$config|epoch=$epoch_tag|log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT_DIR/train.py" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUTS_ROOT" \
    --closed_set True \
    --with_shift_aug False \
    --source_feature_reshaper none \
    --source_feature_dual_path False \
    --source_phase_partition_mode uniform \
    --source_segment_partition_mode uniform \
    --source_phase_count 1 \
    --source_segment_count 1 \
    --source_structure_loss_version segment_boundary_window_residual \
    --source_structure_feature_target raw \
    --source_structure_intra_trade_off 0.0 \
    --source_structure_trend_trade_off 0.0 \
    --source_structure_segment_inter_trade_off 0.0 \
    --source_structure_boundary_window_trade_off 0.0 \
    --num_workers "$NUM_WORKERS" \
    --seed "$seed" \
    -e "$timematch_model" \
    --source "$source_dataset" \
    --target "$target_dataset" \
    timematch \
    --epochs "$TIMEMATCH_EPOCHS" \
    --steps_per_epoch "$TIMEMATCH_STEPS_PER_EPOCH" \
    --weights "$alias_dir" \
    > "$log_file" 2>&1
}

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/queue_gpu${gpu}.tsv"
  local worker_failed=0
  local task source_dataset target_dataset seed config compact_weight est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config compact_weight est_weight; do
    [ -z "$task" ] && continue
    local source_model source_log source_model_dir
    source_model="$(source_model_name "$source_dataset" "$task" "$seed" "$config")"
    source_model_dir="$(output_model_dir "$source_model")"
    source_log="$LOG_DIR/source_gpu${gpu}_${task}_seed${seed}_${config}.log"

    if truthy "$REUSE_SOURCE_CHECKPOINTS" && [ -f "$source_model_dir/fold_0/source_epoch_checkpoints/manifest.tsv" ]; then
      echo "REUSE_SOURCE|gpu=$gpu|task=$task|seed=$seed|config=$config|source=$source_model"
    else
      echo "START_SOURCE|gpu=$gpu|task=$task|seed=$seed|config=$config|compact_weight=$compact_weight|source=$source_model|log=$source_log"
      CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT_DIR/train.py" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUTS_ROOT" \
        --closed_set True \
        --with_shift_aug False \
        --source_feature_reshaper none \
        --source_feature_dual_path False \
        --source_phase_partition_mode uniform \
        --source_segment_partition_mode uniform \
        --source_phase_count 1 \
        --source_segment_count 1 \
        --source_structure_loss_version segment_boundary_window_residual \
        --source_structure_feature_target raw \
        --source_structure_detach_features False \
        --source_structure_intra_trade_off "$compact_weight" \
        --source_structure_amplitude_trade_off 0.0 \
        --source_structure_interphase_trade_off 0.0 \
        --source_structure_shape_trade_off 0.0 \
        --source_structure_trend_trade_off 0.0 \
        --source_structure_season_trade_off 0.0 \
        --source_structure_segment_inter_trade_off 0.0 \
        --source_structure_boundary_window_trade_off 0.0 \
        --source_checkpoint_epochs "$CHECKPOINT_EPOCHS" \
        --epochs "$SOURCE_PRETRAIN_EPOCHS" \
        --num_workers "$NUM_WORKERS" \
        --seed "$seed" \
        -e "$source_model" \
        --source "$source_dataset" \
        --target "$source_dataset" \
        sourcephasecompact \
        > "$source_log" 2>&1
      status="$?"
      if [ "$status" -ne 0 ]; then
        echo "FAIL_SOURCE|gpu=$gpu|task=$task|seed=$seed|config=$config|status=$status"
        worker_failed=1
        continue
      fi
      echo "DONE_SOURCE|gpu=$gpu|task=$task|seed=$seed|config=$config"
    fi

    IFS=',' read -r -a EPOCH_ITEMS <<< "$CHECKPOINT_EPOCHS"
    for epoch_item in "${EPOCH_ITEMS[@]}"; do
      epoch_item="$(echo "$epoch_item" | xargs)"
      [ -z "$epoch_item" ] && continue
      if ! run_timematch_epoch_probe "$gpu" "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$source_model" "$epoch_item"; then
        echo "FAIL_TIMEMATCH|gpu=$gpu|task=$task|seed=$seed|config=$config|epoch=$epoch_item"
        worker_failed=1
      else
        echo "DONE_TIMEMATCH|gpu=$gpu|task=$task|seed=$seed|config=$config|epoch=$epoch_item"
      fi
    done
  done < "$queue"
  return "$worker_failed"
}

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "CONFIGS=$CONFIGS"
echo "CHECKPOINT_EPOCHS=$CHECKPOINT_EPOCHS"
echo "SOURCE_PRETRAIN_EPOCHS=$SOURCE_PRETRAIN_EPOCHS"
echo "TIMEMATCH_EPOCHS=$TIMEMATCH_EPOCHS"
echo "TIMEMATCH_STEPS_PER_EPOCH=$TIMEMATCH_STEPS_PER_EPOCH"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "SOURCE_JOBS=$(wc -l < "$JOBS")"
echo "MANIFEST=$MANIFEST"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "DRY_RUN=True"
    echo "Jobs: $JOBS"
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

python "$ROOT_DIR/analysis/summarize_v243b_readiness_checkpoint_causal.py" \
  "$LOG_DIR" "$BASE_CONFIG" "$SHAPED_CONFIG" || failed=1

echo "Logs saved to: $LOG_DIR"
exit "$failed"
