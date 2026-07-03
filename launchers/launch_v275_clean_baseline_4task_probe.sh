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
QUEUE_SCHEDULE="${QUEUE_SCHEDULE:-size_desc}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-50}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-0}"
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
    v276_timepoint_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v276_raw_timepoint_compactness" "raw" "$V275_WEIGHT" "off" \
        "v2.7.6 probe: source-stage per-timestep class prototype compactness; TimeMatch DA-stage structure off."
      ;;
    v276_timepoint_w0p5)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v276_raw_timepoint_compactness" "raw" "0.5" "off" \
        "A1: ordinary per-timestep compactness with lambda=0.5; TimeMatch DA-stage structure off."
      ;;
    v276_smoothed_timepoint_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v276_raw_smoothed_timepoint_compactness" "raw" "$V275_WEIGHT" "off" \
        "v2.7.6 probe: source-stage smoothed per-timestep class prototype compactness; TimeMatch DA-stage structure off."
      ;;
    v276_smooth_k1_w1|v276_smooth_k3_w1|v276_smooth_k5_w1|v276_smooth_k7_w1)
      local kernel_size
      kernel_size="$(echo "$config" | sed -E 's/^v276_smooth_k([0-9]+)_w1$/\1/')"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v276_raw_smoothed_timepoint_compactness" "raw" "1.0" "off" \
        "A3: smoothed timepoint compactness kernel=${kernel_size}, lambda=1.0; TimeMatch DA-stage structure off."
      ;;
    v276_smooth_k3_w1_detach)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v276_raw_smoothed_timepoint_compactness" "raw-detached" "1.0" "off" \
        "A2: smoothed timepoint compactness kernel=3, lambda=1.0, detached features; TimeMatch DA-stage structure off."
      ;;
    v303_time_permuted_smooth_k3_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v303_time_permuted_smoothed_timepoint_compactness" "raw" "1.0" "off" \
        "v3.0.3 control: smoothed timepoint compactness kernel=3, lambda=1.0, but source structure smoothing uses a fixed pseudo-time permutation."
      ;;
    v276_trimmed_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v276_raw_trimmed_global_compactness" "raw" "$V275_WEIGHT" "off" \
        "v2.7.6 probe: source-stage raw global compactness with trimmed class center; TimeMatch DA-stage structure off."
      ;;
    v277_dct_k2_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v277_raw_lowfreq_dct_k2_compactness" "raw" "$V275_WEIGHT" "off" \
        "v2.7.7 probe: source-stage DCT low-frequency K=2 prototype compactness; TimeMatch DA-stage structure off."
      ;;
    v277_dct_k4_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v277_raw_lowfreq_dct_k4_compactness" "raw" "$V275_WEIGHT" "off" \
        "v2.7.7 probe: source-stage DCT low-frequency K=4 prototype compactness; TimeMatch DA-stage structure off."
      ;;
    v277_dct_k8_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v277_raw_lowfreq_dct_k8_compactness" "raw" "$V275_WEIGHT" "off" \
        "v2.7.7 probe: source-stage DCT low-frequency K=8 prototype compactness; TimeMatch DA-stage structure off."
      ;;
    v283a_umsc_075l3_025linf_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v283a_umsc_dual_075_025_compactness" "raw" "1.0" "off" \
        "v2.8.3a: unified multi-scale compactness, L=1.0*(0.75*L3 + 0.25*Linf); TimeMatch DA-stage structure off."
      ;;
    v283a_umsc_050l3_050linf_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v283a_umsc_dual_050_050_compactness" "raw" "1.0" "off" \
        "v2.8.3a: unified multi-scale compactness, L=1.0*(0.50*L3 + 0.50*Linf); TimeMatch DA-stage structure off."
      ;;
    v283b_umsc_060l3_020l5_020linf_w1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v283b_umsc_triscale_060_020_020_compactness" "raw" "1.0" "off" \
        "v2.8.3b: unified multi-scale compactness, L=1.0*(0.60*L3 + 0.20*L5 + 0.20*Linf); TimeMatch DA-stage structure off."
      ;;
    v284_elastic_k3_r0_w1|v284_elastic_k3_r1_w1|v284_elastic_k3_r2_w1)
      local elastic_radius
      elastic_radius="$(echo "$config" | sed -E 's/^v284_elastic_k3_r([0-9]+)_w1$/\1/')"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config" "sourcephasecompact" "v284_elastic_smoothed_timepoint_compactness" "raw" "1.0" "off" \
        "v2.8.4: elastic smoothed-timepoint compactness, kernel=3, radius=${elastic_radius}, eta=0.1, softmin_tau=0.1, detach_center=False."
      ;;
  esac >> "$MANIFEST"
}

add_job() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$JOBS"
}

IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
IFS=',' read -r -a CONFIG_NAMES <<< "$CONFIGS"

enqueue_job() {
  local task="$1"
  local seed="$2"
  local config="$3"
  local spec source_dataset target_dataset est_weight

  task="$(echo "$task" | xargs)"
  config="$(echo "$config" | xargs)"
  spec="$(task_spec "$task")" || exit 2
  read -r source_dataset target_dataset est_weight <<< "$spec"
  case "$config" in
    plain|v275_raw_w1|v276_timepoint_w0p5|v276_timepoint_w1|v276_smoothed_timepoint_w1|v276_smooth_k1_w1|v276_smooth_k3_w1|v276_smooth_k5_w1|v276_smooth_k7_w1|v276_smooth_k3_w1_detach|v303_time_permuted_smooth_k3_w1|v276_trimmed_w1|v277_dct_k2_w1|v277_dct_k4_w1|v277_dct_k8_w1|v283a_umsc_075l3_025linf_w1|v283a_umsc_050l3_050linf_w1|v283b_umsc_060l3_020l5_020linf_w1|v284_elastic_k3_r0_w1|v284_elastic_k3_r1_w1|v284_elastic_k3_r2_w1) ;;
    *)
      echo "ERROR unknown config: $config" >&2
      exit 2
      ;;
  esac
  add_manifest "$config"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$est_weight" "$RUN_TAG"
}

case "$QUEUE_SCHEDULE" in
  interleave_configs)
    for seed in $SEEDS; do
      for config in "${CONFIG_NAMES[@]}"; do
        for task in "${TASK_NAMES[@]}"; do
          enqueue_job "$task" "$seed" "$config"
        done
      done
    done
    ;;
  *)
    for seed in $SEEDS; do
      for task in "${TASK_NAMES[@]}"; do
        for config in "${CONFIG_NAMES[@]}"; do
          enqueue_job "$task" "$seed" "$config"
        done
      done
    done
    ;;
esac

for gpu in "${GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${gpu}.tsv"
done

if [ "$QUEUE_SCHEDULE" = "interleave_configs" ]; then
  cp "$JOBS" "$SORTED_JOBS"
else
  sort -t $'\t' -k6,6nr "$JOBS" > "$SORTED_JOBS"
fi
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
      --data_loader_timeout "$DATA_LOADER_TIMEOUT" \
      --seed "$seed" \
      -e "$source_model" \
      --source "$source_dataset" \
      --target "$source_dataset" || return "$?"
  else
    local loss_version compact_weight detach_features smooth_kernel elastic_radius elastic_eta elastic_softmin_tau elastic_detach_center time_permutation_seed
    compact_weight="$V275_WEIGHT"
    detach_features="False"
    smooth_kernel="3"
    elastic_radius="0"
    elastic_eta="0.1"
    elastic_softmin_tau="0.1"
    elastic_detach_center="False"
    time_permutation_seed="$seed"
    case "$config" in
      v276_timepoint_w0p5)
        loss_version="v276_raw_timepoint_compactness"
        compact_weight="0.5"
        ;;
      v276_timepoint_w1) loss_version="v276_raw_timepoint_compactness" ;;
      v276_smoothed_timepoint_w1) loss_version="v276_raw_smoothed_timepoint_compactness" ;;
      v276_smooth_k1_w1)
        loss_version="v276_raw_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="1"
        ;;
      v276_smooth_k3_w1)
        loss_version="v276_raw_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="3"
        ;;
      v276_smooth_k5_w1)
        loss_version="v276_raw_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="5"
        ;;
      v276_smooth_k7_w1)
        loss_version="v276_raw_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="7"
        ;;
      v276_smooth_k3_w1_detach)
        loss_version="v276_raw_smoothed_timepoint_compactness"
        compact_weight="1.0"
        detach_features="True"
        smooth_kernel="3"
        ;;
      v303_time_permuted_smooth_k3_w1)
        loss_version="v303_time_permuted_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="3"
        case "$task" in
          FR1_to_FR2) time_permutation_seed=$((seed + 101)) ;;
          AT1_to_DK1) time_permutation_seed=$((seed + 211)) ;;
          FR2_to_AT1) time_permutation_seed=$((seed + 307)) ;;
          DK1_to_AT1) time_permutation_seed=$((seed + 401)) ;;
          FR2_to_FR1) time_permutation_seed=$((seed + 503)) ;;
          AT1_to_FR2) time_permutation_seed=$((seed + 601)) ;;
          *) time_permutation_seed="$seed" ;;
        esac
        ;;
      v276_trimmed_w1) loss_version="v276_raw_trimmed_global_compactness" ;;
      v277_dct_k2_w1) loss_version="v277_raw_lowfreq_dct_k2_compactness" ;;
      v277_dct_k4_w1) loss_version="v277_raw_lowfreq_dct_k4_compactness" ;;
      v277_dct_k8_w1) loss_version="v277_raw_lowfreq_dct_k8_compactness" ;;
      v283a_umsc_075l3_025linf_w1)
        loss_version="v283a_umsc_dual_075_025_compactness"
        compact_weight="1.0"
        ;;
      v283a_umsc_050l3_050linf_w1)
        loss_version="v283a_umsc_dual_050_050_compactness"
        compact_weight="1.0"
        ;;
      v283b_umsc_060l3_020l5_020linf_w1)
        loss_version="v283b_umsc_triscale_060_020_020_compactness"
        compact_weight="1.0"
        ;;
      v284_elastic_k3_r0_w1)
        loss_version="v284_elastic_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="3"
        elastic_radius="0"
        ;;
      v284_elastic_k3_r1_w1)
        loss_version="v284_elastic_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="3"
        elastic_radius="1"
        ;;
      v284_elastic_k3_r2_w1)
        loss_version="v284_elastic_smoothed_timepoint_compactness"
        compact_weight="1.0"
        smooth_kernel="3"
        elastic_radius="2"
        ;;
      *) loss_version="v275_raw_global_compactness" ;;
    esac
    CUDA_VISIBLE_DEVICES="$gpu" python train.py \
      --data_root "$DATA_ROOT" \
      --closed_set "$CLOSED_SET" \
      --with_shift_aug False \
      --source_phase_partition_mode uniform \
      --source_segment_partition_mode uniform \
      --source_phase_count 1 \
      --source_segment_count 1 \
      --source_structure_loss_version "$loss_version" \
      --source_structure_feature_target raw \
      --source_structure_detach_features "$detach_features" \
      --source_structure_intra_trade_off "$compact_weight" \
      --source_structure_time_smooth_kernel_size "$smooth_kernel" \
      --source_structure_time_permutation_seed "$time_permutation_seed" \
      --source_structure_elastic_radius "$elastic_radius" \
      --source_structure_elastic_eta "$elastic_eta" \
      --source_structure_elastic_softmin_tau "$elastic_softmin_tau" \
      --source_structure_elastic_detach_center "$elastic_detach_center" \
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
      --data_loader_timeout "$DATA_LOADER_TIMEOUT" \
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
    --data_loader_timeout "$DATA_LOADER_TIMEOUT" \
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
    --data_loader_timeout "$DATA_LOADER_TIMEOUT" \
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
echo "DATA_LOADER_TIMEOUT=$DATA_LOADER_TIMEOUT"
echo "V275_WEIGHT=$V275_WEIGHT"
echo "QUEUE_SCHEDULE=$QUEUE_SCHEDULE"
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
