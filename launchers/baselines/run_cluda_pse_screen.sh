#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

tasks=(AT1_to_DK1 DK1_to_FR2 FR2_to_FR1)

task_spec() {
  case "$1" in
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017" ;;
    DK1_to_FR2) echo "denmark/32VNH/2017 france/31TCJ/2017" ;;
    FR2_to_FR1) echo "france/31TCJ/2017 france/30TXT/2017" ;;
    *) echo "unknown task: $1" >&2; return 2 ;;
  esac
}

cd "$ROOT"

# Phase 1: exactly three source-only PSE+CLUDA-TCN runs.
for task in "${tasks[@]}"; do
  read -r source target <<< "$(task_spec "$task")"
  source_name="plain_psecludatcn_${task}_seed${SEED}"
  python train.py \
    --data_root "$DATA_ROOT" --source "$source" --target "$target" \
    --closed_set True --seed "$SEED" --device "$DEVICE" \
    --output_dir "$OUTPUT_ROOT" --model cludatcn --epochs 100 \
    --with_extra False --with_shift_aug False \
    -e "$source_name"
done

# Phase 2: strict preflight, then exactly three source-initialized full runs.
for task in "${tasks[@]}"; do
  read -r source target <<< "$(task_spec "$task")"
  source_name="plain_psecludatcn_${task}_seed${SEED}"
  full_name="cluda_pse_full_${task}_seed${SEED}"
  source_run="$OUTPUT_ROOT/$source_name"
  source_checkpoint="$source_run/fold_0/model.pt"

  python tools/validate_cluda_source_checkpoint.py \
    --run-dir "$source_run" --source "$source" --target "$target" --seed "$SEED"

  python train.py \
    --data_root "$DATA_ROOT" --source "$source" --target "$target" \
    --closed_set True --seed "$SEED" --device "$DEVICE" \
    --output_dir "$OUTPUT_ROOT" --model cludatcn \
    --with_extra False --with_shift_aug False \
    -e "$full_name" \
    cluda --cluda_init source_weights --weights "$source_checkpoint" \
    --epochs 20 --steps_per_epoch 500
done
