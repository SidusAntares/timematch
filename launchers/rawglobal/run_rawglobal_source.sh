#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE="${SOURCE:-austria/33UVP/2017}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rawglobal_base_raw_global_source}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"

cd "${ROOT}"
python train.py \
  --data_root "${DATA_ROOT}" \
  --source "${SOURCE}" \
  --target "${SOURCE}" \
  --closed_set True \
  --with_shift_aug False \
  --source_structure_mode raw_global \
  --source_structure_weight "${SOURCE_STRUCTURE_WEIGHT:-1.0}" \
  --epochs "${SOURCE_EPOCHS:-100}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  -e "${EXPERIMENT_NAME}" \
  source_structure
