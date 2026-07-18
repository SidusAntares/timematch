#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE="${SOURCE:-austria/33UVP/2017}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rawglobal_base_plain_source}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"

cd "${ROOT}"
python train.py \
  --data_root "${DATA_ROOT}" \
  --source "${SOURCE}" \
  --target "${SOURCE}" \
  --closed_set True \
  --with_shift_aug False \
  --source_structure_mode off \
  --source_structure_weight 0.0 \
  --epochs "${SOURCE_EPOCHS:-100}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  -e "${EXPERIMENT_NAME}" \
  source_structure
