#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE="${SOURCE:-austria/33UVP/2017}"
TARGET="${TARGET:-denmark/32VNH/2017}"
SOURCE_EXPERIMENT_DIR="${SOURCE_EXPERIMENT_DIR:?set SOURCE_EXPERIMENT_DIR to the source experiment directory containing fold_0/model.pt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rawglobal_base_timematch_AT1_to_DK1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"

cd "${ROOT}"
if [[ ! -f "${SOURCE_EXPERIMENT_DIR}/fold_0/model.pt" ]]; then
  echo "ERROR: missing source checkpoint: ${SOURCE_EXPERIMENT_DIR}/fold_0/model.pt" >&2
  exit 1
fi

python train.py \
  --data_root "${DATA_ROOT}" \
  --source "${SOURCE}" \
  --target "${TARGET}" \
  --closed_set True \
  --with_shift_aug False \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  -e "${EXPERIMENT_NAME}" \
  timematch \
  --weights "${SOURCE_EXPERIMENT_DIR}" \
  --epochs "${DA_EPOCHS:-20}" \
  --steps_per_epoch "${STEPS_PER_EPOCH:-500}"
