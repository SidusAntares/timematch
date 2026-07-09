#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/data/user/timematch}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
RUN_TAG="${RUN_TAG:-v321_local_shift_smoke_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
GPUS="${GPUS:-0}"
MODE="${MODE:-residual}"
LOG_DIR="${ROOT}/logs/${RUN_TAG}"
OUT_DIR="${ROOT}/outputs/${RUN_TAG}_${MODE}"
LOCAL_SHIFT_LOG="${LOG_DIR}/local_shift_${MODE}.tsv"

SOURCE="${SOURCE:-france/30TXT/2017}"
TARGET="${TARGET:-france/31TCJ/2017}"
SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-}"
SOURCE_STAGE_REFERENCE_PATH="${SOURCE_STAGE_REFERENCE_PATH:-}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if [[ -z "${SOURCE_WEIGHTS}" ]]; then
  echo "ERROR: SOURCE_WEIGHTS must point to the source experiment directory that contains fold_0/model.pt" >&2
  exit 2
fi
if [[ -z "${SOURCE_STAGE_REFERENCE_PATH}" ]]; then
  echo "ERROR: SOURCE_STAGE_REFERENCE_PATH must point to source_stage_reference.pt" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_WEIGHTS}/fold_0/model.pt" ]]; then
  echo "ERROR: missing source checkpoint: ${SOURCE_WEIGHTS}/fold_0/model.pt" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_STAGE_REFERENCE_PATH}" ]]; then
  echo "ERROR: missing source stage reference: ${SOURCE_STAGE_REFERENCE_PATH}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"

cd "${ROOT}"
echo "RUN_TAG=${RUN_TAG}"
echo "SOURCE=${SOURCE}"
echo "TARGET=${TARGET}"
echo "MODE=${MODE}"
echo "SOURCE_WEIGHTS=${SOURCE_WEIGHTS}"
echo "SOURCE_STAGE_REFERENCE_PATH=${SOURCE_STAGE_REFERENCE_PATH}"
echo "LOCAL_SHIFT_LOG=${LOCAL_SHIFT_LOG}"

python train.py \
  --data_root "${DATA_ROOT}" \
  --source "${SOURCE}" \
  --target "${TARGET}" \
  --closed_set True \
  --with_shift_aug False \
  --output_dir "${OUT_DIR}" \
  --tensorboard_log_dir "${ROOT}/runs/${RUN_TAG}_${MODE}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --data_loader_timeout "${DATA_LOADER_TIMEOUT:-60}" \
  --seed "${SEED:-1}" \
  -e "${RUN_TAG}_${MODE}" \
  timematch_local_shift \
  --weights "${SOURCE_WEIGHTS}" \
  --epochs "${EPOCHS:-1}" \
  --steps_per_epoch "${STEPS_PER_EPOCH:-5}" \
  --source_stage_reference_path "${SOURCE_STAGE_REFERENCE_PATH}" \
  --local_shift_mode "${MODE}" \
  --local_shift_log_path "${LOCAL_SHIFT_LOG}" \
  --sample_size "${SAMPLE_SIZE:-8}" \
  --balance_source True \
  --run_validation
