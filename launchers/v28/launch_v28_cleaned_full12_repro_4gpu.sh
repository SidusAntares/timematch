#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_TAG="${RUN_TAG:-v28_cleaned_full12_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${RUN_TAG}}"
GPUS="${GPUS:-0 1 2 3}"
SOURCE_DOMAINS="${SOURCE_DOMAINS:-AT1 DK1 FR1 FR2}"
TASKS="${TASKS:-AT1_to_DK1,AT1_to_FR1,AT1_to_FR2,DK1_to_AT1,DK1_to_FR1,DK1_to_FR2,FR1_to_AT1,FR1_to_DK1,FR1_to_FR2,FR2_to_AT1,FR2_to_DK1,FR2_to_FR1}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-base,raw_global,smooth_k3,elastic_r2}"

mkdir -p "${LOG_DIR}"

echo "RUN_TAG=${RUN_TAG}"
echo "LOG_DIR=${LOG_DIR}"
echo "SOURCE_DOMAINS=${SOURCE_DOMAINS}"
echo "TASKS=${TASKS}"
echo "SEEDS=${SEEDS}"
echo "CONFIGS=${CONFIGS}"
echo "GPUS=${GPUS}"

(
  cd "${ROOT}" && \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  GPUS="${GPUS}" \
  SOURCE_DOMAINS="${SOURCE_DOMAINS}" \
  SEEDS="${SEEDS}" \
  CONFIGS="${CONFIGS}" \
  DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  CLOSED_SET="${CLOSED_SET:-True}" \
  SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-60}" \
  DRY_RUN="${DRY_RUN:-False}" \
  bash launchers/v28/launch_v28_cleaned_full12_source_train_4gpu.sh
) 2>&1 | tee "${LOG_DIR}/source_train_controller.log"
source_status=${PIPESTATUS[0]}
if [[ "${source_status}" -ne 0 ]]; then
  echo "SOURCE_STAGE_FAILED|status=${source_status}|log=${LOG_DIR}/source_train_controller.log" >&2
  exit "${source_status}"
fi

(
  cd "${ROOT}" && \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  SOURCE_INVENTORY="${LOG_DIR}/source_checkpoint_inventory.tsv" \
  GPUS="${GPUS}" \
  TASKS="${TASKS}" \
  SEEDS="${SEEDS}" \
  CONFIGS="${CONFIGS}" \
  DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}" \
  CLOSED_SET="${CLOSED_SET:-True}" \
  DA_EPOCHS="${DA_EPOCHS:-20}" \
  STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-60}" \
  DRY_RUN="${DRY_RUN:-False}" \
  bash launchers/v28/launch_v28_cleaned_full12_da_4gpu.sh
) 2>&1 | tee "${LOG_DIR}/da_controller.log"
da_status=${PIPESTATUS[0]}

echo "SOURCE_INVENTORY=${LOG_DIR}/source_checkpoint_inventory.tsv"
echo "JOB_STATUS=${LOG_DIR}/job_status.tsv"
echo "DA_STATUS=${LOG_DIR}/da_job_status.tsv"
echo "SUMMARY=${LOG_DIR}/summary.tsv"
echo "SUMMARY_BY_CONFIG=${LOG_DIR}/summary_by_config.tsv"
echo "SUMMARY_BY_TASK=${LOG_DIR}/summary_by_task.tsv"
echo "DELTA_VS_BASE=${LOG_DIR}/delta_vs_base_by_task.tsv"
echo "PER_SEED_PIVOT=${LOG_DIR}/per_seed_pivot.tsv"
echo "REPORT=${ROOT}/analysis/v28_cleaned_full12_reproduction_report.md"

exit "${da_status}"
