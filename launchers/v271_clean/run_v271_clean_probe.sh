#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v271_clean_probe_${STAMP}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
DA_EPOCHS="${DA_EPOCHS:-20}"
DA_STEPS_PER_EPOCH="${DA_STEPS_PER_EPOCH:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
ADAPTIVE_SUPPORT_ROOT="${ADAPTIVE_SUPPORT_ROOT:-}"
ADAPTIVE_TRADE_OFF="${ADAPTIVE_TRADE_OFF:-0.0}"
ADAPTIVE_WARMUP_EPOCHS="${ADAPTIVE_WARMUP_EPOCHS:-3}"
ADAPTIVE_RAMP_EPOCHS="${ADAPTIVE_RAMP_EPOCHS:-3}"
ADAPTIVE_MIN_SCORE="${ADAPTIVE_MIN_SCORE:-0.0}"
ADAPTIVE_MIN_GATE="${ADAPTIVE_MIN_GATE:-0.0}"
ADAPTIVE_MIN_POINTS="${ADAPTIVE_MIN_POINTS:-2}"

DISCOVER_ADAPTIVE_SUPPORTS="${DISCOVER_ADAPTIVE_SUPPORTS:-0}"
DISCOVERY_SOURCE_MAX_BATCHES="${DISCOVERY_SOURCE_MAX_BATCHES:-64}"
DISCOVERY_TARGET_MAX_BATCHES="${DISCOVERY_TARGET_MAX_BATCHES:-64}"
DISCOVERY_SHIFT_SAMPLE_SIZE="${DISCOVERY_SHIFT_SAMPLE_SIZE:-40}"
DISCOVERY_SHIFT_JITTER="${DISCOVERY_SHIFT_JITTER:-3}"
DISCOVERY_ATOMIC_BINS="${DISCOVERY_ATOMIC_BINS:-12}"
DISCOVERY_TOP_K_SEGMENTS="${DISCOVERY_TOP_K_SEGMENTS:-8}"
DISCOVERY_TOP_M_PER_PAIR="${DISCOVERY_TOP_M_PER_PAIR:-2}"
DISCOVERY_MAX_SUPPORTS="${DISCOVERY_MAX_SUPPORTS:-12}"
DISCOVERY_SCORE_QUANTILE="${DISCOVERY_SCORE_QUANTILE:-0.75}"
DISCOVERY_MIN_SCORE="${DISCOVERY_MIN_SCORE:-0.0}"
DISCOVERY_MIN_RATIO="${DISCOVERY_MIN_RATIO:-1.0}"
DISCOVERY_SOFT_EVIDENCE="${DISCOVERY_SOFT_EVIDENCE:-True}"
DISCOVERY_MAX_MARGIN="${DISCOVERY_MAX_MARGIN:-0.20}"
DISCOVERY_MIN_TOP2_MASS="${DISCOVERY_MIN_TOP2_MASS:-0.35}"
DISCOVERY_PROTOTYPE_TEMPERATURE="${DISCOVERY_PROTOTYPE_TEMPERATURE:-1.0}"
DISCOVERY_BASELINE_PAIRS_PER_SAMPLE="${DISCOVERY_BASELINE_PAIRS_PER_SAMPLE:-4}"
DISCOVERY_BASELINE_MODE="${DISCOVERY_BASELINE_MODE:-mean}"
DISCOVERY_APPLY_SOURCE_RESHAPER="${DISCOVERY_APPLY_SOURCE_RESHAPER:-False}"

TREND_KERNEL_SIZE="${TREND_KERNEL_SIZE:-5}"
TREND_SMOOTHING_MODE="${TREND_SMOOTHING_MODE:-time}"
TREND_BANDWIDTH="${TREND_BANDWIDTH:-0.0}"
TREND_KERNEL="${TREND_KERNEL:-gaussian}"
TREND_DYNAMICS_TRADE_OFF="${TREND_DYNAMICS_TRADE_OFF:-0.05}"
RESIDUAL_VARIANCE_TRADE_OFF="${RESIDUAL_VARIANCE_TRADE_OFF:-0.10}"
RESIDUAL_ENERGY_TRADE_OFF="${RESIDUAL_ENERGY_TRADE_OFF:-0.05}"
RESIDUAL_ENERGY_MARGIN="${RESIDUAL_ENERGY_MARGIN:-1.0}"

TASKS=(
  "FR2 DK1 france/31TCJ/2017 denmark/32VNH/2017"
  "DK1 FR1 denmark/32VNH/2017 france/30TXT/2017"
  "FR1 DK1 france/30TXT/2017 denmark/32VNH/2017"
  "AT1 DK1 austria/33UVP/2017 denmark/32VNH/2017"
)

TASK_FILTER="${TASK_FILTER:-all}"

should_run_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  local filter="${TASK_FILTER}"
  if [ "${filter}" = "all" ] || [ -z "${filter}" ]; then
    return 0
  fi
  local key_arrow="${src_alias}->${tgt_alias}"
  local key_to="${src_alias}_to_${tgt_alias}"
  IFS=',' read -ra items <<< "${filter}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ "${item}" = "${key_arrow}" ] || [ "${item}" = "${key_to}" ]; then
      return 0
    fi
  done
  return 1
}

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l)"
    if [ "${running}" -lt "${#GPUS[@]}" ]; then
      return 0
    fi
    sleep 20
  done
}

support_file_for_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  if [ -z "${ADAPTIVE_SUPPORT_ROOT}" ]; then
    return 1
  fi
  local key="${src_alias}_to_${tgt_alias}"
  local candidates=(
    "${ADAPTIVE_SUPPORT_ROOT}/${key}/adaptive_supports.json"
    "${ADAPTIVE_SUPPORT_ROOT}/${key}/supports.json"
    "${ADAPTIVE_SUPPORT_ROOT}/${key}.json"
  )
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

common_structure_args=(
  --source_feature_reshaper residual_temporal_conv
  --source_feature_reshaper_strength 0.10
  --source_feature_reshaper_kernel_size 3
  --source_feature_reshaper_reg_trade_off 0.05
  --source_feature_dual_path True
  --source_feature_dual_cls_trade_off 1.0
  --source_feature_dual_relation_trade_off 0.03
  --source_phase_partition_mode uniform
  --source_segment_partition_mode uniform
  --source_phase_count 1
  --source_segment_count 1
  --source_phase_min_sample_points 2
  --source_structure_loss_version v271_global
  --source_structure_intra_trade_off 1.0
  --source_structure_v271_trend_kernel_size "${TREND_KERNEL_SIZE}"
  --source_structure_v271_trend_smoothing_mode "${TREND_SMOOTHING_MODE}"
  --source_structure_v271_trend_bandwidth "${TREND_BANDWIDTH}"
  --source_structure_v271_trend_kernel "${TREND_KERNEL}"
  --source_structure_v271_trend_dynamics_trade_off "${TREND_DYNAMICS_TRADE_OFF}"
  --source_structure_v271_residual_variance_trade_off "${RESIDUAL_VARIANCE_TRADE_OFF}"
  --source_structure_v271_residual_energy_trade_off "${RESIDUAL_ENERGY_TRADE_OFF}"
  --source_structure_v271_residual_energy_margin "${RESIDUAL_ENERGY_MARGIN}"
)

run_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local gpu="$5"

  local task_key="${src_alias}_to_${tgt_alias}"
  local source_exp="remote_${task_key}_v271clean_source_${STAMP}"
  local da_exp="remote_${task_key}_v271clean_timematch_${STAMP}"
  local source_log="${LOG_ROOT}/${source_exp}.log"
  local da_log="${LOG_ROOT}/${da_exp}.log"
  local source_weights="${OUT_ROOT}/${source_exp}"
  local support_dir="${OUT_ROOT}/${task_key}_adaptive_support_${STAMP}"
  local support_log=""

  echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} source GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --data_root "${DATA_ROOT}" \
    --closed_set True \
    --with_shift_aug False \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${SOURCE_EPOCHS}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    -e "${source_exp}" \
    --source "${source_dataset}" \
    --target "${target_dataset}" \
    "${common_structure_args[@]}" \
    sourcephasecompact \
    > "${source_log}" 2>&1

  local adaptive_args=(--timematch_source_structure_trade_off 0.0)
  local support_file=""
  if [ "${DISCOVER_ADAPTIVE_SUPPORTS}" = "1" ] || [ "${DISCOVER_ADAPTIVE_SUPPORTS}" = "True" ] || [ "${DISCOVER_ADAPTIVE_SUPPORTS}" = "true" ]; then
    mkdir -p "${support_dir}"
    support_log="${LOG_ROOT}/${task_key}_adaptive_support_${STAMP}.log"
    echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} discover adaptive supports GPU=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python analysis/v272_adaptive_segment_discovery.py \
      --run_dir "${source_weights}" \
      --output_dir "${support_dir}" \
      --device cuda \
      --data_root "${DATA_ROOT}" \
      --source_max_batches "${DISCOVERY_SOURCE_MAX_BATCHES}" \
      --target_max_batches "${DISCOVERY_TARGET_MAX_BATCHES}" \
      --shift_sample_size "${DISCOVERY_SHIFT_SAMPLE_SIZE}" \
      --shift_jitter "${DISCOVERY_SHIFT_JITTER}" \
      --atomic_bins "${DISCOVERY_ATOMIC_BINS}" \
      --top_k_segments "${DISCOVERY_TOP_K_SEGMENTS}" \
      --top_m_per_pair "${DISCOVERY_TOP_M_PER_PAIR}" \
      --max_adaptive_supports "${DISCOVERY_MAX_SUPPORTS}" \
      --segment_score_quantile "${DISCOVERY_SCORE_QUANTILE}" \
      --min_segment_score "${DISCOVERY_MIN_SCORE}" \
      --min_segment_ratio "${DISCOVERY_MIN_RATIO}" \
      --soft_evidence "${DISCOVERY_SOFT_EVIDENCE}" \
      --max_margin "${DISCOVERY_MAX_MARGIN}" \
      --min_top2_mass "${DISCOVERY_MIN_TOP2_MASS}" \
      --prototype_temperature "${DISCOVERY_PROTOTYPE_TEMPERATURE}" \
      --baseline_pairs_per_sample "${DISCOVERY_BASELINE_PAIRS_PER_SAMPLE}" \
      --baseline_mode "${DISCOVERY_BASELINE_MODE}" \
      --apply_source_reshaper "${DISCOVERY_APPLY_SOURCE_RESHAPER}" \
      > "${support_log}" 2>&1
    if [ -f "${support_dir}/adaptive_supports.json" ]; then
      support_file="${support_dir}/adaptive_supports.json"
    fi
  elif support_file="$(support_file_for_task "${src_alias}" "${tgt_alias}")"; then
    :
  fi

  if [ -n "${support_file}" ]; then
    adaptive_args+=(
      --timematch_v271_adaptive_support_file "${support_file}"
      --timematch_v271_adaptive_trade_off "${ADAPTIVE_TRADE_OFF}"
      --timematch_v271_adaptive_warmup_epochs "${ADAPTIVE_WARMUP_EPOCHS}"
      --timematch_v271_adaptive_ramp_epochs "${ADAPTIVE_RAMP_EPOCHS}"
      --timematch_v271_adaptive_min_score "${ADAPTIVE_MIN_SCORE}"
      --timematch_v271_adaptive_min_gate "${ADAPTIVE_MIN_GATE}"
      --timematch_v271_adaptive_min_points "${ADAPTIVE_MIN_POINTS}"
    )
    echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} adaptive_support=${support_file}"
  else
    echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} adaptive support missing; DA adaptive loss disabled"
  fi

  echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} timematch GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --data_root "${DATA_ROOT}" \
    --closed_set True \
    --with_shift_aug False \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    -e "${da_exp}" \
    --source "${source_dataset}" \
    --target "${target_dataset}" \
    "${common_structure_args[@]}" \
    timematch \
    --weights "${source_weights}" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${DA_STEPS_PER_EPOCH}" \
    --estimate_shift True \
    --max_temporal_shift 60 \
    --shift_source True \
    --balance_source True \
    "${adaptive_args[@]}" \
    > "${da_log}" 2>&1

  echo -e "${task_key}\t${source_log}\t${da_log}\t${support_file:-none}\t${support_log:-none}" >> "${LOG_ROOT}/task_logs.tsv"
}

gpu_idx=0
for task in "${TASKS[@]}"; do
  read -r src_alias tgt_alias source_dataset target_dataset <<< "${task}"
  if ! should_run_task "${src_alias}" "${tgt_alias}"; then
    continue
  fi
  wait_for_slot
  gpu="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
  gpu_idx=$((gpu_idx + 1))
  run_task "${src_alias}" "${tgt_alias}" "${source_dataset}" "${target_dataset}" "${gpu}" &
done

wait
echo "[v2.7.1-clean] done. logs=${LOG_ROOT}"
