#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v272_four_setting_full_matrix_${STAMP}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_ROOT}/summary.tsv}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
DATASETS="${DATASETS:-REMOTE HAR HHAR_SA}"
SETTINGS="${SETTINGS:-global_only segment_only global_segment global_segment_fullcore}"
TASK_FILTER="${TASK_FILTER:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SOURCE_WEIGHTS_ROOT="${SOURCE_WEIGHTS_ROOT:-}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-${DATA_ROOT:-/data/user/DBL/timematch_data}}"
HAR_DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/UCIHAR/HAR}"
HHAR_DATA_ROOT="${HHAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"

SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

REMOTE_SOURCE_EPOCHS="${REMOTE_SOURCE_EPOCHS:-50}"
REMOTE_DA_EPOCHS="${REMOTE_DA_EPOCHS:-20}"
REMOTE_STEPS_PER_EPOCH="${REMOTE_STEPS_PER_EPOCH:-300}"
REMOTE_BATCH_SIZE="${REMOTE_BATCH_SIZE:-128}"
REMOTE_LR="${REMOTE_LR:-0.001}"
REMOTE_DA_LR="${REMOTE_DA_LR:-0.0001}"
REMOTE_WEIGHT_DECAY="${REMOTE_WEIGHT_DECAY:-0.0001}"
REMOTE_MAX_TEMPORAL_SHIFT="${REMOTE_MAX_TEMPORAL_SHIFT:-60}"
REMOTE_SHIFT_SAMPLE_SIZE="${REMOTE_SHIFT_SAMPLE_SIZE:-100}"
REMOTE_VAL_RATIO="${REMOTE_VAL_RATIO:-0.1}"
REMOTE_TEST_RATIO="${REMOTE_TEST_RATIO:-0.2}"
REMOTE_SEGMENT_MODE="${REMOTE_SEGMENT_MODE:-doy_gap}"
REMOTE_SEGMENT_COUNT="${REMOTE_SEGMENT_COUNT:-5}"
REMOTE_MIN_SAMPLE_POINTS="${REMOTE_MIN_SAMPLE_POINTS:-2}"
REMOTE_SEGMENT_GAP_THRESHOLD="${REMOTE_SEGMENT_GAP_THRESHOLD:-45}"
REMOTE_SEGMENT_MIN_POINTS="${REMOTE_SEGMENT_MIN_POINTS:-3}"
REMOTE_SEGMENT_MAX_POINTS="${REMOTE_SEGMENT_MAX_POINTS:-8}"
REMOTE_SEGMENT_MAX_SPAN="${REMOTE_SEGMENT_MAX_SPAN:-120}"

HAR_SOURCE_EPOCHS="${HAR_SOURCE_EPOCHS:-40}"
HAR_DA_EPOCHS="${HAR_DA_EPOCHS:-40}"
HAR_STEPS_PER_EPOCH="${HAR_STEPS_PER_EPOCH:-0}"
HAR_BATCH_SIZE="${HAR_BATCH_SIZE:-32}"
HAR_LR="${HAR_LR:-0.001}"
HAR_DA_LR="${HAR_DA_LR:-0.001}"
HAR_WEIGHT_DECAY="${HAR_WEIGHT_DECAY:-0.0001}"
HAR_MAX_TEMPORAL_SHIFT="${HAR_MAX_TEMPORAL_SHIFT:-16}"
HAR_SHIFT_SAMPLE_SIZE="${HAR_SHIFT_SAMPLE_SIZE:-100}"
HAR_VAL_RATIO="${HAR_VAL_RATIO:-0.1}"
HAR_SEGMENT_MODE="${HAR_SEGMENT_MODE:-uniform}"
HAR_SEGMENT_COUNT="${HAR_SEGMENT_COUNT:-5}"
HAR_MIN_SAMPLE_POINTS="${HAR_MIN_SAMPLE_POINTS:-1}"
HAR_SEGMENT_GAP_THRESHOLD="${HAR_SEGMENT_GAP_THRESHOLD:-45}"
HAR_SEGMENT_MIN_POINTS="${HAR_SEGMENT_MIN_POINTS:-3}"
HAR_SEGMENT_MAX_POINTS="${HAR_SEGMENT_MAX_POINTS:-8}"
HAR_SEGMENT_MAX_SPAN="${HAR_SEGMENT_MAX_SPAN:-120}"

TREND_KERNEL_SIZE="${TREND_KERNEL_SIZE:-5}"
TREND_SMOOTHING_MODE="${TREND_SMOOTHING_MODE:-time}"
TREND_BANDWIDTH="${TREND_BANDWIDTH:-0.0}"
TREND_KERNEL="${TREND_KERNEL:-gaussian}"
TREND_DYNAMICS_TRADE_OFF="${TREND_DYNAMICS_TRADE_OFF:-0.05}"
RESIDUAL_VARIANCE_TRADE_OFF="${RESIDUAL_VARIANCE_TRADE_OFF:-0.10}"
RESIDUAL_ENERGY_TRADE_OFF="${RESIDUAL_ENERGY_TRADE_OFF:-0.05}"
RESIDUAL_ENERGY_MARGIN="${RESIDUAL_ENERGY_MARGIN:-1.0}"
SEGMENT_BASIS_TRADE_OFF="${SEGMENT_BASIS_TRADE_OFF:-1.0}"

RESHAPER_STRENGTH="${RESHAPER_STRENGTH:-0.10}"
RESHAPER_KERNEL_SIZE="${RESHAPER_KERNEL_SIZE:-3}"
RESHAPER_REG_TRADE_OFF="${RESHAPER_REG_TRADE_OFF:-0.05}"
DUAL_CLS_TRADE_OFF="${DUAL_CLS_TRADE_OFF:-1.0}"
DUAL_RELATION_TRADE_OFF="${DUAL_RELATION_TRADE_OFF:-0.03}"

SEGMENT_INTRA_TRADE_OFF="${SEGMENT_INTRA_TRADE_OFF:-1.0}"
SEGMENT_TREND_TRADE_OFF="${SEGMENT_TREND_TRADE_OFF:-0.05}"
SEGMENT_INTER_TRADE_OFF="${SEGMENT_INTER_TRADE_OFF:-0.02}"
SEGMENT_BOUNDARY_TRADE_OFF="${SEGMENT_BOUNDARY_TRADE_OFF:-0.20}"
SEGMENT_BOUNDARY_WINDOW_SIZE="${SEGMENT_BOUNDARY_WINDOW_SIZE:-2}"

ADAPTIVE_TRADE_OFF="${ADAPTIVE_TRADE_OFF:-0.01}"
ADAPTIVE_WARMUP_EPOCHS="${ADAPTIVE_WARMUP_EPOCHS:-3}"
ADAPTIVE_RAMP_EPOCHS="${ADAPTIVE_RAMP_EPOCHS:-3}"
ADAPTIVE_MIN_SCORE="${ADAPTIVE_MIN_SCORE:-0.0}"
ADAPTIVE_MIN_GATE="${ADAPTIVE_MIN_GATE:-0.0}"
ADAPTIVE_MIN_POINTS="${ADAPTIVE_MIN_POINTS:-2}"
ADAPTIVE_TAPER_MODE="${ADAPTIVE_TAPER_MODE:-none}"
ADAPTIVE_TAPER_RATIO="${ADAPTIVE_TAPER_RATIO:-0.0}"
ADAPTIVE_TREND_TRADE_OFF="${ADAPTIVE_TREND_TRADE_OFF:-1.0}"
ADAPTIVE_TREND_DYNAMICS_TRADE_OFF="${ADAPTIVE_TREND_DYNAMICS_TRADE_OFF:-0.05}"
ADAPTIVE_RESIDUAL_VARIANCE_TRADE_OFF="${ADAPTIVE_RESIDUAL_VARIANCE_TRADE_OFF:-0.10}"
ADAPTIVE_RESIDUAL_ENERGY_TRADE_OFF="${ADAPTIVE_RESIDUAL_ENERGY_TRADE_OFF:-0.05}"

DA_DISCOVERY_SOURCE_MAX_BATCHES="${DA_DISCOVERY_SOURCE_MAX_BATCHES:-64}"
DA_DISCOVERY_TARGET_MAX_BATCHES="${DA_DISCOVERY_TARGET_MAX_BATCHES:-64}"
DA_DISCOVERY_ATOMIC_BINS="${DA_DISCOVERY_ATOMIC_BINS:-12}"
DA_DISCOVERY_SHIFT_JITTER="${DA_DISCOVERY_SHIFT_JITTER:-3}"
DA_DISCOVERY_SOFT_EVIDENCE="${DA_DISCOVERY_SOFT_EVIDENCE:-False}"
DA_DISCOVERY_MAX_MARGIN="${DA_DISCOVERY_MAX_MARGIN:-0.20}"
DA_DISCOVERY_MIN_TOP2_MASS="${DA_DISCOVERY_MIN_TOP2_MASS:-0.35}"
DA_DISCOVERY_PROTOTYPE_TEMPERATURE="${DA_DISCOVERY_PROTOTYPE_TEMPERATURE:-1.0}"
DA_DISCOVERY_BASELINE_PAIRS_PER_SAMPLE="${DA_DISCOVERY_BASELINE_PAIRS_PER_SAMPLE:-4}"
DA_DISCOVERY_BASELINE_MODE="${DA_DISCOVERY_BASELINE_MODE:-mean}"
DA_DISCOVERY_SHUFFLE_PAIR_BASELINE="${DA_DISCOVERY_SHUFFLE_PAIR_BASELINE:-True}"
DA_DISCOVERY_SCORE_QUANTILE="${DA_DISCOVERY_SCORE_QUANTILE:-0.85}"
DA_DISCOVERY_MIN_SCORE="${DA_DISCOVERY_MIN_SCORE:-0.0}"
DA_DISCOVERY_MIN_RATIO="${DA_DISCOVERY_MIN_RATIO:-1.2}"
DA_DISCOVERY_TOP_M_PER_PAIR="${DA_DISCOVERY_TOP_M_PER_PAIR:-1}"
DA_DISCOVERY_MAX_SUPPORTS="${DA_DISCOVERY_MAX_SUPPORTS:-4}"
DA_DISCOVERY_MIN_SUPPORT_COUNT="${DA_DISCOVERY_MIN_SUPPORT_COUNT:-16}"
DA_DISCOVERY_MIN_SHIFT_STABILITY="${DA_DISCOVERY_MIN_SHIFT_STABILITY:-0.66}"
DA_DISCOVERY_MAX_SUPPORT_ATOMS="${DA_DISCOVERY_MAX_SUPPORT_ATOMS:-2}"
DA_DISCOVERY_MAX_INTERVAL_SPAN="${DA_DISCOVERY_MAX_INTERVAL_SPAN:-90}"
DA_DISCOVERY_GATE_SCORE_HIGH="${DA_DISCOVERY_GATE_SCORE_HIGH:-0.5}"
DA_DISCOVERY_GATE_MODE="${DA_DISCOVERY_GATE_MODE:-score}"
DA_DISCOVERY_GATE_LOW="${DA_DISCOVERY_GATE_LOW:-0.30}"
DA_DISCOVERY_GATE_HIGH="${DA_DISCOVERY_GATE_HIGH:-0.70}"
DA_DISCOVERY_GATE_LIGHT="${DA_DISCOVERY_GATE_LIGHT:-0.30}"
DA_DISCOVERY_GATE_FULL="${DA_DISCOVERY_GATE_FULL:-1.0}"
DA_DISCOVERY_GATE_RATIO_HIGH="${DA_DISCOVERY_GATE_RATIO_HIGH:-4.0}"
DA_DISCOVERY_GATE_COUNT_HIGH="${DA_DISCOVERY_GATE_COUNT_HIGH:-0}"
DA_DISCOVERY_GATE_SOURCE_SEP_HIGH="${DA_DISCOVERY_GATE_SOURCE_SEP_HIGH:-2.0}"
DA_DISCOVERY_GATE_TARGET_EXPLAIN_HIGH="${DA_DISCOVERY_GATE_TARGET_EXPLAIN_HIGH:-0.70}"
DA_DISCOVERY_APPLY_SOURCE_RESHAPER="${DA_DISCOVERY_APPLY_SOURCE_RESHAPER:-False}"

REMOTE_TASKS=(
  "FR1|FR2|france/30TXT/2017|france/31TCJ/2017"
  "FR1|DK1|france/30TXT/2017|denmark/32VNH/2017"
  "FR1|AT1|france/30TXT/2017|austria/33UVP/2017"
  "FR2|FR1|france/31TCJ/2017|france/30TXT/2017"
  "FR2|DK1|france/31TCJ/2017|denmark/32VNH/2017"
  "FR2|AT1|france/31TCJ/2017|austria/33UVP/2017"
  "DK1|FR1|denmark/32VNH/2017|france/30TXT/2017"
  "DK1|FR2|denmark/32VNH/2017|france/31TCJ/2017"
  "DK1|AT1|denmark/32VNH/2017|austria/33UVP/2017"
  "AT1|FR1|austria/33UVP/2017|france/30TXT/2017"
  "AT1|FR2|austria/33UVP/2017|france/31TCJ/2017"
  "AT1|DK1|austria/33UVP/2017|denmark/32VNH/2017"
)

HAR_TASKS=(
  "2|11"
  "6|23"
  "7|13"
  "9|18"
  "12|16"
)

HHAR_TASKS=(
  "0|6"
  "1|6"
  "2|7"
  "3|8"
  "4|5"
)

printf "dataset\ttask\tsetting\tstage\tstatus\tf1\tlog\toutput\n" > "${SUMMARY_FILE}"
JOB_INDEX=0

parse_test_f1() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
try:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
except OSError:
    print("")
    raise SystemExit(0)
matches = re.findall(r"Test result for .*?: accuracy=[0-9.]+, f1=([0-9.]+)", text)
print(matches[-1] if matches else "")
PY
}

append_summary() {
  local dataset="$1"
  local task_key="$2"
  local setting="$3"
  local stage="$4"
  local status="$5"
  local f1="$6"
  local log_file="$7"
  local out_dir="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${dataset}" "${task_key}" "${setting}" "${stage}" "${status}" "${f1}" "${log_file}" "${out_dir}" \
    >> "${SUMMARY_FILE}"
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

should_run_task() {
  local dataset="$1"
  local task_key="$2"
  local task_arrow="${task_key/_to_/->}"
  local filter="${TASK_FILTER}"
  if [ -z "${filter}" ] || [ "${filter}" = "all" ]; then
    return 0
  fi
  IFS=',' read -ra items <<< "${filter}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ "${item}" = "${task_key}" ] || [ "${item}" = "${task_arrow}" ] || [ "${item}" = "${dataset}:${task_key}" ] || [ "${item}" = "${dataset}:${task_arrow}" ]; then
      return 0
    fi
  done
  return 1
}

source_setting_for() {
  local setting="$1"
  if [ "${setting}" = "global_segment_fullcore" ]; then
    printf "global_segment\n"
  elif [ "${setting}" = "global_fullcore" ]; then
    printf "global_only\n"
  else
    printf "%s\n" "${setting}"
  fi
}

source_weights_for_task() {
  local dataset="$1"
  local task_key="$2"
  local source_setting="$3"
  if [ -z "${SOURCE_WEIGHTS_ROOT}" ]; then
    return 1
  fi
  local prefix="${dataset,,}_${task_key}_${source_setting}_source"
  local candidates=(
    "${SOURCE_WEIGHTS_ROOT}/${prefix}"
    "${SOURCE_WEIGHTS_ROOT}/${prefix}_${STAMP}"
  )
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}/fold_0/model.pt" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  local matches=()
  while IFS= read -r match; do
    if [ -f "${match}/fold_0/model.pt" ]; then
      matches+=("${match}")
    fi
  done < <(compgen -G "${SOURCE_WEIGHTS_ROOT}/${prefix}_*" || true)
  if [ "${#matches[@]}" -gt 0 ]; then
    printf '%s\n' "${matches[@]}" | sort | tail -n 1
    return 0
  fi
  return 1
}

dataset_args() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local src_path="$4"
  local tgt_path="$5"
  case "${dataset}" in
    REMOTE)
      DATA_ARGS=(
        --data_root "${REMOTE_DATA_ROOT}"
        --source "${src_path}"
        --target "${tgt_path}"
        --closed_set True
        --num_folds 1
        --val_ratio "${REMOTE_VAL_RATIO}"
        --test_ratio "${REMOTE_TEST_RATIO}"
        --batch_size "${REMOTE_BATCH_SIZE}"
        --lr "${REMOTE_LR}"
        --weight_decay "${REMOTE_WEIGHT_DECAY}"
        --input_dim 10
        --num_pixels 64
        --seq_length 30
        --model pseltae
      )
      ;;
    HAR)
      DATA_ARGS=(
        --dataset_type har
        --har_dataset_name HAR
        --data_root "${HAR_DATA_ROOT}"
        --source "${src}"
        --target "${tgt}"
        --closed_set True
        --num_folds 1
        --val_ratio "${HAR_VAL_RATIO}"
        --test_ratio 0.0
        --batch_size "${HAR_BATCH_SIZE}"
        --lr "${HAR_LR}"
        --weight_decay "${HAR_WEIGHT_DECAY}"
        --input_dim 9
        --num_pixels 1
        --seq_length 128
        --model pseltae
      )
      ;;
    HHAR_SA|HHAR)
      DATA_ARGS=(
        --dataset_type har
        --har_dataset_name HHAR_SA
        --data_root "${HHAR_DATA_ROOT}"
        --source "${src}"
        --target "${tgt}"
        --closed_set True
        --num_folds 1
        --val_ratio "${HAR_VAL_RATIO}"
        --test_ratio 0.0
        --batch_size "${HAR_BATCH_SIZE}"
        --lr "${HAR_LR}"
        --weight_decay "${HAR_WEIGHT_DECAY}"
        --input_dim 3
        --num_pixels 1
        --seq_length 128
        --model pseltae
      )
      ;;
    *)
      echo "Unsupported dataset ${dataset}" >&2
      exit 1
      ;;
  esac
}

timing_args() {
  local dataset="$1"
  case "${dataset}" in
    REMOTE)
      SOURCE_EPOCHS="${REMOTE_SOURCE_EPOCHS}"
      DA_EPOCHS="${REMOTE_DA_EPOCHS}"
      DA_STEPS="${REMOTE_STEPS_PER_EPOCH}"
      DA_LR="${REMOTE_DA_LR}"
      MAX_SHIFT="${REMOTE_MAX_TEMPORAL_SHIFT}"
      SHIFT_SAMPLE="${REMOTE_SHIFT_SAMPLE_SIZE}"
      SEG_MODE="${REMOTE_SEGMENT_MODE}"
      SEG_COUNT="${REMOTE_SEGMENT_COUNT}"
      MIN_SAMPLE_POINTS="${REMOTE_MIN_SAMPLE_POINTS}"
      SEG_GAP_THRESHOLD="${REMOTE_SEGMENT_GAP_THRESHOLD}"
      SEG_MIN_POINTS="${REMOTE_SEGMENT_MIN_POINTS}"
      SEG_MAX_POINTS="${REMOTE_SEGMENT_MAX_POINTS}"
      SEG_MAX_SPAN="${REMOTE_SEGMENT_MAX_SPAN}"
      ;;
    *)
      SOURCE_EPOCHS="${HAR_SOURCE_EPOCHS}"
      DA_EPOCHS="${HAR_DA_EPOCHS}"
      DA_STEPS="${HAR_STEPS_PER_EPOCH}"
      DA_LR="${HAR_DA_LR}"
      MAX_SHIFT="${HAR_MAX_TEMPORAL_SHIFT}"
      SHIFT_SAMPLE="${HAR_SHIFT_SAMPLE_SIZE}"
      SEG_MODE="${HAR_SEGMENT_MODE}"
      SEG_COUNT="${HAR_SEGMENT_COUNT}"
      MIN_SAMPLE_POINTS="${HAR_MIN_SAMPLE_POINTS}"
      SEG_GAP_THRESHOLD="${HAR_SEGMENT_GAP_THRESHOLD}"
      SEG_MIN_POINTS="${HAR_SEGMENT_MIN_POINTS}"
      SEG_MAX_POINTS="${HAR_SEGMENT_MAX_POINTS}"
      SEG_MAX_SPAN="${HAR_SEGMENT_MAX_SPAN}"
      ;;
  esac
}

structure_args() {
  local source_setting="$1"
  case "${source_setting}" in
    global_only)
      STRUCT_ARGS=(
        --source_feature_reshaper residual_temporal_conv
        --source_feature_reshaper_strength "${RESHAPER_STRENGTH}"
        --source_feature_reshaper_kernel_size "${RESHAPER_KERNEL_SIZE}"
        --source_feature_reshaper_reg_trade_off "${RESHAPER_REG_TRADE_OFF}"
        --source_feature_dual_path True
        --source_feature_dual_cls_trade_off "${DUAL_CLS_TRADE_OFF}"
        --source_feature_dual_relation_trade_off "${DUAL_RELATION_TRADE_OFF}"
        --source_phase_partition_mode uniform
        --source_segment_partition_mode uniform
        --source_phase_count 1
        --source_segment_count 1
        --source_phase_min_sample_points "${MIN_SAMPLE_POINTS}"
        --source_structure_loss_version v271_global
        --source_structure_intra_trade_off "${SEGMENT_INTRA_TRADE_OFF}"
        --source_structure_v271_trend_kernel_size "${TREND_KERNEL_SIZE}"
        --source_structure_v271_trend_smoothing_mode "${TREND_SMOOTHING_MODE}"
        --source_structure_v271_trend_bandwidth "${TREND_BANDWIDTH}"
        --source_structure_v271_trend_kernel "${TREND_KERNEL}"
        --source_structure_v271_trend_dynamics_trade_off "${TREND_DYNAMICS_TRADE_OFF}"
        --source_structure_v271_residual_variance_trade_off "${RESIDUAL_VARIANCE_TRADE_OFF}"
        --source_structure_v271_residual_energy_trade_off "${RESIDUAL_ENERGY_TRADE_OFF}"
        --source_structure_v271_residual_energy_margin "${RESIDUAL_ENERGY_MARGIN}"
      )
      ;;
    segment_only)
      STRUCT_ARGS=(
        --source_feature_reshaper residual_temporal_conv
        --source_feature_reshaper_strength "${RESHAPER_STRENGTH}"
        --source_feature_reshaper_kernel_size "${RESHAPER_KERNEL_SIZE}"
        --source_feature_reshaper_reg_trade_off "${RESHAPER_REG_TRADE_OFF}"
        --source_feature_dual_path True
        --source_feature_dual_cls_trade_off "${DUAL_CLS_TRADE_OFF}"
        --source_feature_dual_relation_trade_off "${DUAL_RELATION_TRADE_OFF}"
        --source_phase_partition_mode "${SEG_MODE}"
        --source_segment_partition_mode "${SEG_MODE}"
        --source_phase_count "${SEG_COUNT}"
        --source_segment_count "${SEG_COUNT}"
        --source_phase_gap_threshold "${SEG_GAP_THRESHOLD}"
        --source_phase_min_points "${SEG_MIN_POINTS}"
        --source_phase_max_points "${SEG_MAX_POINTS}"
        --source_phase_max_span "${SEG_MAX_SPAN}"
        --source_phase_min_sample_points "${MIN_SAMPLE_POINTS}"
        --source_structure_loss_version segment_boundary_window_residual
        --source_structure_intra_trade_off "${SEGMENT_INTRA_TRADE_OFF}"
        --source_structure_trend_trade_off "${SEGMENT_TREND_TRADE_OFF}"
        --source_structure_segment_inter_trade_off "${SEGMENT_INTER_TRADE_OFF}"
        --source_structure_boundary_window_trade_off "${SEGMENT_BOUNDARY_TRADE_OFF}"
        --source_structure_boundary_window_size "${SEGMENT_BOUNDARY_WINDOW_SIZE}"
      )
      ;;
    global_segment)
      STRUCT_ARGS=(
        --source_feature_reshaper residual_temporal_conv
        --source_feature_reshaper_strength "${RESHAPER_STRENGTH}"
        --source_feature_reshaper_kernel_size "${RESHAPER_KERNEL_SIZE}"
        --source_feature_reshaper_reg_trade_off "${RESHAPER_REG_TRADE_OFF}"
        --source_feature_dual_path True
        --source_feature_dual_cls_trade_off "${DUAL_CLS_TRADE_OFF}"
        --source_feature_dual_relation_trade_off "${DUAL_RELATION_TRADE_OFF}"
        --source_phase_partition_mode "${SEG_MODE}"
        --source_segment_partition_mode "${SEG_MODE}"
        --source_phase_count "${SEG_COUNT}"
        --source_segment_count "${SEG_COUNT}"
        --source_phase_gap_threshold "${SEG_GAP_THRESHOLD}"
        --source_phase_min_points "${SEG_MIN_POINTS}"
        --source_phase_max_points "${SEG_MAX_POINTS}"
        --source_phase_max_span "${SEG_MAX_SPAN}"
        --source_phase_min_sample_points "${MIN_SAMPLE_POINTS}"
        --source_structure_loss_version v271_global_segment_basis
        --source_structure_intra_trade_off "${SEGMENT_INTRA_TRADE_OFF}"
        --source_structure_trend_trade_off "${SEGMENT_TREND_TRADE_OFF}"
        --source_structure_segment_inter_trade_off "${SEGMENT_INTER_TRADE_OFF}"
        --source_structure_boundary_window_trade_off "${SEGMENT_BOUNDARY_TRADE_OFF}"
        --source_structure_boundary_window_size "${SEGMENT_BOUNDARY_WINDOW_SIZE}"
        --source_structure_v271_trend_kernel_size "${TREND_KERNEL_SIZE}"
        --source_structure_v271_trend_smoothing_mode "${TREND_SMOOTHING_MODE}"
        --source_structure_v271_trend_bandwidth "${TREND_BANDWIDTH}"
        --source_structure_v271_trend_kernel "${TREND_KERNEL}"
        --source_structure_v271_trend_dynamics_trade_off "${TREND_DYNAMICS_TRADE_OFF}"
        --source_structure_v271_residual_variance_trade_off "${RESIDUAL_VARIANCE_TRADE_OFF}"
        --source_structure_v271_residual_energy_trade_off "${RESIDUAL_ENERGY_TRADE_OFF}"
        --source_structure_v271_residual_energy_margin "${RESIDUAL_ENERGY_MARGIN}"
        --source_structure_v271_segment_basis_trade_off "${SEGMENT_BASIS_TRADE_OFF}"
      )
      ;;
    *)
      echo "Unsupported source setting ${source_setting}" >&2
      exit 1
      ;;
  esac
}

adaptive_args() {
  local setting="$1"
  ADAPT_ARGS=(--timematch_source_structure_trade_off 0.0)
  if [ "${setting}" = "global_segment_fullcore" ] || [ "${setting}" = "global_fullcore" ]; then
    ADAPT_ARGS+=(
      --timematch_v271_adaptive_trade_off "${ADAPTIVE_TRADE_OFF}"
      --timematch_v271_adaptive_warmup_epochs "${ADAPTIVE_WARMUP_EPOCHS}"
      --timematch_v271_adaptive_ramp_epochs "${ADAPTIVE_RAMP_EPOCHS}"
      --timematch_v271_adaptive_min_score "${ADAPTIVE_MIN_SCORE}"
      --timematch_v271_adaptive_min_gate "${ADAPTIVE_MIN_GATE}"
      --timematch_v271_adaptive_min_points "${ADAPTIVE_MIN_POINTS}"
      --timematch_v271_adaptive_taper_mode "${ADAPTIVE_TAPER_MODE}"
      --timematch_v271_adaptive_taper_ratio "${ADAPTIVE_TAPER_RATIO}"
      --timematch_v271_adaptive_trend_trade_off "${ADAPTIVE_TREND_TRADE_OFF}"
      --timematch_v271_adaptive_trend_dynamics_trade_off "${ADAPTIVE_TREND_DYNAMICS_TRADE_OFF}"
      --timematch_v271_adaptive_residual_variance_trade_off "${ADAPTIVE_RESIDUAL_VARIANCE_TRADE_OFF}"
      --timematch_v271_adaptive_residual_energy_trade_off "${ADAPTIVE_RESIDUAL_ENERGY_TRADE_OFF}"
      --timematch_v271_adaptive_discover_in_da True
      --timematch_v271_adaptive_source_max_batches "${DA_DISCOVERY_SOURCE_MAX_BATCHES}"
      --timematch_v271_adaptive_target_max_batches "${DA_DISCOVERY_TARGET_MAX_BATCHES}"
      --timematch_v271_adaptive_atomic_bins "${DA_DISCOVERY_ATOMIC_BINS}"
      --timematch_v271_adaptive_shift_jitter "${DA_DISCOVERY_SHIFT_JITTER}"
      --timematch_v271_adaptive_soft_evidence "${DA_DISCOVERY_SOFT_EVIDENCE}"
      --timematch_v271_adaptive_max_margin "${DA_DISCOVERY_MAX_MARGIN}"
      --timematch_v271_adaptive_min_top2_mass "${DA_DISCOVERY_MIN_TOP2_MASS}"
      --timematch_v271_adaptive_prototype_temperature "${DA_DISCOVERY_PROTOTYPE_TEMPERATURE}"
      --timematch_v271_adaptive_baseline_pairs_per_sample "${DA_DISCOVERY_BASELINE_PAIRS_PER_SAMPLE}"
      --timematch_v271_adaptive_baseline_mode "${DA_DISCOVERY_BASELINE_MODE}"
      --timematch_v271_adaptive_shuffle_pair_baseline "${DA_DISCOVERY_SHUFFLE_PAIR_BASELINE}"
      --timematch_v271_adaptive_score_quantile "${DA_DISCOVERY_SCORE_QUANTILE}"
      --timematch_v271_adaptive_min_discovery_score "${DA_DISCOVERY_MIN_SCORE}"
      --timematch_v271_adaptive_min_discovery_ratio "${DA_DISCOVERY_MIN_RATIO}"
      --timematch_v271_adaptive_top_m_per_pair "${DA_DISCOVERY_TOP_M_PER_PAIR}"
      --timematch_v271_adaptive_max_supports "${DA_DISCOVERY_MAX_SUPPORTS}"
      --timematch_v271_adaptive_min_support_count "${DA_DISCOVERY_MIN_SUPPORT_COUNT}"
      --timematch_v271_adaptive_min_shift_stability "${DA_DISCOVERY_MIN_SHIFT_STABILITY}"
      --timematch_v271_adaptive_max_support_atoms "${DA_DISCOVERY_MAX_SUPPORT_ATOMS}"
      --timematch_v271_adaptive_max_interval_span "${DA_DISCOVERY_MAX_INTERVAL_SPAN}"
      --timematch_v271_adaptive_gate_score_high "${DA_DISCOVERY_GATE_SCORE_HIGH}"
      --timematch_v271_adaptive_gate_mode "${DA_DISCOVERY_GATE_MODE}"
      --timematch_v271_adaptive_gate_low "${DA_DISCOVERY_GATE_LOW}"
      --timematch_v271_adaptive_gate_high "${DA_DISCOVERY_GATE_HIGH}"
      --timematch_v271_adaptive_gate_light "${DA_DISCOVERY_GATE_LIGHT}"
      --timematch_v271_adaptive_gate_full "${DA_DISCOVERY_GATE_FULL}"
      --timematch_v271_adaptive_gate_ratio_high "${DA_DISCOVERY_GATE_RATIO_HIGH}"
      --timematch_v271_adaptive_gate_count_high "${DA_DISCOVERY_GATE_COUNT_HIGH}"
      --timematch_v271_adaptive_gate_source_sep_high "${DA_DISCOVERY_GATE_SOURCE_SEP_HIGH}"
      --timematch_v271_adaptive_gate_target_explain_high "${DA_DISCOVERY_GATE_TARGET_EXPLAIN_HIGH}"
      --timematch_v271_adaptive_discovery_apply_source_reshaper "${DA_DISCOVERY_APPLY_SOURCE_RESHAPER}"
    )
  fi
}

train_source_if_needed() {
  local dataset="$1"
  local task_key="$2"
  local source_setting="$3"
  local gpu="$4"
  local source_exp="${dataset,,}_${task_key}_${source_setting}_source_${STAMP}"
  local source_out="${OUT_ROOT}/${source_exp}"
  local source_log="${LOG_ROOT}/${source_exp}.log"

  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${source_out}/fold_0/model.pt" ]; then
    echo "[${dataset} ${task_key} ${source_setting}] source exists, skip"
    append_summary "${dataset}" "${task_key}" "${source_setting}" "source" "skipped" "" "${source_log}" "${source_out}"
    SOURCE_OUT="${source_out}"
    return 0
  fi
  local reused_source=""
  if reused_source="$(source_weights_for_task "${dataset}" "${task_key}" "${source_setting}")"; then
    echo "[${dataset} ${task_key} ${source_setting}] reuse source=${reused_source}"
    append_summary "${dataset}" "${task_key}" "${source_setting}" "source" "reused" "" "reuse:${reused_source}" "${reused_source}"
    SOURCE_OUT="${reused_source}"
    return 0
  fi

  echo "[${dataset} ${task_key} ${source_setting}] source GPU=${gpu}"
  if CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      "${DATA_ARGS[@]}" \
      --seed "${SEED}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${SOURCE_EPOCHS}" \
      --output_dir "${OUT_ROOT}" \
      --tensorboard_log_dir "${RUN_ROOT}" \
      --experiment_name "${source_exp}" \
      "${STRUCT_ARGS[@]}" \
      sourcephasecompact \
      > "${source_log}" 2>&1; then
    local f1
    f1="$(parse_test_f1 "${source_log}")"
    append_summary "${dataset}" "${task_key}" "${source_setting}" "source" "ok" "${f1}" "${source_log}" "${source_out}"
    SOURCE_OUT="${source_out}"
    return 0
  fi

  append_summary "${dataset}" "${task_key}" "${source_setting}" "source" "failed" "" "${source_log}" "${source_out}"
  SOURCE_OUT="${source_out}"
  return 1
}

run_da_setting() {
  local dataset="$1"
  local task_key="$2"
  local setting="$3"
  local source_out="$4"
  local gpu="$5"
  local da_exp="${dataset,,}_${task_key}_${setting}_timematch_${STAMP}"
  local da_out="${OUT_ROOT}/${da_exp}"
  local da_log="${LOG_ROOT}/${da_exp}.log"

  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${da_out}/fold_0/model.pt" ]; then
    echo "[${dataset} ${task_key} ${setting}] DA exists, skip"
    local f1_existing
    f1_existing="$(parse_test_f1 "${da_log}")"
    append_summary "${dataset}" "${task_key}" "${setting}" "da" "skipped" "${f1_existing}" "${da_log}" "${da_out}"
    return 0
  fi

  adaptive_args "${setting}"
  echo "[${dataset} ${task_key} ${setting}] TimeMatch GPU=${gpu}"
  if CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      "${DATA_ARGS[@]}" \
      --seed "${SEED}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${SOURCE_EPOCHS}" \
      --output_dir "${OUT_ROOT}" \
      --tensorboard_log_dir "${RUN_ROOT}" \
      --experiment_name "${da_exp}" \
      "${STRUCT_ARGS[@]}" \
      timematch \
      --weights "${source_out}" \
      --lr "${DA_LR}" \
      --epochs "${DA_EPOCHS}" \
      --steps_per_epoch "${DA_STEPS}" \
      --estimate_shift True \
      --max_temporal_shift "${MAX_SHIFT}" \
      --sample_size "${SHIFT_SAMPLE}" \
      --shift_source True \
      --balance_source True \
      "${ADAPT_ARGS[@]}" \
      > "${da_log}" 2>&1; then
    local f1
    f1="$(parse_test_f1 "${da_log}")"
    append_summary "${dataset}" "${task_key}" "${setting}" "da" "ok" "${f1}" "${da_log}" "${da_out}"
    return 0
  fi

  append_summary "${dataset}" "${task_key}" "${setting}" "da" "failed" "" "${da_log}" "${da_out}"
  return 1
}

run_task_pipeline() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local src_path="${4:-}"
  local tgt_path="${5:-}"
  local gpu="$6"
  local task_key="${src}_to_${tgt}"

  timing_args "${dataset}"
  dataset_args "${dataset}" "${src}" "${tgt}" "${src_path}" "${tgt_path}"

  local trained_source_settings=()
  for setting in ${SETTINGS}; do
    local source_setting
    source_setting="$(source_setting_for "${setting}")"
    local already_trained=0
    for seen in "${trained_source_settings[@]}"; do
      if [ "${seen}" = "${source_setting}" ]; then
        already_trained=1
        break
      fi
    done
    structure_args "${source_setting}"
    local source_exp="${dataset,,}_${task_key}_${source_setting}_source_${STAMP}"
    local source_out="${OUT_ROOT}/${source_exp}"
    if [ "${already_trained}" = "0" ]; then
      if train_source_if_needed "${dataset}" "${task_key}" "${source_setting}" "${gpu}"; then
        trained_source_settings+=("${source_setting}")
        source_out="${SOURCE_OUT}"
      else
        echo "[${dataset} ${task_key} ${source_setting}] source failed; skip dependent DA" >&2
        continue
      fi
    fi
    run_da_setting "${dataset}" "${task_key}" "${setting}" "${source_out}" "${gpu}" || true
  done
}

schedule_dataset() {
  local dataset="$1"
  local -n specs_ref="$2"
  for spec in "${specs_ref[@]}"; do
    local src tgt src_path tgt_path
    IFS='|' read -r src tgt src_path tgt_path <<< "${spec}"
    local task_key="${src}_to_${tgt}"
    if ! should_run_task "${dataset}" "${task_key}"; then
      continue
    fi
    wait_for_slot
    local gpu="${GPUS[$((JOB_INDEX % ${#GPUS[@]}))]}"
    JOB_INDEX=$((JOB_INDEX + 1))
    run_task_pipeline "${dataset}" "${src}" "${tgt}" "${src_path:-}" "${tgt_path:-}" "${gpu}" &
    sleep 2
  done
}

echo "RUN_TAG=${RUN_TAG}"
echo "DATASETS=${DATASETS}"
echo "SETTINGS=${SETTINGS}"
echo "GPUS=${GPUS[*]}"
echo "Logs: ${LOG_ROOT}"
echo "Outputs: ${OUT_ROOT}"
echo "Summary: ${SUMMARY_FILE}"

for dataset in ${DATASETS}; do
  case "${dataset}" in
    REMOTE)
      schedule_dataset "REMOTE" REMOTE_TASKS
      ;;
    HAR)
      schedule_dataset "HAR" HAR_TASKS
      ;;
    HHAR|HHAR_SA)
      schedule_dataset "HHAR_SA" HHAR_TASKS
      ;;
    *)
      echo "Unknown dataset ${dataset}; expected REMOTE, HAR, HHAR_SA" >&2
      exit 1
      ;;
  esac
done

wait
echo "v2.7.2 four-setting full matrix finished."
echo "Summary: ${SUMMARY_FILE}"
