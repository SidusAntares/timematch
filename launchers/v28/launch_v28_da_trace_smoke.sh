#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANED_ROOT="${CLEANED_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OLD_ROOT="${OLD_ROOT:-/data/user/timematch_old_da_f04e1e0}"
OLD_COMMIT="${OLD_COMMIT:-f04e1e06805270d4e98db688ae869fbdeb6493b6}"
SMOOTH_COMMIT="${SMOOTH_COMMIT:-89d9df4e52744cb955168b0d203a2ddd61c3199e}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
RUN_TAG="${RUN_TAG:-v28_da_causal_audit_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${CLEANED_ROOT}/logs/${RUN_TAG}}"
SOURCE_INVENTORY="${SOURCE_INVENTORY:-${CLEANED_ROOT}/logs/v28_cleaned_full12_20260710_120037/source_checkpoint_inventory.tsv}"
GPU="${GPU:-0}"

mkdir -p "${LOG_ROOT}/trace" "${LOG_ROOT}/common_eval"
if [[ -f "${OLD_ROOT}/.v28_old_commit" ]]; then
  [[ "$(cat "${OLD_ROOT}/.v28_old_commit")" == "${OLD_COMMIT}" ]] || {
    echo "ERROR old source marker mismatch: ${OLD_ROOT}" >&2
    exit 2
  }
elif git -C "${OLD_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  [[ "$(git -C "${OLD_ROOT}" rev-parse HEAD)" == "${OLD_COMMIT}" ]] || {
    echo "ERROR old worktree commit mismatch: ${OLD_ROOT}" >&2
    exit 2
  }
else
  echo "ERROR missing exported old source: ${OLD_ROOT}" >&2
  echo "Run bash ./sync_v28_old_da_to_server.sh locally before starting this audit." >&2
  exit 2
fi
if [[ ! -f "${OLD_ROOT}/.v276_smooth_commit" ]] || \
   [[ "$(cat "${OLD_ROOT}/.v276_smooth_commit")" != "${SMOOTH_COMMIT}" ]] || \
   [[ ! -f "${OLD_ROOT}/v276_source_raw_compactness.py" ]]; then
  echo "ERROR missing matching v276 smooth source in ${OLD_ROOT}" >&2
  echo "Run bash ./sync_v28_old_da_to_server.sh locally before starting this audit." >&2
  exit 2
fi

python -B "${CLEANED_ROOT}/tools/compare_v28_smooth_loss.py" \
  --old_source_file "${OLD_ROOT}/v276_source_raw_compactness.py" \
  --output "${LOG_ROOT}/smooth_loss_equivalence.txt"

checkpoint="$(awk -F '\t' '
  NR == 1 { for (i=1; i<=NF; i++) idx[$i]=i; next }
  $idx["source_domain"] == "AT1" && $idx["seed"] == "1" && $idx["source_config"] == "smooth_k3" {
    print $idx["checkpoint_path"]; exit
  }
' "${SOURCE_INVENTORY}")"
[[ -f "${checkpoint}" ]] || { echo "ERROR missing checkpoint: ${checkpoint}" >&2; exit 2; }
source_dir="$(dirname "$(dirname "${checkpoint}")")"
trace_inputs="${LOG_ROOT}/trace/trace_inputs.pt"

common_train_args=(
  --data_root "${DATA_ROOT}" --closed_set True --with_shift_aug False
  --num_workers 0 --seed 1 --source "austria/33UVP/2017" --target "france/31TCJ/2017"
)

CUDA_VISIBLE_DEVICES="${GPU}" python -B "${CLEANED_ROOT}/tools/trace_timematch_da_step.py" \
  --mode capture --implementation cleaned --repo_root "${CLEANED_ROOT}" \
  --trace_inputs "${trace_inputs}" --trace_output "${LOG_ROOT}/trace/trace_cleaned_eps1e5.tsv" \
  --steps 3 --shift_batches 3 -- \
  "${common_train_args[@]}" --output_dir "${LOG_ROOT}/trace/cleaned_eps1e5_outputs" \
  -e trace_cleaned_eps1e5 timematch --weights "${source_dir}" --epochs 1 --steps_per_epoch 3 \
  --sample_size 3 --timematch_topk_shifts 5 --timematch_shift_score_epsilon 1e-5

CUDA_VISIBLE_DEVICES="${GPU}" python -B "${CLEANED_ROOT}/tools/trace_timematch_da_step.py" \
  --mode replay --implementation old --repo_root "${OLD_ROOT}" \
  --trace_inputs "${trace_inputs}" --trace_output "${LOG_ROOT}/trace/trace_old.tsv" \
  --steps 3 --shift_batches 3 -- \
  "${common_train_args[@]}" --output_dir "${LOG_ROOT}/trace/old_outputs" \
  -e trace_old timematch --weights "${source_dir}" --epochs 1 --steps_per_epoch 3 --sample_size 3

CUDA_VISIBLE_DEVICES="${GPU}" python -B "${CLEANED_ROOT}/tools/trace_timematch_da_step.py" \
  --mode replay --implementation cleaned --repo_root "${CLEANED_ROOT}" \
  --trace_inputs "${trace_inputs}" --trace_output "${LOG_ROOT}/trace/trace_cleaned_eps1e12.tsv" \
  --steps 3 --shift_batches 3 -- \
  "${common_train_args[@]}" --output_dir "${LOG_ROOT}/trace/cleaned_eps1e12_outputs" \
  -e trace_cleaned_eps1e12 timematch --weights "${source_dir}" --epochs 1 --steps_per_epoch 3 \
  --sample_size 3 --timematch_topk_shifts 5 --timematch_shift_score_epsilon 1e-12

python -B "${CLEANED_ROOT}/tools/trace_timematch_da_step.py" \
  --mode compare --old_trace "${LOG_ROOT}/trace/trace_old.tsv" \
  --cleaned_trace "${LOG_ROOT}/trace/trace_cleaned_eps1e5.tsv" \
  --trace_output "${LOG_ROOT}/trace/trace_diff_old_vs_cleaned.tsv"
python -B "${CLEANED_ROOT}/tools/trace_timematch_da_step.py" \
  --mode compare --old_trace "${LOG_ROOT}/trace/trace_cleaned_eps1e5.tsv" \
  --cleaned_trace "${LOG_ROOT}/trace/trace_cleaned_eps1e12.tsv" \
  --trace_output "${LOG_ROOT}/trace/trace_diff_epsilon.tsv"

for implementation in old cleaned_eps1e5; do
  checkpoint_path="${LOG_ROOT}/trace/${implementation}_outputs/trace_${implementation}/fold_0/model.pt"
  [[ "${implementation}" == "cleaned_eps1e5" ]] && checkpoint_path="${LOG_ROOT}/trace/cleaned_eps1e5_outputs/trace_cleaned_eps1e5/fold_0/model.pt"
  CUDA_VISIBLE_DEVICES="${GPU}" python -B "${CLEANED_ROOT}/tools/evaluate_common_timematch_checkpoint.py" \
    --checkpoint "${checkpoint_path}" --data_root "${DATA_ROOT}" \
    --source "austria/33UVP/2017" --target "france/31TCJ/2017" \
    --seed 1 --closed_set True --num_workers 0 \
    --output "${LOG_ROOT}/common_eval/${implementation}.json"
done

echo "TRACE_SMOKE_OK|log_root=${LOG_ROOT}|old_commit=${OLD_COMMIT}|checkpoint=${checkpoint}"
