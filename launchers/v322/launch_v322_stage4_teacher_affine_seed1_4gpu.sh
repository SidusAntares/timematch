#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_TAG="${RUN_TAG:-v322_stage4_teacher_affine_seed1_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/$RUN_TAG}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REPOSITORY_BRANCH="${REPOSITORY_BRANCH:-exp/v322-affine-temporal-alignment}"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-26c2fae6c8f90825222362df753c1754bad5a4ca}"

mkdir -p "$LOG_ROOT"
STATUS_FILE="$LOG_ROOT/job_status.tsv"
printf 'task\tmode\tgpu\tstatus\tlog\n' > "$STATUS_FILE"

run_job() {
    local gpu="$1"
    local task="$2"
    local mode="$3"
    local source="$4"
    local target="$5"
    local weights="$6"
    local stretch_json="$7"
    local experiment="v322_stage4_${task}_${mode}_seed1_${RUN_TAG}"
    local log="$LOG_ROOT/${task}_${mode}.log"
    local command=(
        python -B train.py
        --data_root "$DATA_ROOT"
        --closed_set True
        --with_shift_aug False
        --seed 1
        --device cuda
        --batch_size 128
        --num_workers "$NUM_WORKERS"
        -e "$experiment"
        --source "$source"
        --target "$target"
        timematch
        --weights "$weights"
        --epochs 20
        --steps_per_epoch 500
        --estimate_shift True
        --shift_estimator IS
        --sample_size 100
        --timematch_shift_policy fixed_initial_shift
        --timematch_shift_score_epsilon 1e-5
        --timematch_target_teacher_position_mode "$mode"
        --timematch_diagnostic_task "$task"
        --output_student True
    )
    if [[ "$mode" == "affine" ]]; then
        command+=(
            --timematch_affine_stretch_json "$stretch_json"
            --timematch_affine_task "$task"
            --timematch_affine_repository_branch "$REPOSITORY_BRANCH"
            --timematch_affine_repository_commit "$REPOSITORY_COMMIT"
        )
    fi
    (
        cd "$ROOT_DIR"
        CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" > "$log" 2>&1
    )
    local status=$?
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$task" "$mode" "$gpu" "$status" "$log" >> "$STATUS_FILE"
    return "$status"
}

run_job 0 AT1_to_FR2 global_only \
    austria/33UVP/2017 france/31TCJ/2017 \
    outputs/v28_cleaned_source_AT1_smooth_k3_seed1 "" &
pid0=$!

run_job 1 AT1_to_FR2 affine \
    austria/33UVP/2017 france/31TCJ/2017 \
    outputs/v28_cleaned_source_AT1_smooth_k3_seed1 \
    logs/v322_stretch_formal_AT1_to_FR2_20260714_145935/AT1_to_FR2.json &
pid1=$!

run_job 2 FR2_to_FR1 global_only \
    france/31TCJ/2017 france/30TXT/2017 \
    outputs/v28_cleaned_source_FR2_smooth_k3_seed1 "" &
pid2=$!

run_job 3 FR2_to_FR1 affine \
    france/31TCJ/2017 france/30TXT/2017 \
    outputs/v28_cleaned_source_FR2_smooth_k3_seed1 \
    logs/v322_stretch_formal_FR2_to_FR1_20260714_175936/FR2_to_FR1.json &
pid3=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

echo "V322_STAGE4_DONE|run_tag=$RUN_TAG|failed=$failed|status=$STATUS_FILE"
exit "$failed"
