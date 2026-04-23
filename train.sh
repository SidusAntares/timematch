#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/ti/bin/python"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Centralized configuration
SOURCES=("france/30TXT/2017")
TARGETS=("denmark/32VNH/2017")
SEEDS=(111)

match() {
    local domain="$1"
    if [[ "$domain" == "france/30TXT/2017" ]]; then
        echo "FR1"
    elif [[ "$domain" == "france/31TCJ/2017" ]]; then
        echo "FR2"
    elif [[ "$domain" == "denmark/32VNH/2017" ]]; then
        echo "DK1"
    else
        echo "AT1"
    fi
}

extract_macro_f1() {
    local metrics_path="$1"
    "$PYTHON_EXEC" -c "import json; print(json.load(open(r'''$metrics_path'''))['macro_f1'])"
}

extract_pra_flag() {
    local metrics_path="$1"
    "$PYTHON_EXEC" -c "import json; print(json.load(open(r'''$metrics_path'''))['use_prototype_relation_alignment'])"
}

write_summary_csv() {
    local source_path="$1"
    local target_path="$2"
    local seed="$3"
    local source_exp="$4"
    local timematch_exp="$5"
    local timematch_pra_exp="$6"

    local source_tag
    local target_tag
    local timestamp
    local results_root
    local result_dir
    local csv_path
    local target_file_name
    local source_metrics_path
    local timematch_metrics_path
    local timematch_pra_metrics_path
    local source_macro_f1
    local timematch_macro_f1
    local timematch_pra_macro_f1
    local timematch_pra_flag

    source_tag="$(match "$source_path")"
    target_tag="$(match "$target_path")"
    timestamp="$(date +"%Y%m%d_%H%M%S")"
    results_root="$SCRIPT_DIR/result"
    result_dir="$results_root/$source_tag/$target_tag/$timestamp"
    mkdir -p "$result_dir"

    target_file_name="${target_path//\//_}"
    source_metrics_path="$SCRIPT_DIR/outputs/$source_exp/fold_0/test_metrics_${target_file_name}.json"
    timematch_metrics_path="$SCRIPT_DIR/outputs/$timematch_exp/fold_0/test_metrics_${target_file_name}.json"
    timematch_pra_metrics_path="$SCRIPT_DIR/outputs/$timematch_pra_exp/fold_0/test_metrics_${target_file_name}.json"

    source_macro_f1="$(extract_macro_f1 "$source_metrics_path")"
    timematch_macro_f1="$(extract_macro_f1 "$timematch_metrics_path")"
    timematch_pra_macro_f1="$(extract_macro_f1 "$timematch_pra_metrics_path")"
    timematch_pra_flag="$(extract_pra_flag "$timematch_pra_metrics_path")"

    csv_path="$result_dir/results.csv"
    {
        echo "source,target,source_f1,timematch_macro_f1,timematch_pra_macro_f1,timematch_pra_enabled,timestamp,source_tag,target_tag,seed,source_experiment,timematch_experiment,timematch_pra_experiment"
        echo "$source_path,$target_path,$source_macro_f1,$timematch_macro_f1,$timematch_pra_macro_f1,$timematch_pra_flag,$timestamp,$source_tag,$target_tag,$seed,$source_exp,$timematch_exp,$timematch_pra_exp"
    } > "$csv_path"

    echo "[INFO] Experiment results saved to: $csv_path"
}

run_experiment() {
    local source_path="$1"
    local target_path="$2"
    local seed="$3"
    local source_tag
    local target_tag
    local source_exp
    local timematch_exp
    local timematch_pra_exp

    echo "--------------------------------------------------"
    echo "[INFO] Starting training: source=$source_path, target=$target_path, seed=$seed"
    echo "--------------------------------------------------"

    source_tag="$(match "$source_path")"
    target_tag="$(match "$target_path")"
    source_exp="pseltae_${source_tag}"
    timematch_exp="timematch_${source_tag}_to_${target_tag}"
    timematch_pra_exp="timematch_${source_tag}_to_${target_tag}_pra"

    "$PYTHON_EXEC" ./train.py \
        -e "$source_exp" \
        --source "$source_path" \
        --target "$target_path" \
        --seed "$seed"

    "$PYTHON_EXEC" ./train.py \
        -e "$timematch_exp" \
        --source "$source_path" \
        --target "$target_path" \
        --seed "$seed" \
        timematch \
        --weights "outputs/$source_exp"

    "$PYTHON_EXEC" ./train.py \
        -e "$timematch_pra_exp" \
        --source "$source_path" \
        --target "$target_path" \
        --seed "$seed" \
        timematch \
        --weights "outputs/$source_exp" \
        --use_pra

    write_summary_csv "$source_path" "$target_path" "$seed" "$source_exp" "$timematch_exp" "$timematch_pra_exp"
}

for source in "${SOURCES[@]}"; do
    for target in "${TARGETS[@]}"; do
        if [[ "$source" == "$target" ]]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            run_experiment "$source" "$target" "$seed"
        done
    done
done

echo "[SUCCESS] Finished all experiments."
