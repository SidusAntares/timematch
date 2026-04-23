#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/ti/bin/python"

SOURCES=("france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017")
SEEDS=(111)

match() {
    local domain="$1"
    if [[ "$domain" == 'france/30TXT/2017' ]]; then
        echo 'FR1'
    elif [[ "$domain" == 'france/31TCJ/2017' ]]; then
        echo 'FR2'
    elif [[ "$domain" == 'denmark/32VNH/2017' ]]; then
        echo 'DK1'
    else
        echo 'AT1'
    fi
}

extract_separability() {
    local result_path="$1"
    "$PYTHON_EXEC" -c "import json; print(json.load(open(r'''$result_path'''))['separability'])"
}

write_summary_csv() {
    local source_path="$1"
    local target_path="$2"
    local seed="$3"
    local exp_name="$4"

    local source_tag
    local target_tag
    local timestamp
    local results_root
    local result_dir
    local csv_path
    local source_file_name
    local separability_path
    local separability

    source_tag="$(match "$source_path")"
    target_tag="$(match "$target_path")"
    timestamp="$(date +"%Y%m%d_%H%M%S")"
    results_root="$SCRIPT_DIR/result"
    result_dir="$results_root/$source_tag/$target_tag/$timestamp"
    mkdir -p "$result_dir"

    source_file_name="${source_path//\//_}"
    separability_path="$SCRIPT_DIR/outputs/$exp_name/separability_${source_file_name}.json"
    separability="$(extract_separability "$separability_path")"

    csv_path="$result_dir/results.csv"
    {
        echo "source,target,separability,timestamp,source_tag,target_tag,seed,experiment"
        echo "$source_path,$target_path,$separability,$timestamp,$source_tag,$target_tag,$seed,$exp_name"
    } > "$csv_path"

    echo "[INFO] Separability result saved to: $csv_path"
}

run_experiment() {
    local source_path="$1"
    local seed="$2"
    local source_tag
    local exp_name

    echo "--------------------------------------------------"
    echo "[INFO] Computing separability: source=$source_path, seed=$seed"
    echo "--------------------------------------------------"

    source_tag="$(match "$source_path")"
    exp_name="separability_${source_tag}"

    "$PYTHON_EXEC" ./train.py \
        -e "$exp_name" \
        --source "$source_path" \
        --target "$source_path" \
        --seed "$seed" \
        --compute_separability

    write_summary_csv "$source_path" "$source_path" "$seed" "$exp_name"
}

for source in "${SOURCES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "$source" "$seed"
    done
done

echo "[SUCCESS] Completed separability runs for ${#SOURCES[@]} datasets."
