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

# PRA-only follow-up script: reuse completed source-only and baseline TimeMatch runs.
RUN_SOURCE_ONLY="0"
RUN_TIMEMATCH_BASELINE="0"
RUN_TIMEMATCH_PRA="1"
SKIP_EXISTING_RUNS="${SKIP_EXISTING_RUNS:-1}"

# Default PRA configuration for the current FR1 -> DK1 follow-up experiment.
# `PRA_RUN_LABEL` is the human-readable experiment name.
# `PRA_CONFIG_TAG` keeps the key hyperparameters visible in the output folder name.
PRA_RUN_LABEL="${PRA_RUN_LABEL:-pra}"
PRA_CONFIG_TAG="${PRA_CONFIG_TAG:-t002_w8_m8_pt093}"
PRA_EXP_SUFFIX="${PRA_EXP_SUFFIX:-${PRA_RUN_LABEL}_${PRA_CONFIG_TAG}}"
PRA_TRADE_OFF="${PRA_TRADE_OFF:-0.02}"
PRA_WARMUP_EPOCHS="${PRA_WARMUP_EPOCHS:-8}"
PRA_MIN_SAMPLES_PER_CLASS="${PRA_MIN_SAMPLES_PER_CLASS:-8}"
PRA_PSEUDO_THRESHOLD="${PRA_PSEUDO_THRESHOLD:-0.93}"

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

target_file_name() {
    local target_path="$1"
    echo "${target_path//\//_}"
}

metrics_path_for_exp() {
    local exp_name="$1"
    local target_path="$2"
    echo "$SCRIPT_DIR/outputs/$exp_name/fold_0/test_metrics_$(target_file_name "$target_path").json"
}

class_report_path_for_exp() {
    local exp_name="$1"
    local target_path="$2"
    echo "$SCRIPT_DIR/outputs/$exp_name/fold_0/class_report_$(target_file_name "$target_path").txt"
}

assert_file_exists() {
    local file_path="$1"
    local description="$2"
    if [[ ! -f "$file_path" ]]; then
        echo "[ERROR] Missing $description: $file_path" >&2
        exit 1
    fi
}

extract_macro_f1() {
    local metrics_path="$1"
    "$PYTHON_EXEC" -c "import json; print(json.load(open(r'''$metrics_path'''))['macro_f1'])"
}

extract_pra_flag() {
    local metrics_path="$1"
    "$PYTHON_EXEC" -c "import json; data=json.load(open(r'''$metrics_path''')); print(data.get('use_prototype_relation_alignment', data.get('metadata', {}).get('use_prototype_relation_alignment', False)))"
}

copy_artifact_if_exists() {
    local source_path="$1"
    local destination_path="$2"
    if [[ -f "$source_path" ]]; then
        cp "$source_path" "$destination_path"
    fi
}

write_summary_csv() {
    local source_path="$1"
    local target_path="$2"
    local seed="$3"
    local source_exp="$4"
    local timematch_exp="$5"
    local timematch_pra_exp="$6"
    local result_dir="$7"

    local csv_path
    local source_metrics_path
    local timematch_metrics_path
    local timematch_pra_metrics_path
    local source_macro_f1
    local timematch_macro_f1
    local timematch_pra_macro_f1
    local timematch_pra_flag
    local delta_pra_vs_timematch
    local delta_pra_vs_source

    source_metrics_path="$(metrics_path_for_exp "$source_exp" "$target_path")"
    timematch_metrics_path="$(metrics_path_for_exp "$timematch_exp" "$target_path")"
    timematch_pra_metrics_path="$(metrics_path_for_exp "$timematch_pra_exp" "$target_path")"

    assert_file_exists "$source_metrics_path" "source-only metrics"
    assert_file_exists "$timematch_metrics_path" "baseline TimeMatch metrics"
    assert_file_exists "$timematch_pra_metrics_path" "PRA TimeMatch metrics"

    source_macro_f1="$(extract_macro_f1 "$source_metrics_path")"
    timematch_macro_f1="$(extract_macro_f1 "$timematch_metrics_path")"
    timematch_pra_macro_f1="$(extract_macro_f1 "$timematch_pra_metrics_path")"
    timematch_pra_flag="$(extract_pra_flag "$timematch_pra_metrics_path")"
    delta_pra_vs_timematch="$("$PYTHON_EXEC" -c "print(float('$timematch_pra_macro_f1') - float('$timematch_macro_f1'))")"
    delta_pra_vs_source="$("$PYTHON_EXEC" -c "print(float('$timematch_pra_macro_f1') - float('$source_macro_f1'))")"

    csv_path="$result_dir/results.csv"
    {
        echo "source,target,seed,source_experiment,timematch_experiment,timematch_pra_experiment,source_macro_f1,timematch_macro_f1,timematch_pra_macro_f1,pra_minus_timematch_macro_f1,pra_minus_source_macro_f1,timematch_pra_enabled,pra_trade_off,pra_warmup_epochs,pra_min_samples_per_class,pra_pseudo_threshold"
        echo "$source_path,$target_path,$seed,$source_exp,$timematch_exp,$timematch_pra_exp,$source_macro_f1,$timematch_macro_f1,$timematch_pra_macro_f1,$delta_pra_vs_timematch,$delta_pra_vs_source,$timematch_pra_flag,$PRA_TRADE_OFF,$PRA_WARMUP_EPOCHS,$PRA_MIN_SAMPLES_PER_CLASS,$PRA_PSEUDO_THRESHOLD"
    } > "$csv_path"

    echo "[INFO] Summary CSV saved to: $csv_path"
}

write_classwise_csv() {
    local target_path="$1"
    local source_exp="$2"
    local timematch_exp="$3"
    local timematch_pra_exp="$4"
    local result_dir="$5"

    local source_report_path
    local timematch_report_path
    local timematch_pra_report_path
    local csv_path

    source_report_path="$(class_report_path_for_exp "$source_exp" "$target_path")"
    timematch_report_path="$(class_report_path_for_exp "$timematch_exp" "$target_path")"
    timematch_pra_report_path="$(class_report_path_for_exp "$timematch_pra_exp" "$target_path")"

    assert_file_exists "$source_report_path" "source-only class report"
    assert_file_exists "$timematch_report_path" "baseline TimeMatch class report"
    assert_file_exists "$timematch_pra_report_path" "PRA TimeMatch class report"

    csv_path="$result_dir/classwise_comparison.csv"
    "$PYTHON_EXEC" -c "
import csv
import re
from pathlib import Path

def parse_report(path):
    rows = {}
    pattern = re.compile(r'^\s*(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s*$')
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.rstrip()
        match = pattern.match(line)
        if not match:
            continue
        class_name, precision, recall, f1_score, support = match.groups()
        class_name = class_name.strip()
        if class_name in {'macro avg', 'weighted avg', 'accuracy'}:
            continue
        rows[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1_score),
            'support': int(support),
        }
    return rows

source_rows = parse_report(r'''$source_report_path''')
timematch_rows = parse_report(r'''$timematch_report_path''')
pra_rows = parse_report(r'''$timematch_pra_report_path''')
all_classes = sorted(set(source_rows) | set(timematch_rows) | set(pra_rows))

with open(r'''$csv_path''', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'class_name',
        'source_precision', 'source_recall', 'source_f1', 'source_support',
        'timematch_precision', 'timematch_recall', 'timematch_f1', 'timematch_support',
        'pra_precision', 'pra_recall', 'pra_f1', 'pra_support',
        'pra_minus_timematch_f1', 'pra_minus_source_f1'
    ])
    for class_name in all_classes:
        source = source_rows.get(class_name, {})
        timematch = timematch_rows.get(class_name, {})
        pra = pra_rows.get(class_name, {})
        source_f1 = source.get('f1')
        timematch_f1 = timematch.get('f1')
        pra_f1 = pra.get('f1')
        writer.writerow([
            class_name,
            source.get('precision', ''),
            source.get('recall', ''),
            source_f1 if source_f1 is not None else '',
            source.get('support', ''),
            timematch.get('precision', ''),
            timematch.get('recall', ''),
            timematch_f1 if timematch_f1 is not None else '',
            timematch.get('support', ''),
            pra.get('precision', ''),
            pra.get('recall', ''),
            pra_f1 if pra_f1 is not None else '',
            pra.get('support', ''),
            '' if pra_f1 is None or timematch_f1 is None else pra_f1 - timematch_f1,
            '' if pra_f1 is None or source_f1 is None else pra_f1 - source_f1,
        ])
" 

    copy_artifact_if_exists "$source_report_path" "$result_dir/${source_exp}_class_report_$(target_file_name "$target_path").txt"
    copy_artifact_if_exists "$timematch_report_path" "$result_dir/${timematch_exp}_class_report_$(target_file_name "$target_path").txt"
    copy_artifact_if_exists "$timematch_pra_report_path" "$result_dir/${timematch_pra_exp}_class_report_$(target_file_name "$target_path").txt"

    echo "[INFO] Class-wise comparison CSV saved to: $csv_path"
}

run_or_skip_experiment() {
    local should_run="$1"
    local exp_name="$2"
    local metrics_path="$3"
    shift 3

    if [[ "$should_run" == "1" ]]; then
        if [[ "$SKIP_EXISTING_RUNS" == "1" && -f "$metrics_path" ]]; then
            echo "[INFO] Skip existing run: $exp_name"
            return
        fi
        echo "[INFO] Running experiment: $exp_name"
        "$PYTHON_EXEC" ./train.py "$@"
        return
    fi

    assert_file_exists "$metrics_path" "reused metrics for $exp_name"
    echo "[INFO] Reusing existing experiment: $exp_name"
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
    local source_metrics_path
    local timematch_metrics_path
    local timematch_pra_metrics_path
    local timestamp
    local result_dir

    echo "--------------------------------------------------"
    echo "[INFO] Starting experiment flow: source=$source_path, target=$target_path, seed=$seed"
    echo "[INFO] RUN_SOURCE_ONLY=$RUN_SOURCE_ONLY RUN_TIMEMATCH_BASELINE=$RUN_TIMEMATCH_BASELINE RUN_TIMEMATCH_PRA=$RUN_TIMEMATCH_PRA"
    echo "[INFO] PRA naming: label=$PRA_RUN_LABEL config_tag=$PRA_CONFIG_TAG exp_suffix=$PRA_EXP_SUFFIX"
    echo "[INFO] PRA config: trade_off=$PRA_TRADE_OFF warmup=$PRA_WARMUP_EPOCHS min_samples=$PRA_MIN_SAMPLES_PER_CLASS pseudo_threshold=$PRA_PSEUDO_THRESHOLD"
    echo "--------------------------------------------------"

    source_tag="$(match "$source_path")"
    target_tag="$(match "$target_path")"
    source_exp="pseltae_${source_tag}"
    timematch_exp="timematch_${source_tag}_to_${target_tag}"
    timematch_pra_exp="timematch_${source_tag}_to_${target_tag}_${PRA_EXP_SUFFIX}"

    source_metrics_path="$(metrics_path_for_exp "$source_exp" "$target_path")"
    timematch_metrics_path="$(metrics_path_for_exp "$timematch_exp" "$target_path")"
    timematch_pra_metrics_path="$(metrics_path_for_exp "$timematch_pra_exp" "$target_path")"

    run_or_skip_experiment \
        "$RUN_SOURCE_ONLY" \
        "$source_exp" \
        "$source_metrics_path" \
        -e "$source_exp" \
        --source "$source_path" \
        --target "$target_path" \
        --seed "$seed"

    run_or_skip_experiment \
        "$RUN_TIMEMATCH_BASELINE" \
        "$timematch_exp" \
        "$timematch_metrics_path" \
        -e "$timematch_exp" \
        --source "$source_path" \
        --target "$target_path" \
        --seed "$seed" \
        timematch \
        --weights "outputs/$source_exp"

    run_or_skip_experiment \
        "$RUN_TIMEMATCH_PRA" \
        "$timematch_pra_exp" \
        "$timematch_pra_metrics_path" \
        -e "$timematch_pra_exp" \
        --source "$source_path" \
        --target "$target_path" \
        --seed "$seed" \
        timematch \
        --weights "outputs/$source_exp" \
        --use_pra \
        --pra_trade_off "$PRA_TRADE_OFF" \
        --pra_warmup_epochs "$PRA_WARMUP_EPOCHS" \
        --pra_min_samples_per_class "$PRA_MIN_SAMPLES_PER_CLASS" \
        --pseudo_threshold "$PRA_PSEUDO_THRESHOLD"

    timestamp="$(date +"%Y%m%d_%H%M%S")"
    result_dir="$SCRIPT_DIR/result/$source_tag/$target_tag/$timestamp"
    mkdir -p "$result_dir"

    write_summary_csv "$source_path" "$target_path" "$seed" "$source_exp" "$timematch_exp" "$timematch_pra_exp" "$result_dir"
    write_classwise_csv "$target_path" "$source_exp" "$timematch_exp" "$timematch_pra_exp" "$result_dir"
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
