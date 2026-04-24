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

# Follow-up script: by default, reuse completed source-only and baseline TimeMatch runs,
# then launch a small PRA sweep for FR1 -> DK1.
RUN_SOURCE_ONLY="0"
RUN_TIMEMATCH_BASELINE="0"
RUN_TIMEMATCH_PRA="1"
SKIP_EXISTING_RUNS="${SKIP_EXISTING_RUNS:-1}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

# PRA sweep configurations for the final focused FR1 -> DK1 validation round.
# Format:
#   config_tag|point_trade_off|edge_trade_off|warmup_epochs|min_samples_per_class|pseudo_threshold|bank_momentum
PRA_CONFIGS=(
    "p0002_e0005_w8_m2_pt090_b097|0.002|0.005|8|2|0.90|0.97"
    "p0001_e0005_w8_m2_pt090_b097|0.001|0.005|8|2|0.90|0.97"
    "p0002_e0005_w8_m2_pt092_b097|0.002|0.005|8|2|0.92|0.97"
)

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

extract_accuracy() {
    local metrics_path="$1"
    "$PYTHON_EXEC" -c "import json; print(json.load(open(r'''$metrics_path'''))['accuracy'])"
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
    local pra_point_trade_off="$8"
    local pra_trade_off="$9"
    local pra_warmup_epochs="${10}"
    local pra_min_samples_per_class="${11}"
    local pra_pseudo_threshold="${12}"
    local pra_bank_momentum="${13}"

    local csv_path
    local source_metrics_path
    local timematch_metrics_path
    local timematch_pra_metrics_path
    local source_accuracy
    local source_macro_f1
    local timematch_accuracy
    local timematch_macro_f1
    local timematch_pra_accuracy
    local timematch_pra_macro_f1
    local timematch_pra_flag
    local delta_acc_pra_vs_timematch
    local delta_acc_pra_vs_source
    local delta_pra_vs_timematch
    local delta_pra_vs_source

    source_metrics_path="$(metrics_path_for_exp "$source_exp" "$target_path")"
    timematch_metrics_path="$(metrics_path_for_exp "$timematch_exp" "$target_path")"
    timematch_pra_metrics_path="$(metrics_path_for_exp "$timematch_pra_exp" "$target_path")"

    assert_file_exists "$source_metrics_path" "source-only metrics"
    assert_file_exists "$timematch_metrics_path" "baseline TimeMatch metrics"
    assert_file_exists "$timematch_pra_metrics_path" "PRA TimeMatch metrics"

    source_accuracy="$(extract_accuracy "$source_metrics_path")"
    source_macro_f1="$(extract_macro_f1 "$source_metrics_path")"
    timematch_accuracy="$(extract_accuracy "$timematch_metrics_path")"
    timematch_macro_f1="$(extract_macro_f1 "$timematch_metrics_path")"
    timematch_pra_accuracy="$(extract_accuracy "$timematch_pra_metrics_path")"
    timematch_pra_macro_f1="$(extract_macro_f1 "$timematch_pra_metrics_path")"
    timematch_pra_flag="$(extract_pra_flag "$timematch_pra_metrics_path")"
    delta_acc_pra_vs_timematch="$("$PYTHON_EXEC" -c "print(float('$timematch_pra_accuracy') - float('$timematch_accuracy'))")"
    delta_acc_pra_vs_source="$("$PYTHON_EXEC" -c "print(float('$timematch_pra_accuracy') - float('$source_accuracy'))")"
    delta_pra_vs_timematch="$("$PYTHON_EXEC" -c "print(float('$timematch_pra_macro_f1') - float('$timematch_macro_f1'))")"
    delta_pra_vs_source="$("$PYTHON_EXEC" -c "print(float('$timematch_pra_macro_f1') - float('$source_macro_f1'))")"

    csv_path="$result_dir/results.csv"
    {
        echo "source,target,seed,source_experiment,timematch_experiment,timematch_pra_experiment,source_accuracy,timematch_accuracy,timematch_pra_accuracy,pra_minus_timematch_accuracy,pra_minus_source_accuracy,source_macro_f1,timematch_macro_f1,timematch_pra_macro_f1,pra_minus_timematch_macro_f1,pra_minus_source_macro_f1,timematch_pra_enabled,pra_point_trade_off,pra_trade_off,pra_warmup_epochs,pra_min_samples_per_class,pra_pseudo_threshold,pra_bank_momentum"
        echo "$source_path,$target_path,$seed,$source_exp,$timematch_exp,$timematch_pra_exp,$source_accuracy,$timematch_accuracy,$timematch_pra_accuracy,$delta_acc_pra_vs_timematch,$delta_acc_pra_vs_source,$source_macro_f1,$timematch_macro_f1,$timematch_pra_macro_f1,$delta_pra_vs_timematch,$delta_pra_vs_source,$timematch_pra_flag,$pra_point_trade_off,$pra_trade_off,$pra_warmup_epochs,$pra_min_samples_per_class,$pra_pseudo_threshold,$pra_bank_momentum"
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
    local pra_config_tag="$4"
    local pra_point_trade_off="$5"
    local pra_trade_off="$6"
    local pra_warmup_epochs="$7"
    local pra_min_samples_per_class="$8"
    local pra_pseudo_threshold="$9"
    local pra_bank_momentum="${10}"
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
    echo "[INFO] SKIP_EXISTING_RUNS=$SKIP_EXISTING_RUNS OVERWRITE_EXISTING=$OVERWRITE_EXISTING"
    echo "[INFO] PRA naming: label=pra config_tag=$pra_config_tag exp_suffix=pra_$pra_config_tag"
    echo "[INFO] PRA config: point_trade_off=$pra_point_trade_off edge_trade_off=$pra_trade_off warmup=$pra_warmup_epochs min_samples=$pra_min_samples_per_class pseudo_threshold=$pra_pseudo_threshold bank_momentum=$pra_bank_momentum"
    echo "--------------------------------------------------"

    source_tag="$(match "$source_path")"
    target_tag="$(match "$target_path")"
    source_exp="pseltae_${source_tag}"
    timematch_exp="timematch_${source_tag}_to_${target_tag}"
    timematch_pra_exp="timematch_${source_tag}_to_${target_tag}_pra_${pra_config_tag}"

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
        --seed "$seed" \
        $( [[ "$OVERWRITE_EXISTING" == "1" ]] && printf '%s' '--overwrite_existing' )

    run_or_skip_experiment \
        "$RUN_TIMEMATCH_BASELINE" \
        "$timematch_exp" \
        "$timematch_metrics_path" \
        -e "$timematch_exp" \
        --source "$source_path" \
        --target "$target_path" \
        --seed "$seed" \
        $( [[ "$OVERWRITE_EXISTING" == "1" ]] && printf '%s' '--overwrite_existing' ) \
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
        $( [[ "$OVERWRITE_EXISTING" == "1" ]] && printf '%s' '--overwrite_existing' ) \
        timematch \
        --weights "outputs/$source_exp" \
        --use_pra \
        --pra_point_trade_off "$pra_point_trade_off" \
        --pra_trade_off "$pra_trade_off" \
        --pra_warmup_epochs "$pra_warmup_epochs" \
        --pra_min_samples_per_class "$pra_min_samples_per_class" \
        --pseudo_threshold "$pra_pseudo_threshold" \
        --pra_bank_momentum "$pra_bank_momentum"

    timestamp="$(date +"%Y%m%d_%H%M%S")_${pra_config_tag}"
    result_dir="$SCRIPT_DIR/result/$source_tag/$target_tag/$timestamp"
    mkdir -p "$result_dir"

    write_summary_csv \
        "$source_path" \
        "$target_path" \
        "$seed" \
        "$source_exp" \
        "$timematch_exp" \
        "$timematch_pra_exp" \
        "$result_dir" \
        "$pra_point_trade_off" \
        "$pra_trade_off" \
        "$pra_warmup_epochs" \
        "$pra_min_samples_per_class" \
        "$pra_pseudo_threshold" \
        "$pra_bank_momentum"
    write_classwise_csv "$target_path" "$source_exp" "$timematch_exp" "$timematch_pra_exp" "$result_dir"
}

for source in "${SOURCES[@]}"; do
    for target in "${TARGETS[@]}"; do
        if [[ "$source" == "$target" ]]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            for pra_config in "${PRA_CONFIGS[@]}"; do
                IFS='|' read -r pra_config_tag pra_point_trade_off pra_trade_off pra_warmup_epochs pra_min_samples_per_class pra_pseudo_threshold pra_bank_momentum <<< "$pra_config"
                run_experiment \
                    "$source" \
                    "$target" \
                    "$seed" \
                    "$pra_config_tag" \
                    "$pra_point_trade_off" \
                    "$pra_trade_off" \
                    "$pra_warmup_epochs" \
                    "$pra_min_samples_per_class" \
                    "$pra_pseudo_threshold" \
                    "$pra_bank_momentum"
            done
        done
    done
done

echo "[SUCCESS] Finished all experiments."
