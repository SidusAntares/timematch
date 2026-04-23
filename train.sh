#!/usr/bin/env bash
set -euo pipefail  # 启用严格模式：出错即停
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/ti/bin/python"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
# ================== 配置区（集中管理，便于修改）==================
SOURCES=("france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017")
TARGETS=("france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017" )
# "france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017"
SEEDS=(111)

# ================================================================

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

extract_macro_f1() {
    local metrics_path="$1"
    "$PYTHON_EXEC" -c "import json; print(json.load(open(r'''$metrics_path'''))['macro_f1'])"
}

write_summary_csv() {
    local source_path="$1"
    local target_path="$2"
    local seed="$3"
    local source_exp="$4"
    local timematch_exp="$5"

    local source_tag
    local target_tag
    local timestamp
    local results_root
    local result_dir
    local csv_path
    local target_file_name
    local source_metrics_path
    local timematch_metrics_path
    local source_macro_f1
    local timematch_macro_f1

    source_tag="$(match "$source_path")"
    target_tag="$(match "$target_path")"
    timestamp="$(date +"%Y%m%d_%H%M%S")"
    results_root="$SCRIPT_DIR/result"
    result_dir="$results_root/$source_tag/$target_tag/$timestamp"
    mkdir -p "$result_dir"

    target_file_name="${target_path//\//_}"
    source_metrics_path="$SCRIPT_DIR/outputs/$source_exp/fold_0/test_metrics_${target_file_name}.json"
    timematch_metrics_path="$SCRIPT_DIR/outputs/$timematch_exp/fold_0/test_metrics_${target_file_name}.json"

    source_macro_f1="$(extract_macro_f1 "$source_metrics_path")"
    timematch_macro_f1="$(extract_macro_f1 "$timematch_metrics_path")"

    csv_path="$result_dir/results.csv"
    {
        echo "source,target,timematch_macro_f1,source_f1,timestamp,source_tag,target_tag,seed,source_experiment,timematch_experiment"
        echo "$source_path,$target_path,$timematch_macro_f1,$source_macro_f1,$timestamp,$source_tag,$target_tag,$seed,$source_exp,$timematch_exp"
    } > "$csv_path"

    echo "[INFO] 实验结果已保存到: $csv_path"
}

# 执行单个训练任务的函数
run_experiment() {
    local source_path="$1"
    local target="$2"
    local seed="$3"
    local source_tag
    local target_tag
    local source_exp
    local timematch_exp

#    local seed="$2"
    echo "--------------------------------------------------"
    echo "[INFO] 开始训练: source=$source_path, target=$target, seed=$seed"
    echo "[CMD] "$PYTHON_EXEC" train.py --source '$source_path' --target '$target'"
    echo "--------------------------------------------------"

    source_tag="$(match "$source_path")"
    target_tag="$(match "$target")"
    source_exp="pseltae_${source_tag}"
    timematch_exp="timematch_${source_tag}_to_${target_tag}"

    # 执行命令，失败则退出
#    "$PYTHON_EXEC" ./train.py --source "$source_path" --target "$target" --seed "$seed"
    "$PYTHON_EXEC" ./train.py -e "$source_exp" --source "$source_path" --target "$target" --seed "$seed"
    "$PYTHON_EXEC" ./train.py -e "$timematch_exp" --source "$source_path" --target "$target" --seed "$seed" timematch --weights "outputs/$source_exp"

    write_summary_csv "$source_path" "$target" "$seed" "$source_exp" "$timematch_exp"
}

# 主循环：遍历所有组合
for source in "${SOURCES[@]}"; do
    for target in "${TARGETS[@]}"; do
        if [[ "$source" == "$target" ]]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            run_experiment "$source"  "$target" "$seed"
        done
    done
done

echo "[SUCCESS] 所有实验已完成！共 $((${#SOURCES[@]} * (${#SOURCES[@]} -1) * ${#SEEDS[@]})) 个任务。"
