#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


TASKS = [
    "FR1_to_FR2",
    "AT1_to_DK1",
    "FR2_to_AT1",
    "DK1_to_AT1",
    "FR2_to_FR1",
    "AT1_to_FR2",
]
SEEDS = {"1", "2", "3"}
CONFIGS = ["plain", "raw_global", "smooth_k3", "time_permuted_smooth_k3"]

CONFIG_ALIASES = {
    "plain": "plain",
    "raw": "raw_global",
    "raw1": "raw_global",
    "raw_global": "raw_global",
    "v275_raw_w1": "raw_global",
    "smooth": "smooth_k3",
    "smooth_k3": "smooth_k3",
    "v276_smooth_k3_w1": "smooth_k3",
    "time_permuted_smooth_k3": "time_permuted_smooth_k3",
    "v303_time_permuted_smooth_k3_w1": "time_permuted_smooth_k3",
}

CORRELATION_PREDICTORS = [
    "delta_source_on_target",
    "delta_order_gap",
    "delta_rank_perm_gap",
    "delta_transition_gap",
    "delta_dtw_gap",
    "source_entropy_improvement",
    "source_order_improvement",
    "source_transition_improvement",
    "source_dtw_improvement",
    "target_entropy_improvement",
    "target_order_improvement",
    "target_transition_improvement",
    "target_dtw_improvement",
]


def canonical_config(name):
    return CONFIG_ALIASES.get(str(name or "").strip(), str(name or "").strip())


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def read_tsv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(fmt(row.get(field)) for field in fields) + "\n")


def require_fields(path, rows, required):
    if not rows:
        raise RuntimeError(f"{path} is empty")
    available = set(rows[0].keys())
    missing = [field for field in required if field not in available]
    if missing:
        raise RuntimeError(f"{path} missing required fields: {', '.join(missing)}")


def first_existing(row, names):
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return None


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return None
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def rankdata(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_vals, y_vals = zip(*pairs)
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_vals, y_vals = zip(*pairs)
    return pearson(rankdata(list(x_vals)), rankdata(list(y_vals)))


def same_direction(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    valid = [(x, y) for x, y in pairs if x != 0.0 and y != 0.0]
    if not valid:
        return 0, None
    count = sum(1 for x, y in valid if x * y > 0.0)
    return count, count / len(valid)


def parse_da_rows(path):
    rows = read_tsv(path)
    require_fields(path, rows, ["task", "seed", "config", "source_on_target_f1", "da_f1", "da_gain"])
    parsed = {}
    for row in rows:
        task = row["task"]
        seed = str(row["seed"])
        config = canonical_config(row["config"])
        if task not in TASKS or seed not in SEEDS or config not in CONFIGS:
            continue
        parsed[(task, seed, config)] = {
            "task": task,
            "source": row.get("source", ""),
            "target": row.get("target", ""),
            "seed": seed,
            "config": config,
            "source_on_target_f1": safe_float(row.get("source_on_target_f1")),
            "da_f1": safe_float(row.get("da_f1")),
            "da_gain": safe_float(row.get("da_gain")),
            "pseudo_coverage": safe_float(row.get("pseudo_coverage")),
            "pseudo_confidence": safe_float(row.get("pseudo_confidence")),
        }
    return parsed


def parse_order_rows(path):
    rows = read_tsv(path)
    require_fields(path, rows, ["task", "seed", "config", "weighted_order_gap", "weighted_transition_gap", "weighted_dtw_gap"])
    parsed = {}
    for row in rows:
        task = row["task"]
        seed = str(row["seed"])
        config = canonical_config(row["config"])
        if task not in TASKS or seed not in SEEDS or config not in CONFIGS:
            continue
        parsed[(task, seed, config)] = {
            "order_gap": safe_float(row.get("weighted_order_gap")),
            "rank_perm_gap": safe_float(first_existing(row, ["weighted_rank_perm_gap", "weighted_rank_permutation_gap"])),
            "transition_gap": safe_float(row.get("weighted_transition_gap")),
            "dtw_gap": safe_float(row.get("weighted_dtw_gap")),
            "dtw_path_ratio": safe_float(first_existing(row, ["weighted_dtw_path_ratio", "weighted_dtw_path_length_ratio"])),
        }
    return parsed


def parse_intraclass_rows(path):
    rows = read_tsv(path)
    require_fields(
        path,
        rows,
        [
            "task",
            "seed",
            "config",
            "domain",
            "weighted_anchor_entropy",
            "weighted_pairwise_order_corr",
            "weighted_constrained_dtw",
        ],
    )
    parsed = {}
    for row in rows:
        task = row["task"]
        seed = str(row["seed"])
        config = canonical_config(row["config"])
        domain = str(row.get("domain", "")).strip().lower()
        if task not in TASKS or seed not in SEEDS or config not in CONFIGS or domain not in {"source", "target"}:
            continue
        parsed[(task, seed, config, domain)] = {
            f"{domain}_anchor_entropy": safe_float(row.get("weighted_anchor_entropy")),
            f"{domain}_pairwise_order_corr": safe_float(row.get("weighted_pairwise_order_corr")),
            f"{domain}_transition_js": safe_float(first_existing(row, ["weighted_transition_js", "weighted_transition_js_to_class_mean"])),
            f"{domain}_constrained_dtw": safe_float(row.get("weighted_constrained_dtw")),
        }
    return parsed


def build_master(da_map, order_map, intra_map):
    rows = []
    missing = []
    for task in TASKS:
        for seed in sorted(SEEDS):
            for config in CONFIGS:
                key = (task, seed, config)
                if key not in da_map or key not in order_map:
                    missing.append({"task": task, "seed": seed, "config": config, "missing": "da" if key not in da_map else "order"})
                    continue
                row = dict(da_map[key])
                row.update(order_map[key])
                for domain in ["source", "target"]:
                    intra = intra_map.get((task, seed, config, domain))
                    if intra is None:
                        missing.append({"task": task, "seed": seed, "config": config, "missing": f"{domain}_intraclass"})
                        intra = {}
                    row.update(intra)
                rows.append(row)
    return rows, missing


def build_delta_rows(master_rows):
    by_key = {(r["task"], r["seed"], r["config"]): r for r in master_rows}
    delta_rows = []
    for row in master_rows:
        if row["config"] == "plain":
            continue
        plain = by_key.get((row["task"], row["seed"], "plain"))
        if plain is None:
            continue
        out = {
            "task": row["task"],
            "source": row.get("source", ""),
            "target": row.get("target", ""),
            "seed": row["seed"],
            "config": row["config"],
        }
        metric_pairs = [
            ("source_on_target_f1", "delta_source_on_target"),
            ("da_f1", "delta_da_f1"),
            ("da_gain", "delta_da_gain"),
            ("order_gap", "delta_order_gap"),
            ("rank_perm_gap", "delta_rank_perm_gap"),
            ("transition_gap", "delta_transition_gap"),
            ("dtw_gap", "delta_dtw_gap"),
            ("source_anchor_entropy", "delta_source_anchor_entropy"),
            ("source_pairwise_order_corr", "delta_source_pairwise_order_corr"),
            ("source_transition_js", "delta_source_transition_js"),
            ("source_constrained_dtw", "delta_source_constrained_dtw"),
            ("target_anchor_entropy", "delta_target_anchor_entropy"),
            ("target_pairwise_order_corr", "delta_target_pairwise_order_corr"),
            ("target_transition_js", "delta_target_transition_js"),
            ("target_constrained_dtw", "delta_target_constrained_dtw"),
        ]
        for raw_field, delta_field in metric_pairs:
            value = row.get(raw_field)
            base = plain.get(raw_field)
            out[delta_field] = None if value is None or base is None else value - base
        out["source_entropy_improvement"] = negate(out["delta_source_anchor_entropy"])
        out["source_order_improvement"] = out["delta_source_pairwise_order_corr"]
        out["source_transition_improvement"] = negate(out["delta_source_transition_js"])
        out["source_dtw_improvement"] = negate(out["delta_source_constrained_dtw"])
        out["target_entropy_improvement"] = negate(out["delta_target_anchor_entropy"])
        out["target_order_improvement"] = out["delta_target_pairwise_order_corr"]
        out["target_transition_improvement"] = negate(out["delta_target_transition_js"])
        out["target_dtw_improvement"] = negate(out["delta_target_constrained_dtw"])
        delta_rows.append(out)
    return delta_rows


def negate(value):
    return None if value is None else -value


def summarize_correlations(delta_rows):
    rows = []
    ys = [r.get("delta_da_f1") for r in delta_rows]
    for predictor in CORRELATION_PREDICTORS:
        xs = [r.get(predictor) for r in delta_rows]
        count, rate = same_direction(xs, ys)
        valid_n = sum(1 for x, y in zip(xs, ys) if x is not None and y is not None)
        rows.append(
            {
                "predictor": predictor,
                "n": valid_n,
                "pearson_corr": pearson(xs, ys),
                "spearman_corr": spearman(xs, ys),
                "same_direction_count": count,
                "same_direction_rate": rate,
                "mean_predictor_value": mean(xs),
                "mean_delta_da_f1": mean(ys),
                "notes": "single_variable_association_not_causal",
            }
        )
    rows.sort(
        key=lambda r: (
            abs(r["spearman_corr"]) if r["spearman_corr"] is not None else -1.0,
            abs(r["pearson_corr"]) if r["pearson_corr"] is not None else -1.0,
            r["same_direction_rate"] if r["same_direction_rate"] is not None else -1.0,
        ),
        reverse=True,
    )
    return rows


def summarize_by_config(delta_rows):
    rows = []
    for config in [c for c in CONFIGS if c != "plain"]:
        subset = [r for r in delta_rows if r["config"] == config]
        row = {"config": config, "n": len(subset)}
        for field in [
            "delta_da_f1",
            "delta_source_on_target",
            "delta_order_gap",
            "delta_rank_perm_gap",
            "delta_transition_gap",
            "delta_dtw_gap",
            "source_entropy_improvement",
            "source_order_improvement",
            "source_transition_improvement",
            "source_dtw_improvement",
            "target_entropy_improvement",
            "target_order_improvement",
            "target_transition_improvement",
            "target_dtw_improvement",
        ]:
            row[f"mean_{field}"] = mean([r.get(field) for r in subset])
            row[f"std_{field}"] = std([r.get(field) for r in subset])
        ys = [r.get("delta_da_f1") for r in subset]
        corr_fields = {
            "corr_delta_da_source_on_target": "delta_source_on_target",
            "corr_delta_da_order_gap": "delta_order_gap",
            "corr_delta_da_transition_gap": "delta_transition_gap",
            "corr_delta_da_source_entropy_improvement": "source_entropy_improvement",
            "corr_delta_da_source_order_improvement": "source_order_improvement",
            "corr_delta_da_target_entropy_improvement": "target_entropy_improvement",
            "corr_delta_da_target_order_improvement": "target_order_improvement",
        }
        for out_name, field in corr_fields.items():
            row[out_name] = spearman([r.get(field) for r in subset], ys)
        rows.append(row)
    return rows


def summarize_task_config(delta_rows):
    grouped = defaultdict(list)
    for row in delta_rows:
        grouped[(row["task"], row["config"])].append(row)
    rows = []
    for (task, config), subset in sorted(grouped.items()):
        out = {"task": task, "config": config, "n": len(subset)}
        for field in [
            "delta_source_on_target",
            "delta_da_f1",
            "delta_order_gap",
            "delta_transition_gap",
            "source_entropy_improvement",
            "source_order_improvement",
            "target_entropy_improvement",
            "target_order_improvement",
        ]:
            out[f"mean_{field}"] = mean([r.get(field) for r in subset])
            out[f"std_{field}"] = std([r.get(field) for r in subset])
        rows.append(out)
    return rows


def build_hypothesis_checks(corr_rows, config_rows):
    corr = {r["predictor"]: r for r in corr_rows}
    by_config = {r["config"]: r for r in config_rows}
    rows = []
    source_corr = abs(corr.get("delta_source_on_target", {}).get("spearman_corr") or 0.0)
    order_corr = abs(corr.get("delta_order_gap", {}).get("spearman_corr") or 0.0)
    rows.append(
        {
            "hypothesis": "source_on_target_more_correlated_than_order_gap",
            "result": source_corr > order_corr,
            "supporting_values": f"abs_spearman_source_on_target={source_corr:.6f};abs_spearman_order_gap={order_corr:.6f}",
            "interpretation": "source-on-target association is stronger than cross-domain order-gap association"
            if source_corr > order_corr
            else "cross-domain order-gap association is not weaker than source-on-target association",
        }
    )

    source_fields = [
        "source_entropy_improvement",
        "source_order_improvement",
        "source_transition_improvement",
        "source_dtw_improvement",
    ]
    target_fields = [
        "target_entropy_improvement",
        "target_order_improvement",
        "target_transition_improvement",
        "target_dtw_improvement",
    ]
    target_wins = 0
    comparisons = []
    for sf, tf in zip(source_fields, target_fields):
        sc = abs(corr.get(sf, {}).get("spearman_corr") or 0.0)
        tc = abs(corr.get(tf, {}).get("spearman_corr") or 0.0)
        target_wins += int(tc > sc)
        comparisons.append(f"{tf}={tc:.6f}>{sf}={sc:.6f}:{tc > sc}")
    rows.append(
        {
            "hypothesis": "target_oracle_intraclass_more_correlated_than_source_intraclass",
            "result": target_wins > len(source_fields) / 2,
            "supporting_values": ";".join(comparisons),
            "interpretation": "oracle target intraclass metrics are more associated with DA deltas in most paired comparisons"
            if target_wins > len(source_fields) / 2
            else "source intraclass metrics are not weaker than oracle target intraclass metrics in most paired comparisons",
        }
    )

    smooth = by_config.get("smooth_k3", {})
    perm = by_config.get("time_permuted_smooth_k3", {})
    perm_not_weaker = (
        (perm.get("mean_delta_da_f1") is not None and smooth.get("mean_delta_da_f1") is not None and perm["mean_delta_da_f1"] >= smooth["mean_delta_da_f1"])
        or (
            perm.get("mean_delta_order_gap") is not None
            and smooth.get("mean_delta_order_gap") is not None
            and perm["mean_delta_order_gap"] >= smooth["mean_delta_order_gap"]
        )
    )
    rows.append(
        {
            "hypothesis": "time_permuted_not_weaker_than_smooth",
            "result": perm_not_weaker,
            "supporting_values": (
                f"perm_delta_da={perm.get('mean_delta_da_f1')};smooth_delta_da={smooth.get('mean_delta_da_f1')};"
                f"perm_delta_order={perm.get('mean_delta_order_gap')};smooth_delta_order={smooth.get('mean_delta_order_gap')}"
            ),
            "interpretation": "real temporal neighborhood is not necessary for the observed effect in this control set"
            if perm_not_weaker
            else "time-permuted control is weaker than real smooth on both DA and order gap",
        }
    )

    raw = by_config.get("raw_global", {})
    raw_confound = (
        raw.get("mean_delta_da_f1") is not None
        and smooth.get("mean_delta_da_f1") is not None
        and raw["mean_delta_da_f1"] >= smooth["mean_delta_da_f1"]
    ) or (
        raw.get("mean_delta_source_on_target") is not None
        and smooth.get("mean_delta_source_on_target") is not None
        and raw["mean_delta_source_on_target"] >= smooth["mean_delta_source_on_target"]
    )
    rows.append(
        {
            "hypothesis": "raw_global_strong_feature_quality_confound",
            "result": raw_confound,
            "supporting_values": (
                f"raw_delta_da={raw.get('mean_delta_da_f1')};smooth_delta_da={smooth.get('mean_delta_da_f1')};"
                f"raw_delta_source_on_target={raw.get('mean_delta_source_on_target')};smooth_delta_source_on_target={smooth.get('mean_delta_source_on_target')}"
            ),
            "interpretation": "raw_global suggests general feature quality/non-temporal compactness is a strong confound"
            if raw_confound
            else "raw_global is not stronger than smooth_k3 on DA or source-on-target",
        }
    )
    return rows


def write_markdown(path, input_dir, master_rows, missing_rows, corr_rows, config_rows, checks):
    top_corr = corr_rows[:8]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v3.0.4 机制重分析总结\n\n")
        handle.write("## 1. 分析目的\n\n")
        handle.write(
            "本轮不训练新模型，只分析 v3.0.3 已有结果，比较 DA F1 变化与多个候选解释变量的单变量关联强度。"
            "所有结论均为相关性诊断，不作因果证明。\n\n"
        )
        handle.write("## 2. 数据来源\n\n")
        handle.write(f"- 输入目录：`{input_dir}`\n")
        handle.write("- 输入文件：`control_da_results.tsv`、`control_order_results.tsv`、`intraclass_consistency_summary.tsv`\n")
        handle.write(f"- 有效主表记录：{len(master_rows)}\n")
        handle.write(f"- 缺失记录：{len(missing_rows)}\n\n")
        handle.write("## 3. 主相关性结果\n\n")
        handle.write("| predictor | n | Spearman | Pearson | same-direction |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in top_corr:
            handle.write(
                f"| {row['predictor']} | {row['n']} | {fmt(row['spearman_corr'])} | "
                f"{fmt(row['pearson_corr'])} | {fmt(row['same_direction_rate'])} |\n"
            )
        handle.write("\n")
        handle.write("## 4. 配置级发现\n\n")
        handle.write("| config | n | mean ΔDA | mean Δsource-on-target | mean Δorder_gap | mean Δtransition_gap |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in config_rows:
            handle.write(
                f"| {row['config']} | {row['n']} | {fmt(row.get('mean_delta_da_f1'))} | "
                f"{fmt(row.get('mean_delta_source_on_target'))} | {fmt(row.get('mean_delta_order_gap'))} | "
                f"{fmt(row.get('mean_delta_transition_gap'))} |\n"
            )
        handle.write("\n")
        handle.write("## 5. target label 指标说明\n\n")
        handle.write(
            "target 域内同类一致性使用了目标域真实标签，因此只属于 oracle 离线机制诊断，"
            "不能直接作为无监督训练信号；若后续进入方法设计，需要另做 pseudo-label 版本。\n\n"
        )
        handle.write("## 6. 假设检查\n\n")
        handle.write("| hypothesis | result | interpretation |\n")
        handle.write("|---|---:|---|\n")
        for row in checks:
            handle.write(f"| {row['hypothesis']} | {row['result']} | {row['interpretation']} |\n")
        handle.write("\n")
        handle.write("## 7. 最小结论\n\n")
        handle.write(
            "以下结论只在 v3.0.3 控制任务集合上成立。当前分析只能说明哪些变量与 DA 变化关联更强，"
            "不能证明某个变量导致 DA 提升，也不能直接推出新的训练方法或 selector。\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="logs/v303_control_and_intraclass_diagnostic_20260702_120442")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else Path("logs") / f"v304_mechanism_reanalysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    da_path = input_dir / "control_da_results.tsv"
    if not da_path.exists():
        fallback = input_dir / "control_runs" / "raw_strength_rows.tsv"
        if fallback.exists():
            da_path = fallback
        else:
            raise SystemExit(f"Missing DA input: {da_path} and fallback {fallback}")
    order_path = input_dir / "control_order_results.tsv"
    intra_path = input_dir / "intraclass_consistency_summary.tsv"
    if not order_path.exists():
        raise SystemExit(f"Missing order input: {order_path}")
    if not intra_path.exists():
        raise SystemExit(f"Missing intraclass input: {intra_path}")

    da_map = parse_da_rows(da_path)
    order_map = parse_order_rows(order_path)
    intra_map = parse_intraclass_rows(intra_path)
    master_rows, missing_rows = build_master(da_map, order_map, intra_map)
    delta_rows = build_delta_rows(master_rows)
    corr_rows = summarize_correlations(delta_rows)
    config_rows = summarize_by_config(delta_rows)
    task_rows = summarize_task_config(delta_rows)
    checks = build_hypothesis_checks(corr_rows, config_rows)

    master_fields = [
        "task",
        "source",
        "target",
        "seed",
        "config",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "order_gap",
        "rank_perm_gap",
        "transition_gap",
        "dtw_gap",
        "dtw_path_ratio",
        "source_anchor_entropy",
        "source_pairwise_order_corr",
        "source_transition_js",
        "source_constrained_dtw",
        "target_anchor_entropy",
        "target_pairwise_order_corr",
        "target_transition_js",
        "target_constrained_dtw",
        "pseudo_coverage",
        "pseudo_confidence",
    ]
    delta_fields = [
        "task",
        "source",
        "target",
        "seed",
        "config",
        "delta_source_on_target",
        "delta_da_f1",
        "delta_da_gain",
        "delta_order_gap",
        "delta_rank_perm_gap",
        "delta_transition_gap",
        "delta_dtw_gap",
        "delta_source_anchor_entropy",
        "delta_source_pairwise_order_corr",
        "delta_source_transition_js",
        "delta_source_constrained_dtw",
        "delta_target_anchor_entropy",
        "delta_target_pairwise_order_corr",
        "delta_target_transition_js",
        "delta_target_constrained_dtw",
        "source_entropy_improvement",
        "source_order_improvement",
        "source_transition_improvement",
        "source_dtw_improvement",
        "target_entropy_improvement",
        "target_order_improvement",
        "target_transition_improvement",
        "target_dtw_improvement",
    ]
    corr_fields = [
        "predictor",
        "n",
        "pearson_corr",
        "spearman_corr",
        "same_direction_count",
        "same_direction_rate",
        "mean_predictor_value",
        "mean_delta_da_f1",
        "notes",
    ]
    config_fields = ["config", "n"]
    for field in [
        "delta_da_f1",
        "delta_source_on_target",
        "delta_order_gap",
        "delta_rank_perm_gap",
        "delta_transition_gap",
        "delta_dtw_gap",
        "source_entropy_improvement",
        "source_order_improvement",
        "source_transition_improvement",
        "source_dtw_improvement",
        "target_entropy_improvement",
        "target_order_improvement",
        "target_transition_improvement",
        "target_dtw_improvement",
    ]:
        config_fields.extend([f"mean_{field}", f"std_{field}"])
    config_fields.extend(
        [
            "corr_delta_da_source_on_target",
            "corr_delta_da_order_gap",
            "corr_delta_da_transition_gap",
            "corr_delta_da_source_entropy_improvement",
            "corr_delta_da_source_order_improvement",
            "corr_delta_da_target_entropy_improvement",
            "corr_delta_da_target_order_improvement",
        ]
    )

    task_fields = ["task", "config", "n"]
    for field in [
        "delta_source_on_target",
        "delta_da_f1",
        "delta_order_gap",
        "delta_transition_gap",
        "source_entropy_improvement",
        "source_order_improvement",
        "target_entropy_improvement",
        "target_order_improvement",
    ]:
        task_fields.extend([f"mean_{field}", f"std_{field}"])

    write_tsv(output_dir / "mechanism_master_table.tsv", master_rows, master_fields)
    write_tsv(output_dir / "mechanism_delta_vs_plain.tsv", delta_rows, delta_fields)
    write_tsv(output_dir / "mechanism_correlation_summary.tsv", corr_rows, corr_fields)
    write_tsv(output_dir / "mechanism_by_config_summary.tsv", config_rows, config_fields)
    write_tsv(output_dir / "mechanism_task_summary.tsv", task_rows, task_fields)
    write_tsv(output_dir / "mechanism_hypothesis_checks.tsv", checks, ["hypothesis", "result", "supporting_values", "interpretation"])
    write_tsv(output_dir / "missing_records.tsv", missing_rows, ["task", "seed", "config", "missing"])
    write_markdown(output_dir / "v304_mechanism_reanalysis_summary.md", input_dir, master_rows, missing_rows, corr_rows, config_rows, checks)

    print(f"OUTPUT_DIR={output_dir}")
    print(f"INPUT_DA={da_path}")
    print(f"INPUT_ORDER={order_path}")
    print(f"INPUT_INTRACLASS={intra_path}")
    print(f"MASTER_ROWS={len(master_rows)}")
    print(f"DELTA_ROWS={len(delta_rows)}")
    print(f"MISSING_ROWS={len(missing_rows)}")
    print("FILES=" + ",".join(p.name for p in sorted(output_dir.iterdir())))


if __name__ == "__main__":
    main()
