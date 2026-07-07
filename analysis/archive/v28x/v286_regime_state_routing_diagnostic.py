#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


TASK_ORDER = [
    "FR1_to_FR2",
    "FR1_to_DK1",
    "FR1_to_AT1",
    "FR2_to_FR1",
    "FR2_to_DK1",
    "FR2_to_AT1",
    "DK1_to_FR1",
    "DK1_to_FR2",
    "DK1_to_AT1",
    "AT1_to_FR1",
    "AT1_to_FR2",
    "AT1_to_DK1",
]

ALL_CONFIGS = [
    "plain",
    "raw1",
    "timepoint1",
    "smooth_k3",
    "smooth_detach",
    "umsc075",
    "umsc050",
    "umsc3scale",
    "elastic_r1",
    "elastic_r2",
]

RESTRICTED_CONFIGS = ["plain", "raw1", "timepoint1", "smooth_k3", "elastic_r2"]

CONFIG_FAMILY = {
    "plain": "off_or_weak",
    "raw1": "global_compact",
    "timepoint1": "high_rigidity_shape",
    "smooth_k3": "default_smooth",
    "smooth_detach": "default_smooth_detached",
    "elastic_r2": "loose_elastic",
    "elastic_r1": "loose_elastic_ref",
    "umsc075": "multiscale",
    "umsc050": "multiscale",
    "umsc3scale": "multiscale",
}

RULES = [
    "rule0_fixed_smooth",
    "rule1_source_family_coarse",
    "rule2_source_family_conservative",
    "rule3_group_mean_approx",
    "rule4_descriptor_placeholder",
]


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def fmt_md(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def parse_task(task):
    source, target = task.split("_to_")
    return source, target


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_md_table(handle, rows, fields, headers=None, limit=None):
    headers = headers or fields
    rows = rows[:limit] if limit else rows
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("|" + "|".join("---" for _ in headers) + "|\n")
    for row in rows:
        handle.write("| " + " | ".join(fmt_md(row.get(field)) for field in fields) + " |\n")
    handle.write("\n")


def load_v285(v285_dir):
    root = Path(v285_dir)
    files = {
        "task_config_matrix": root / "task_config_matrix.tsv",
        "task_config_rank": root / "task_config_rank.tsv",
        "task_oracle_summary": root / "task_oracle_summary.tsv",
        "source_group_summary": root / "source_group_summary.tsv",
        "target_group_summary": root / "target_group_summary.tsv",
        "pair_regime_descriptor": root / "pair_regime_descriptor.tsv",
        "descriptor_outcome_correlation": root / "descriptor_outcome_correlation.tsv",
    }
    loaded = {}
    input_rows = []
    for name, path in files.items():
        if path.exists():
            rows = read_tsv(path)
            loaded[name] = rows
            input_rows.append({"file": str(path), "rows": len(rows), "status": "ok"})
        else:
            loaded[name] = []
            input_rows.append({"file": str(path), "rows": 0, "status": "missing"})
    return loaded, input_rows


def numeric_matrix(rows):
    out = []
    for row in rows:
        item = dict(row)
        for key in [
            "n",
            "source_self_mean",
            "source_on_target_mean",
            "da_f1_mean",
            "da_gain_mean",
            "da_f1_std",
            "da_gain_std",
        ]:
            item[key] = safe_float(item.get(key))
        out.append(item)
    return out


def matrix_lookup(matrix):
    return {(row["task"], row["config"]): row for row in matrix}


def get_da(matrix_by_key, task, config):
    row = matrix_by_key.get((task, config))
    return row.get("da_f1_mean") if row else None


def get_gain(matrix_by_key, task, config):
    row = matrix_by_key.get((task, config))
    return row.get("da_gain_mean") if row else None


def restricted_ranks(task, matrix_by_key):
    rows = []
    for config in RESTRICTED_CONFIGS:
        row = matrix_by_key.get((task, config))
        if row and row.get("da_f1_mean") is not None:
            rows.append((config, row["da_f1_mean"]))
    rows.sort(key=lambda item: item[1], reverse=True)
    return {config: rank for rank, (config, _) in enumerate(rows, start=1)}


def all_ranks(task, rank_rows):
    return {
        row["config"]: safe_int(row.get("rank"))
        for row in rank_rows
        if row.get("task") == task
    }


def build_state_candidate_matrix(matrix, rank_rows):
    matrix_by_key = matrix_lookup(matrix)
    out = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        plain = get_da(matrix_by_key, task, "plain")
        smooth = get_da(matrix_by_key, task, "smooth_k3")
        ranks_all = all_ranks(task, rank_rows)
        ranks_restricted = restricted_ranks(task, matrix_by_key)
        for config in ALL_CONFIGS:
            row = matrix_by_key.get((task, config), {})
            da = row.get("da_f1_mean")
            out.append(
                {
                    "task": task,
                    "source": source,
                    "target": target,
                    "config": config,
                    "family": CONFIG_FAMILY.get(config),
                    "da_f1_mean": da,
                    "da_gain_mean": row.get("da_gain_mean"),
                    "source_on_target_mean": row.get("source_on_target_mean"),
                    "delta_vs_plain": delta(da, plain),
                    "delta_vs_smooth": delta(da, smooth),
                    "rank_all_configs": ranks_all.get(config),
                    "rank_restricted_configs": ranks_restricted.get(config),
                }
            )
    return out


def build_restricted_oracle(matrix_by_key):
    out = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        candidates = []
        for config in RESTRICTED_CONFIGS:
            da = get_da(matrix_by_key, task, config)
            if da is not None:
                candidates.append((config, da))
        candidates.sort(key=lambda item: item[1], reverse=True)
        best = candidates[0] if candidates else (None, None)
        second = candidates[1] if len(candidates) > 1 else (None, None)
        ranks = {config: rank for rank, (config, _) in enumerate(candidates, start=1)}
        plain_da = get_da(matrix_by_key, task, "plain")
        smooth_da = get_da(matrix_by_key, task, "smooth_k3")
        out.append(
            {
                "task": task,
                "source": source,
                "target": target,
                "plain_da": plain_da,
                "raw1_da": get_da(matrix_by_key, task, "raw1"),
                "timepoint1_da": get_da(matrix_by_key, task, "timepoint1"),
                "smooth_k3_da": smooth_da,
                "elastic_r2_da": get_da(matrix_by_key, task, "elastic_r2"),
                "best_restricted_config": best[0],
                "best_restricted_da": best[1],
                "second_restricted_config": second[0],
                "second_restricted_da": second[1],
                "restricted_oracle_gain_vs_plain": delta(best[1], plain_da),
                "restricted_oracle_gain_vs_smooth": delta(best[1], smooth_da),
                "best_family": CONFIG_FAMILY.get(best[0]),
                "smooth_rank_restricted": ranks.get("smooth_k3"),
                "raw_rank_restricted": ranks.get("raw1"),
                "timepoint_rank_restricted": ranks.get("timepoint1"),
                "elastic_r2_rank_restricted": ranks.get("elastic_r2"),
            }
        )
    return out


def group_best_config(matrix, group_field, group_value, configs):
    rows = [
        row for row in matrix
        if row[group_field] == group_value and row["config"] in configs and row.get("da_f1_mean") is not None
    ]
    means = {}
    for config in configs:
        means[config] = mean([row["da_f1_mean"] for row in rows if row["config"] == config])
    valid = {config: value for config, value in means.items() if value is not None}
    if not valid:
        return None, None
    best = max(valid, key=valid.get)
    return best, valid[best]


def task_eval_row(task, selected_config, matrix_by_key, oracle_by_task, prefix):
    source, target = parse_task(task)
    selected_da = get_da(matrix_by_key, task, selected_config) if selected_config else None
    smooth_da = get_da(matrix_by_key, task, "smooth_k3")
    oracle_da = oracle_by_task[task]["best_restricted_da"]
    return {
        "row_type": "task_result",
        "task": task,
        "source": source,
        "target": target,
        prefix: selected_config,
        "selected_config": selected_config,
        "selected_family": CONFIG_FAMILY.get(selected_config),
        "selected_da": selected_da,
        "smooth_da": smooth_da,
        "delta_vs_smooth": delta(selected_da, smooth_da),
        "oracle_restricted_da": oracle_da,
        "gap_to_restricted_oracle": delta(oracle_da, selected_da),
    }


def summarize_task_results(rows, rule_name=None):
    task_rows = [row for row in rows if row.get("row_type") == "task_result"]
    avg_da = mean([row.get("selected_da") for row in task_rows])
    smooth_avg = mean([row.get("smooth_da") for row in task_rows])
    oracle_avg = mean([row.get("oracle_restricted_da") for row in task_rows])
    item = {
        "row_type": "summary",
        "rule_name": rule_name,
        "avg_da": avg_da,
        "smooth_avg_da": smooth_avg,
        "gain_vs_smooth": delta(avg_da, smooth_avg),
        "gap_to_restricted_oracle": delta(oracle_avg, avg_da),
        "positive_tasks_vs_smooth": sum(
            1 for row in task_rows
            if row.get("delta_vs_smooth") is not None and row["delta_vs_smooth"] > 0
        ),
        "negative_tasks_vs_smooth": sum(
            1 for row in task_rows
            if row.get("delta_vs_smooth") is not None and row["delta_vs_smooth"] < 0
        ),
        "selected_config_counts": ";".join(
            f"{config}:{count}" for config, count in sorted(
                Counter(row.get("selected_config") for row in task_rows).items()
            )
        ),
    }
    return item


def build_source_router(matrix, matrix_by_key, restricted_oracle):
    oracle_by_task = {row["task"]: row for row in restricted_oracle}
    out = []
    for source in sorted({row["source"] for row in matrix}):
        selected, selected_mean = group_best_config(matrix, "source", source, RESTRICTED_CONFIGS)
        source_tasks = [task for task in TASK_ORDER if parse_task(task)[0] == source]
        smooth_mean = mean([get_da(matrix_by_key, task, "smooth_k3") for task in source_tasks])
        out.append(
            {
                "row_type": "source_policy",
                "source": source,
                "selected_config": selected,
                "selected_family": CONFIG_FAMILY.get(selected),
                "n_tasks": len(source_tasks),
                "selected_mean_da": selected_mean,
                "smooth_mean_da": smooth_mean,
                "gain_vs_smooth": delta(selected_mean, smooth_mean),
                "tasks_covered": ",".join(source_tasks),
            }
        )
        for task in source_tasks:
            out.append(task_eval_row(task, selected, matrix_by_key, oracle_by_task, "selected_config_by_source"))
    out.append(summarize_task_results(out))
    return out


def build_target_router(matrix, matrix_by_key, restricted_oracle):
    oracle_by_task = {row["task"]: row for row in restricted_oracle}
    out = []
    for target in sorted({row["target"] for row in matrix}):
        selected, selected_mean = group_best_config(matrix, "target", target, RESTRICTED_CONFIGS)
        target_tasks = [task for task in TASK_ORDER if parse_task(task)[1] == target]
        smooth_mean = mean([get_da(matrix_by_key, task, "smooth_k3") for task in target_tasks])
        out.append(
            {
                "row_type": "target_policy",
                "target": target,
                "selected_config": selected,
                "selected_family": CONFIG_FAMILY.get(selected),
                "n_tasks": len(target_tasks),
                "selected_mean_da": selected_mean,
                "smooth_mean_da": smooth_mean,
                "gain_vs_smooth": delta(selected_mean, smooth_mean),
                "tasks_covered": ",".join(target_tasks),
            }
        )
        for task in target_tasks:
            out.append(task_eval_row(task, selected, matrix_by_key, oracle_by_task, "selected_config_by_target"))
    out.append(summarize_task_results(out))
    return out


def source_rule1(source):
    return {
        "AT1": "elastic_r2",
        "FR1": "raw1",
        "FR2": "smooth_k3",
        "DK1": "smooth_k3",
    }.get(source)


def source_rule2(source, dk_choice):
    return {
        "AT1": "elastic_r2",
        "FR1": "raw1",
        "FR2": "smooth_k3",
        "DK1": dk_choice,
    }.get(source)


def group_mean_approx(task, matrix, matrix_by_key):
    source, target = parse_task(task)
    scores = {}
    for config in RESTRICTED_CONFIGS:
        source_mean = mean([
            row["da_f1_mean"] for row in matrix
            if row["source"] == source and row["config"] == config
        ])
        target_mean = mean([
            row["da_f1_mean"] for row in matrix
            if row["target"] == target and row["config"] == config
        ])
        scores[config] = mean([source_mean, target_mean])
    valid = {config: value for config, value in scores.items() if value is not None}
    return max(valid, key=valid.get) if valid else None


def descriptor_placeholder_config(pair_row):
    required = [
        "pair_shift_gain",
        "source_temporal_shape_unit_fisher",
        "pair_mean_curve_distance",
        "doy_overlap_ratio",
    ]
    if any(pair_row.get(key) in (None, "") for key in required):
        return None
    return None


def build_rule_router(matrix, matrix_by_key, restricted_oracle, pair_rows):
    oracle_by_task = {row["task"]: row for row in restricted_oracle}
    pair_by_task = {row["task"]: row for row in pair_rows}
    dk_raw = mean([
        row["da_f1_mean"] for row in matrix
        if row["source"] == "DK1" and row["config"] == "raw1"
    ])
    dk_smooth = mean([
        row["da_f1_mean"] for row in matrix
        if row["source"] == "DK1" and row["config"] == "smooth_k3"
    ])
    dk_choice = "raw1" if (dk_raw or -1) > (dk_smooth or -1) else "smooth_k3"
    out = []
    for rule_name in RULES:
        rule_rows = []
        for task in TASK_ORDER:
            source, target = parse_task(task)
            selected = None
            skipped = ""
            if rule_name == "rule0_fixed_smooth":
                selected = "smooth_k3"
            elif rule_name == "rule1_source_family_coarse":
                selected = source_rule1(source)
            elif rule_name == "rule2_source_family_conservative":
                selected = source_rule2(source, dk_choice)
            elif rule_name == "rule3_group_mean_approx":
                selected = group_mean_approx(task, matrix, matrix_by_key)
            elif rule_name == "rule4_descriptor_placeholder":
                selected = descriptor_placeholder_config(pair_by_task.get(task, {}))
                if selected is None:
                    skipped = "descriptor_missing"
            row = task_eval_row(task, selected, matrix_by_key, oracle_by_task, "selected_config")
            row.update(
                {
                    "rule_name": rule_name,
                    "skipped_reason": skipped,
                }
            )
            rule_rows.append(row)
        out.extend(rule_rows)
        out.append(summarize_task_results(rule_rows, rule_name=rule_name))
    return out


def descriptor_vector(row):
    fields = [
        "same_country_flag",
        "france_pair_flag",
        "source_AT",
        "source_DK",
        "source_FR",
        "target_AT",
        "target_DK",
        "target_FR",
    ]
    vals = []
    for field in fields:
        value = safe_float(row.get(field))
        vals.append(0.0 if value is None else value)
    return vals


def nearest_config(task, train_tasks, pair_by_task, restricted_oracle):
    if task not in pair_by_task:
        return None, "descriptor_missing"
    vec = descriptor_vector(pair_by_task[task])
    best_task = None
    best_dist = None
    for candidate in train_tasks:
        if candidate not in pair_by_task:
            continue
        other = descriptor_vector(pair_by_task[candidate])
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, other)))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_task = candidate
    if best_task is None:
        return None, "descriptor_missing"
    return restricted_oracle[best_task]["best_restricted_config"], f"nearest={best_task}"


def build_leave_one_eval(matrix, matrix_by_key, restricted_oracle_rows, pair_rows, group_field):
    oracle_by_task = {row["task"]: row for row in restricted_oracle_rows}
    pair_by_task = {row["task"]: row for row in pair_rows}
    groups = sorted({parse_task(task)[0 if group_field == "source" else 1] for task in TASK_ORDER})
    out = []
    for heldout in groups:
        eval_tasks = [
            task for task in TASK_ORDER
            if parse_task(task)[0 if group_field == "source" else 1] == heldout
        ]
        train_tasks = [task for task in TASK_ORDER if task not in eval_tasks]
        train_rows = [
            row for row in matrix
            if row["task"] in train_tasks and row["config"] in RESTRICTED_CONFIGS
        ]
        means = {
            config: mean([row["da_f1_mean"] for row in train_rows if row["config"] == config])
            for config in RESTRICTED_CONFIGS
        }
        valid = {config: value for config, value in means.items() if value is not None}
        global_best = max(valid, key=valid.get) if valid else None
        for method in ["global_best_from_training", "descriptor_nearest"]:
            task_details = []
            selected_das = []
            smooth_das = []
            oracle_das = []
            for task in eval_tasks:
                if method == "global_best_from_training":
                    selected = global_best
                    policy = f"global_best={global_best}"
                else:
                    selected, policy = nearest_config(task, train_tasks, pair_by_task, oracle_by_task)
                selected_da = get_da(matrix_by_key, task, selected) if selected else None
                smooth_da = get_da(matrix_by_key, task, "smooth_k3")
                oracle_da = oracle_by_task[task]["best_restricted_da"]
                selected_das.append(selected_da)
                smooth_das.append(smooth_da)
                oracle_das.append(oracle_da)
                task_details.append(
                    f"{task}:{selected}:{fmt(selected_da)}:{fmt(delta(selected_da, smooth_da))}"
                )
            avg_da = mean(selected_das)
            smooth_avg = mean(smooth_das)
            oracle_avg = mean(oracle_das)
            out.append(
                {
                    f"heldout_{group_field}": heldout,
                    "method": method,
                    "selected_policy": policy,
                    "n_eval_tasks": len(eval_tasks),
                    "avg_da": avg_da,
                    "smooth_avg_da": smooth_avg,
                    "gain_vs_smooth": delta(avg_da, smooth_avg),
                    "gap_to_restricted_oracle": delta(oracle_avg, avg_da),
                    "task_level_details": ";".join(task_details),
                }
            )
    return out


def classify_error(selected_config, selected_da, best_config, best_da, smooth_da, plain_da, matrix_by_key, task):
    gap = delta(best_da, selected_da)
    err_smooth = delta(selected_da, smooth_da)
    if selected_config == "elastic_r2" and err_smooth is not None and err_smooth < 0:
        return "elastic_overuse"
    if selected_config == "timepoint1" and err_smooth is not None and err_smooth < 0:
        return "rigidity_overuse"
    if selected_config == "plain" and best_da is not None and plain_da is not None and best_da - plain_da > 0.02:
        return "under_structured"
    if selected_config == "raw1":
        smooth = get_da(matrix_by_key, task, "smooth_k3")
        timepoint = get_da(matrix_by_key, task, "timepoint1")
        better_shape = max([v for v in [smooth, timepoint] if v is not None], default=None)
        raw = get_da(matrix_by_key, task, "raw1")
        if better_shape is not None and raw is not None and better_shape - raw > 0.01:
            return "too_global"
    if err_smooth is not None and err_smooth >= 0 and gap is not None and gap <= 0.005:
        return "near_oracle_good"
    if err_smooth is not None and err_smooth >= 0 and gap is not None and gap > 0.01:
        return "improves_but_not_best"
    if err_smooth is not None and err_smooth < 0:
        return "hurts_vs_smooth"
    return "other"


def collect_routing_task_rows(source_router, target_router, rule_router):
    rows = []
    for row in source_router:
        if row.get("row_type") == "task_result":
            item = dict(row)
            item["rule_name"] = "source_router"
            rows.append(item)
    for row in target_router:
        if row.get("row_type") == "task_result":
            item = dict(row)
            item["rule_name"] = "target_router"
            rows.append(item)
    for row in rule_router:
        if row.get("row_type") == "task_result":
            rows.append(row)
    return rows


def build_error_analysis(routing_rows, restricted_oracle, matrix_by_key):
    oracle_by_task = {row["task"]: row for row in restricted_oracle}
    out = []
    for row in routing_rows:
        task = row["task"]
        oracle = oracle_by_task[task]
        selected_config = row.get("selected_config")
        selected_da = row.get("selected_da")
        smooth_da = row.get("smooth_da")
        best_config = oracle.get("best_restricted_config")
        best_da = oracle.get("best_restricted_da")
        plain_da = oracle.get("plain_da")
        out.append(
            {
                "rule_name": row.get("rule_name"),
                "task": task,
                "source": row.get("source"),
                "target": row.get("target"),
                "selected_config": selected_config,
                "selected_da": selected_da,
                "best_restricted_config": best_config,
                "best_restricted_da": best_da,
                "smooth_da": smooth_da,
                "error_vs_best": delta(selected_da, best_da),
                "error_vs_smooth": delta(selected_da, smooth_da),
                "error_type": classify_error(
                    selected_config,
                    selected_da,
                    best_config,
                    best_da,
                    smooth_da,
                    plain_da,
                    matrix_by_key,
                    task,
                ),
            }
        )
    return out


def write_summary(path, input_rows, restricted_oracle, source_router, target_router, rule_router, los_rows, lot_rows, error_rows):
    def summary_row(rows):
        return next((row for row in rows if row.get("row_type") == "summary"), {})

    fixed = {
        config: mean([row.get(f"{config}_da") for row in restricted_oracle])
        for config in ["plain", "raw1", "timepoint1", "smooth_k3", "elastic_r2"]
    }
    restricted_avg = mean([row.get("best_restricted_da") for row in restricted_oracle])
    smooth_avg = fixed.get("smooth_k3")
    rule_summary = [row for row in rule_router if row.get("row_type") == "summary"]
    error_counts = Counter(row["error_type"] for row in error_rows)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v286 Regime-State Routing Diagnostic\n\n")
        handle.write("## 1. Input\n\n")
        write_md_table(handle, input_rows, ["file", "rows", "status"])

        handle.write("## 2. Restricted Candidate Set\n\n")
        handle.write("Restricted configs: `plain`, `raw1`, `timepoint1`, `smooth_k3`, `elastic_r2`.\n\n")
        handle.write("UMSC configs are reference-only and are not used as first-version routing actions.\n\n")

        handle.write("## 3. Restricted Oracle\n\n")
        write_md_table(
            handle,
            [
                {
                    "fixed_plain_avg": fixed.get("plain"),
                    "fixed_raw1_avg": fixed.get("raw1"),
                    "fixed_timepoint1_avg": fixed.get("timepoint1"),
                    "fixed_smooth_k3_avg": smooth_avg,
                    "fixed_elastic_r2_avg": fixed.get("elastic_r2"),
                    "restricted_oracle_avg": restricted_avg,
                    "restricted_oracle_gain_vs_smooth": delta(restricted_avg, smooth_avg),
                }
            ],
            [
                "fixed_plain_avg",
                "fixed_raw1_avg",
                "fixed_timepoint1_avg",
                "fixed_smooth_k3_avg",
                "fixed_elastic_r2_avg",
                "restricted_oracle_avg",
                "restricted_oracle_gain_vs_smooth",
            ],
        )

        handle.write("## 4. Source Router Simulation\n\n")
        write_md_table(handle, [summary_row(source_router)], ["avg_da", "smooth_avg_da", "gain_vs_smooth", "gap_to_restricted_oracle", "selected_config_counts"])

        handle.write("## 5. Target Router Simulation\n\n")
        write_md_table(handle, [summary_row(target_router)], ["avg_da", "smooth_avg_da", "gain_vs_smooth", "gap_to_restricted_oracle", "selected_config_counts"])

        handle.write("## 6. Descriptor Rule Router Simulation\n\n")
        write_md_table(handle, rule_summary, ["rule_name", "avg_da", "gain_vs_smooth", "gap_to_restricted_oracle", "positive_tasks_vs_smooth", "negative_tasks_vs_smooth", "selected_config_counts"])

        handle.write("## 7. Leave-one-source Evaluation\n\n")
        write_md_table(handle, los_rows, ["heldout_source", "method", "selected_policy", "n_eval_tasks", "avg_da", "smooth_avg_da", "gain_vs_smooth", "gap_to_restricted_oracle"])

        handle.write("## 8. Leave-one-target Evaluation\n\n")
        write_md_table(handle, lot_rows, ["heldout_target", "method", "selected_policy", "n_eval_tasks", "avg_da", "smooth_avg_da", "gain_vs_smooth", "gap_to_restricted_oracle"])

        handle.write("## 9. Error Analysis\n\n")
        write_md_table(handle, [{"error_type": key, "n": value} for key, value in sorted(error_counts.items())], ["error_type", "n"])

        handle.write("## 10. Minimal Observations\n\n")
        handle.write(f"- restricted_oracle_gain_vs_smooth = {fmt_md(delta(restricted_avg, smooth_avg))}\n")
        handle.write(f"- source_router_gain_vs_smooth = {fmt_md(summary_row(source_router).get('gain_vs_smooth'))}\n")
        handle.write(f"- target_router_gain_vs_smooth = {fmt_md(summary_row(target_router).get('gain_vs_smooth'))}\n")
        handle.write("- descriptor placeholder rule is skipped when required descriptors are NA.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v285_dir", default="logs/v285_regime_diagnostic_20260627_201553")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path("logs") / f"v286_regime_state_routing_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    v285 = Path(args.v285_dir)
    input_files = {
        "task_config_matrix": v285 / "task_config_matrix.tsv",
        "task_config_rank": v285 / "task_config_rank.tsv",
        "pair_regime_descriptor": v285 / "pair_regime_descriptor.tsv",
    }
    input_rows = []
    for name, path in input_files.items():
        input_rows.append({"file": str(path), "rows": len(read_tsv(path)) if path.exists() else 0, "status": "ok" if path.exists() else "missing"})

    matrix = numeric_matrix(read_tsv(input_files["task_config_matrix"]))
    rank_rows = read_tsv(input_files["task_config_rank"])
    pair_rows = read_tsv(input_files["pair_regime_descriptor"])
    for row in pair_rows:
        for key, value in list(row.items()):
            number = safe_float(value)
            if number is not None:
                row[key] = number

    matrix_by_key = matrix_lookup(matrix)
    state_candidates = build_state_candidate_matrix(matrix, rank_rows)
    restricted_oracle = build_restricted_oracle(matrix_by_key)
    source_router = build_source_router(matrix, matrix_by_key, restricted_oracle)
    target_router = build_target_router(matrix, matrix_by_key, restricted_oracle)
    rule_router = build_rule_router(matrix, matrix_by_key, restricted_oracle, pair_rows)
    los_rows = build_leave_one_eval(matrix, matrix_by_key, restricted_oracle, pair_rows, "source")
    lot_rows = build_leave_one_eval(matrix, matrix_by_key, restricted_oracle, pair_rows, "target")
    error_rows = build_error_analysis(
        collect_routing_task_rows(source_router, target_router, rule_router),
        restricted_oracle,
        matrix_by_key,
    )

    write_tsv(
        out_dir / "state_candidate_matrix.tsv",
        state_candidates,
        [
            "task",
            "source",
            "target",
            "config",
            "family",
            "da_f1_mean",
            "da_gain_mean",
            "source_on_target_mean",
            "delta_vs_plain",
            "delta_vs_smooth",
            "rank_all_configs",
            "rank_restricted_configs",
        ],
    )
    write_tsv(
        out_dir / "restricted_oracle_summary.tsv",
        restricted_oracle,
        [
            "task",
            "source",
            "target",
            "plain_da",
            "raw1_da",
            "timepoint1_da",
            "smooth_k3_da",
            "elastic_r2_da",
            "best_restricted_config",
            "best_restricted_da",
            "second_restricted_config",
            "second_restricted_da",
            "restricted_oracle_gain_vs_plain",
            "restricted_oracle_gain_vs_smooth",
            "best_family",
            "smooth_rank_restricted",
            "raw_rank_restricted",
            "timepoint_rank_restricted",
            "elastic_r2_rank_restricted",
        ],
    )
    router_fields = [
        "row_type",
        "source",
        "target",
        "task",
        "selected_config",
        "selected_family",
        "n_tasks",
        "selected_mean_da",
        "smooth_mean_da",
        "gain_vs_smooth",
        "tasks_covered",
        "selected_config_by_source",
        "selected_config_by_target",
        "selected_da",
        "smooth_da",
        "delta_vs_smooth",
        "oracle_restricted_da",
        "gap_to_restricted_oracle",
        "avg_da",
        "smooth_avg_da",
        "selected_config_counts",
    ]
    write_tsv(out_dir / "source_state_router_sim.tsv", source_router, router_fields)
    write_tsv(out_dir / "target_state_router_sim.tsv", target_router, router_fields)
    rule_fields = [
        "row_type",
        "rule_name",
        "task",
        "source",
        "target",
        "selected_config",
        "selected_family",
        "selected_da",
        "smooth_da",
        "delta_vs_smooth",
        "oracle_restricted_da",
        "gap_to_restricted_oracle",
        "skipped_reason",
        "avg_da",
        "gain_vs_smooth",
        "positive_tasks_vs_smooth",
        "negative_tasks_vs_smooth",
        "selected_config_counts",
    ]
    write_tsv(out_dir / "descriptor_rule_router_sim.tsv", rule_router, rule_fields)
    write_tsv(
        out_dir / "leave_one_source_eval.tsv",
        los_rows,
        ["heldout_source", "method", "selected_policy", "n_eval_tasks", "avg_da", "smooth_avg_da", "gain_vs_smooth", "gap_to_restricted_oracle", "task_level_details"],
    )
    write_tsv(
        out_dir / "leave_one_target_eval.tsv",
        lot_rows,
        ["heldout_target", "method", "selected_policy", "n_eval_tasks", "avg_da", "smooth_avg_da", "gain_vs_smooth", "gap_to_restricted_oracle", "task_level_details"],
    )
    write_tsv(
        out_dir / "routing_error_analysis.tsv",
        error_rows,
        [
            "rule_name",
            "task",
            "source",
            "target",
            "selected_config",
            "selected_da",
            "best_restricted_config",
            "best_restricted_da",
            "smooth_da",
            "error_vs_best",
            "error_vs_smooth",
            "error_type",
        ],
    )
    write_summary(
        out_dir / "regime_state_routing_summary.md",
        input_rows,
        restricted_oracle,
        source_router,
        target_router,
        rule_router,
        los_rows,
        lot_rows,
        error_rows,
    )

    for name in [
        "state_candidate_matrix.tsv",
        "restricted_oracle_summary.tsv",
        "source_state_router_sim.tsv",
        "target_state_router_sim.tsv",
        "descriptor_rule_router_sim.tsv",
        "leave_one_source_eval.tsv",
        "leave_one_target_eval.tsv",
        "routing_error_analysis.tsv",
        "regime_state_routing_summary.md",
    ]:
        print("Wrote:", out_dir / name)


if __name__ == "__main__":
    main()
