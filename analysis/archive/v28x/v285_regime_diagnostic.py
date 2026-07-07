#!/usr/bin/env python3
import argparse
import csv
import math
import statistics as stats
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

CONFIG_ORDER = [
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

CONFIG_ALIAS = {
    "plain": "plain",
    "v275_raw_w1": "raw1",
    "v276_timepoint_w1": "timepoint1",
    "v276_smooth_k3_w1": "smooth_k3",
    "v276_smoothed_timepoint_w1": "smooth_k3",
    "v276_smooth_k3_w1_detach": "smooth_detach",
    "v283a_umsc_075l3_025linf_w1": "umsc075",
    "v283a_umsc_050l3_050linf_w1": "umsc050",
    "v283b_umsc_060l3_020l5_020linf_w1": "umsc3scale",
    "v284_elastic_k3_r0_w1": "smooth_k3",
    "v284_elastic_k3_r1_w1": "elastic_r1",
    "v284_elastic_k3_r2_w1": "elastic_r2",
}

CONFIG_PRIORITY = {
    "plain": 10,
    "raw1": 10,
    "timepoint1": 10,
    "smooth_k3": 20,  # prefer v284 r0 over older smooth rows when both exist
    "smooth_detach": 10,
    "umsc075": 10,
    "umsc050": 10,
    "umsc3scale": 10,
    "elastic_r1": 10,
    "elastic_r2": 10,
}

SOURCE_PRIORITY = {
    "v284_elastic_full12": 20,
    "v281_a_group_full": 10,
    "v275_closedset_baseline_v275": 10,
    "v283_umsc_full": 10,
}

FAMILY = {
    "AT1": "AT",
    "DK1": "DK",
    "FR1": "FR",
    "FR2": "FR",
}

FAMILY_BY_CONFIG = {
    "plain": "off_or_weak",
    "raw1": "global_compact",
    "timepoint1": "high_rigidity_shape",
    "smooth_k3": "default_smooth",
    "smooth_detach": "default_smooth_detached",
    "umsc075": "multiscale",
    "umsc050": "multiscale",
    "umsc3scale": "multiscale",
    "elastic_r1": "loose_elastic",
    "elastic_r2": "loose_elastic",
}

NA = None


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


def stdev(values):
    values = [value for value in values if value is not None]
    return None if len(values) < 2 else stats.stdev(values)


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    var_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def parse_task(task):
    source, target = task.split("_to_")
    return source, target


def source_name_from_path(path):
    text = str(path)
    for marker in SOURCE_PRIORITY:
        if marker in text:
            return marker
    return Path(path).parent.name


def source_priority(path, canonical_config):
    source_name = source_name_from_path(path)
    priority = SOURCE_PRIORITY.get(source_name, 0)
    if canonical_config == "smooth_k3" and "v284_elastic_full12" in str(path):
        priority += 100
    return priority


def load_rows(paths):
    loaded = []
    input_counts = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "raw_strength_rows.tsv"
        if not path.exists():
            input_counts.append({"path": str(path), "rows": 0, "status": "missing"})
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        input_counts.append({"path": str(path), "rows": len(rows), "status": "ok"})
        for row in rows:
            if row.get("status") != "ok":
                continue
            canonical = CONFIG_ALIAS.get(row.get("config", ""))
            if canonical is None:
                continue
            task = row.get("task")
            if task not in TASK_ORDER:
                continue
            source, target = parse_task(task)
            item = {
                "task": task,
                "source": source,
                "target": target,
                "seed": safe_int(row.get("seed")),
                "config": canonical,
                "raw_config": row.get("config"),
                "input_path": str(path),
                "input_source": source_name_from_path(path),
                "source_self_f1": safe_float(row.get("source_self_f1")),
                "source_on_target_f1": safe_float(row.get("source_on_target_f1")),
                "da_f1": safe_float(row.get("da_f1")),
                "da_gain": safe_float(row.get("da_gain")),
                "elastic_mean_abs_offset": safe_float(row.get("elastic_mean_abs_offset")),
                "elastic_center_weight": safe_float(row.get("elastic_center_weight")),
                "elastic_boundary_weight": safe_float(row.get("elastic_boundary_weight")),
                "elastic_struct_loss": safe_float(row.get("elastic_struct_loss")),
            }
            loaded.append(item)
    return loaded, input_counts


def dedupe_rows(rows):
    best = {}
    for row in rows:
        key = (row["task"], row["seed"], row["config"])
        score = (
            source_priority(row["input_path"], row["config"]),
            CONFIG_PRIORITY.get(row["config"], 0),
        )
        if key not in best or score > best[key][0]:
            best[key] = (score, row)
    return [item[1] for item in best.values()]


def aggregate_task_config(rows):
    by_key = {}
    for row in rows:
        by_key.setdefault((row["task"], row["config"]), []).append(row)
    matrix = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        for config in CONFIG_ORDER:
            group = by_key.get((task, config), [])
            matrix.append(
                {
                    "task": task,
                    "source": source,
                    "target": target,
                    "config": config,
                    "n": len(group),
                    "source_self_mean": mean([row["source_self_f1"] for row in group]),
                    "source_on_target_mean": mean([row["source_on_target_f1"] for row in group]),
                    "da_f1_mean": mean([row["da_f1"] for row in group]),
                    "da_gain_mean": mean([row["da_gain"] for row in group]),
                    "da_f1_std": stdev([row["da_f1"] for row in group]),
                    "da_gain_std": stdev([row["da_gain"] for row in group]),
                    "elastic_offset_mean": mean([row["elastic_mean_abs_offset"] for row in group]),
                    "elastic_center_weight_mean": mean([row["elastic_center_weight"] for row in group]),
                    "elastic_boundary_weight_mean": mean([row["elastic_boundary_weight"] for row in group]),
                    "elastic_loss_mean": mean([row["elastic_struct_loss"] for row in group]),
                }
            )
    return matrix


def value_lookup(matrix, task, config, field="da_f1_mean"):
    for row in matrix:
        if row["task"] == task and row["config"] == config:
            return row.get(field)
    return None


def build_rank(matrix):
    out = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        rows = [
            row for row in matrix
            if row["task"] == task and row.get("da_f1_mean") is not None
        ]
        rows.sort(key=lambda row: row["da_f1_mean"], reverse=True)
        plain_da = value_lookup(matrix, task, "plain")
        smooth_da = value_lookup(matrix, task, "smooth_k3")
        for idx, row in enumerate(rows, start=1):
            out.append(
                {
                    "task": task,
                    "source": source,
                    "target": target,
                    "rank": idx,
                    "config": row["config"],
                    "da_f1_mean": row["da_f1_mean"],
                    "da_gain_mean": row["da_gain_mean"],
                    "source_on_target_mean": row["source_on_target_mean"],
                    "delta_vs_plain": delta(row["da_f1_mean"], plain_da),
                    "delta_vs_smooth_k3": delta(row["da_f1_mean"], smooth_da),
                }
            )
    return out


def config_rank(rank_rows, task, config):
    for row in rank_rows:
        if row["task"] == task and row["config"] == config:
            return row["rank"]
    return None


def build_oracle(matrix, rank_rows):
    out = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        task_rows = [
            row for row in matrix
            if row["task"] == task and row.get("da_f1_mean") is not None
        ]
        task_rows.sort(key=lambda row: row["da_f1_mean"], reverse=True)
        best = task_rows[0] if task_rows else {}
        second = task_rows[1] if len(task_rows) > 1 else {}
        plain_da = value_lookup(matrix, task, "plain")
        smooth_da = value_lookup(matrix, task, "smooth_k3")
        raw_da = value_lookup(matrix, task, "raw1")
        timepoint_da = value_lookup(matrix, task, "timepoint1")
        elastic_r1_da = value_lookup(matrix, task, "elastic_r1")
        elastic_r2_da = value_lookup(matrix, task, "elastic_r2")
        elastic_best_da = mean([value for value in [elastic_r1_da, elastic_r2_da] if value is not None])
        max_elastic = max(
            [value for value in [elastic_r1_da, elastic_r2_da] if value is not None],
            default=None,
        )
        best_config = best.get("config")
        out.append(
            {
                "task": task,
                "source": source,
                "target": target,
                "plain_da": plain_da,
                "raw1_da": raw_da,
                "timepoint1_da": timepoint_da,
                "smooth_k3_da": smooth_da,
                "umsc075_da": value_lookup(matrix, task, "umsc075"),
                "elastic_r1_da": elastic_r1_da,
                "elastic_r2_da": elastic_r2_da,
                "elastic_mean_da": elastic_best_da,
                "best_config": best_config,
                "best_da": best.get("da_f1_mean"),
                "second_best_config": second.get("config"),
                "second_best_da": second.get("da_f1_mean"),
                "oracle_gain_vs_plain": delta(best.get("da_f1_mean"), plain_da),
                "oracle_gain_vs_smooth": delta(best.get("da_f1_mean"), smooth_da),
                "smooth_rank": config_rank(rank_rows, task, "smooth_k3"),
                "raw_rank": config_rank(rank_rows, task, "raw1"),
                "elastic_r1_rank": config_rank(rank_rows, task, "elastic_r1"),
                "elastic_r2_rank": config_rank(rank_rows, task, "elastic_r2"),
                "structure_helps": (
                    best.get("da_f1_mean") > plain_da
                    if best.get("da_f1_mean") is not None and plain_da is not None
                    else None
                ),
                "elastic_helps": (
                    max_elastic > smooth_da
                    if max_elastic is not None and smooth_da is not None
                    else None
                ),
                "plain_best": best_config == "plain",
                "smooth_best": best_config == "smooth_k3",
                "elastic_best": best_config in {"elastic_r1", "elastic_r2"},
                "raw_best": best_config == "raw1",
            }
        )
    return out


def build_group_summary(matrix, group_field):
    groups = sorted({row[group_field] for row in matrix})
    out = []
    for group in groups:
        rows = [row for row in matrix if row[group_field] == group]
        tasks = sorted({row["task"] for row in rows})
        item = {group_field: group, "n_tasks": len(tasks)}
        for config in [
            "plain",
            "raw1",
            "timepoint1",
            "smooth_k3",
            "umsc075",
            "elastic_r1",
            "elastic_r2",
        ]:
            item[f"{config}_da_mean"] = mean(
                [row["da_f1_mean"] for row in rows if row["config"] == config]
            )
        item["elastic_r1_delta_vs_smooth"] = delta(
            item.get("elastic_r1_da_mean"),
            item.get("smooth_k3_da_mean"),
        )
        item["elastic_r2_delta_vs_smooth"] = delta(
            item.get("elastic_r2_da_mean"),
            item.get("smooth_k3_da_mean"),
        )
        item["smooth_delta_vs_plain"] = delta(item.get("smooth_k3_da_mean"), item.get("plain_da_mean"))
        item["raw_delta_vs_plain"] = delta(item.get("raw1_da_mean"), item.get("plain_da_mean"))
        config_means = {
            config: item.get(f"{config}_da_mean")
            for config in [
                "plain",
                "raw1",
                "timepoint1",
                "smooth_k3",
                "umsc075",
                "elastic_r1",
                "elastic_r2",
            ]
        }
        valid = {key: value for key, value in config_means.items() if value is not None}
        item["best_config_by_mean_da"] = max(valid, key=valid.get) if valid else None
        out.append(item)
    return out


def bool_to_int(value):
    return 1 if value else 0


def build_pair_descriptor(oracle_rows):
    out = []
    for row in oracle_rows:
        source_family = FAMILY.get(row["source"])
        target_family = FAMILY.get(row["target"])
        best_family = FAMILY_BY_CONFIG.get(row.get("best_config"))
        source, target = row["source"], row["target"]
        item = {
            "task": row["task"],
            "source": source,
            "target": target,
            "plain_da": row.get("plain_da"),
            "raw1_da": row.get("raw1_da"),
            "timepoint1_da": row.get("timepoint1_da"),
            "smooth_k3_da": row.get("smooth_k3_da"),
            "elastic_r1_da": row.get("elastic_r1_da"),
            "elastic_r2_da": row.get("elastic_r2_da"),
            "smooth_gain_vs_plain": delta(row.get("smooth_k3_da"), row.get("plain_da")),
            "raw_gain_vs_plain": delta(row.get("raw1_da"), row.get("plain_da")),
            "timepoint_gain_vs_plain": delta(row.get("timepoint1_da"), row.get("plain_da")),
            "elastic_r1_gain_vs_smooth": delta(row.get("elastic_r1_da"), row.get("smooth_k3_da")),
            "elastic_r2_gain_vs_smooth": delta(row.get("elastic_r2_da"), row.get("smooth_k3_da")),
            "oracle_gain_vs_smooth": row.get("oracle_gain_vs_smooth"),
            "best_config": row.get("best_config"),
            "best_family": best_family,
            "source_family": source_family,
            "target_family": target_family,
            "same_country_flag": bool_to_int(source_family == target_family),
            "france_pair_flag": bool_to_int(source_family == "FR" and target_family == "FR"),
            "source_AT": bool_to_int(source_family == "AT"),
            "source_DK": bool_to_int(source_family == "DK"),
            "source_FR": bool_to_int(source_family == "FR"),
            "target_AT": bool_to_int(target_family == "AT"),
            "target_DK": bool_to_int(target_family == "DK"),
            "target_FR": bool_to_int(target_family == "FR"),
        }
        for key in [
            "doy_mean_source",
            "doy_mean_target",
            "doy_mean_gap",
            "doy_std_source",
            "doy_std_target",
            "doy_overlap_ratio",
            "doy_wasserstein",
            "source_temporal_activity",
            "source_pooled_unit_fisher",
            "source_timepoint_unit_fisher",
            "source_smoothed_timepoint_unit_fisher",
            "source_temporal_shape_unit_fisher",
            "source_cov_trace",
            "source_feature_norm",
            "pair_mean_curve_distance",
            "pair_best_shift",
            "pair_best_shift_distance",
            "pair_shift_gain",
            "pair_norm_gap",
            "pair_cov_trace_gap",
        ]:
            item[key] = None
        out.append(item)
    return out


def build_correlations(descriptor_rows):
    descriptor_fields = [
        "same_country_flag",
        "france_pair_flag",
        "source_AT",
        "source_DK",
        "source_FR",
        "target_AT",
        "target_DK",
        "target_FR",
        "doy_mean_gap",
        "doy_overlap_ratio",
        "doy_wasserstein",
        "source_temporal_activity",
        "source_temporal_shape_unit_fisher",
        "source_pooled_unit_fisher",
        "pair_best_shift",
        "pair_shift_gain",
        "pair_norm_gap",
        "pair_cov_trace_gap",
    ]
    outcome_fields = [
        "smooth_gain_vs_plain",
        "raw_gain_vs_plain",
        "timepoint_gain_vs_plain",
        "elastic_r1_gain_vs_smooth",
        "elastic_r2_gain_vs_smooth",
        "oracle_gain_vs_smooth",
    ]
    out = []
    for descriptor in descriptor_fields:
        xs = [row.get(descriptor) for row in descriptor_rows]
        xs = [safe_float(value) for value in xs]
        for outcome in outcome_fields:
            ys = [row.get(outcome) for row in descriptor_rows]
            pairs = [
                (x, y) for x, y in zip(xs, ys)
                if x is not None and y is not None
            ]
            x1 = [y for x, y in pairs if x == 1]
            x0 = [y for x, y in pairs if x == 0]
            out.append(
                {
                    "descriptor": descriptor,
                    "outcome": outcome,
                    "n": len(pairs),
                    "pearson": pearson([x for x, _ in pairs], [y for _, y in pairs]),
                    "group_mean_1": mean(x1),
                    "group_mean_0": mean(x0),
                    "group_mean_diff_1_minus_0": delta(mean(x1), mean(x0)),
                    "note": "exploratory_only_n12" if len(pairs) <= 12 else "",
                }
            )
    return out


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_md_table(handle, rows, fields, headers=None, limit=None):
    headers = headers or fields
    if limit is not None:
        rows = rows[:limit]
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("|" + "|".join("---" for _ in headers) + "|\n")
    for row in rows:
        handle.write("| " + " | ".join(fmt_md(row.get(field)) for field in fields) + " |\n")
    handle.write("\n")


def pivot_da_table(matrix):
    out = []
    for task in TASK_ORDER:
        row = {"task": task}
        for config in CONFIG_ORDER:
            row[config] = value_lookup(matrix, task, config)
        out.append(row)
    return out


def write_summary(path, input_counts, matrix, rank_rows, oracle_rows, source_summary, target_summary, pair_rows, corr_rows):
    config_coverage = []
    for config in CONFIG_ORDER:
        rows = [row for row in matrix if row["config"] == config and row["n"] > 0]
        config_coverage.append(
            {
                "config": config,
                "tasks": len({row["task"] for row in rows}),
                "total_runs": sum(row["n"] for row in rows),
                "mean_n_per_task": mean([row["n"] for row in rows]),
            }
        )
    elastic_rows = [
        {
            "task": row["task"],
            "elastic_r1_gain_vs_smooth": row.get("elastic_r1_gain_vs_smooth"),
            "elastic_r2_gain_vs_smooth": row.get("elastic_r2_gain_vs_smooth"),
        }
        for row in pair_rows
    ]
    corr_top = sorted(
        [row for row in corr_rows if row.get("pearson") is not None],
        key=lambda row: abs(row["pearson"]),
        reverse=True,
    )[:20]

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v285 Regime Diagnostic Summary\n\n")
        handle.write("## 1. Input Logs\n\n")
        write_md_table(handle, input_counts, ["path", "rows", "status"])

        handle.write("## 2. Config Coverage\n\n")
        write_md_table(handle, config_coverage, ["config", "tasks", "total_runs", "mean_n_per_task"])

        handle.write("## 3. Task × Config Matrix\n\n")
        write_md_table(handle, pivot_da_table(matrix), ["task"] + CONFIG_ORDER)

        handle.write("## 4. Task-level Oracle\n\n")
        write_md_table(
            handle,
            oracle_rows,
            ["task", "best_config", "best_da", "oracle_gain_vs_smooth", "smooth_rank", "raw_rank", "elastic_r1_rank", "elastic_r2_rank"],
        )

        handle.write("## 5. Source-group Pattern\n\n")
        write_md_table(handle, source_summary, list(source_summary[0].keys()) if source_summary else [])

        handle.write("## 6. Target-group Pattern\n\n")
        write_md_table(handle, target_summary, list(target_summary[0].keys()) if target_summary else [])

        handle.write("## 7. Elastic Effect Pattern\n\n")
        write_md_table(handle, elastic_rows, ["task", "elastic_r1_gain_vs_smooth", "elastic_r2_gain_vs_smooth"])

        handle.write("## 8. Regime Descriptor Table\n\n")
        write_md_table(
            handle,
            pair_rows,
            [
                "task",
                "source",
                "target",
                "best_config",
                "best_family",
                "source_family",
                "target_family",
                "same_country_flag",
                "france_pair_flag",
                "smooth_gain_vs_plain",
                "elastic_r1_gain_vs_smooth",
                "elastic_r2_gain_vs_smooth",
            ],
        )

        handle.write("## 9. Descriptor-outcome Correlation\n\n")
        write_md_table(
            handle,
            corr_top,
            ["descriptor", "outcome", "n", "pearson", "group_mean_diff_1_minus_0", "note"],
        )

        handle.write("## 10. Minimal Observations\n\n")
        at_source = next((row for row in source_summary if row.get("source") == "AT1"), {})
        fr2_source = next((row for row in source_summary if row.get("source") == "FR2"), {})
        handle.write(
            "- AT1 source elastic_r2_delta_vs_smooth = "
            f"{fmt_md(at_source.get('elastic_r2_delta_vs_smooth'))}\n"
        )
        handle.write(
            "- FR2 source elastic_r2_delta_vs_smooth = "
            f"{fmt_md(fr2_source.get('elastic_r2_delta_vs_smooth'))}\n"
        )
        smooth_mean = mean([row.get("smooth_k3_da_mean") for row in source_summary])
        elastic_r2_mean = mean([row.get("elastic_r2_da_mean") for row in source_summary])
        handle.write(f"- source-group smooth_k3 mean = {fmt_md(smooth_mean)}\n")
        handle.write(f"- source-group elastic_r2 mean = {fmt_md(elastic_r2_mean)}\n")
        handle.write("- descriptor correlations are exploratory only, n=12.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: logs/v285_regime_diagnostic_TIMESTAMP",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input raw_strength_rows.tsv or directory containing it. Can be repeated.",
    )
    args = parser.parse_args()

    default_inputs = [
        "logs/v281_a_group_full_20260622_121128/full12",
        "logs/v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121",
        "logs/v283_umsc_full_20260624_155911",
        "logs/v284_elastic_full12_20260626_181827_20260626_181827",
    ]
    inputs = args.input or default_inputs
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        from datetime import datetime

        out_dir = Path("logs") / f"v285_regime_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_rows, input_counts = load_rows(inputs)
    rows = dedupe_rows(loaded_rows)
    matrix = aggregate_task_config(rows)
    rank_rows = build_rank(matrix)
    oracle_rows = build_oracle(matrix, rank_rows)
    source_summary = build_group_summary(matrix, "source")
    target_summary = build_group_summary(matrix, "target")
    pair_rows = build_pair_descriptor(oracle_rows)
    corr_rows = build_correlations(pair_rows)

    write_tsv(
        out_dir / "task_config_matrix.tsv",
        matrix,
        [
            "task",
            "source",
            "target",
            "config",
            "n",
            "source_self_mean",
            "source_on_target_mean",
            "da_f1_mean",
            "da_gain_mean",
            "da_f1_std",
            "da_gain_std",
        ],
    )
    write_tsv(
        out_dir / "task_config_rank.tsv",
        rank_rows,
        [
            "task",
            "source",
            "target",
            "rank",
            "config",
            "da_f1_mean",
            "da_gain_mean",
            "source_on_target_mean",
            "delta_vs_plain",
            "delta_vs_smooth_k3",
        ],
    )
    write_tsv(
        out_dir / "task_oracle_summary.tsv",
        oracle_rows,
        [
            "task",
            "source",
            "target",
            "plain_da",
            "raw1_da",
            "timepoint1_da",
            "smooth_k3_da",
            "umsc075_da",
            "elastic_r1_da",
            "elastic_r2_da",
            "best_config",
            "best_da",
            "second_best_config",
            "second_best_da",
            "oracle_gain_vs_plain",
            "oracle_gain_vs_smooth",
            "smooth_rank",
            "raw_rank",
            "elastic_r1_rank",
            "elastic_r2_rank",
            "structure_helps",
            "elastic_helps",
            "plain_best",
            "smooth_best",
            "elastic_best",
            "raw_best",
        ],
    )
    group_fields = [
        "n_tasks",
        "plain_da_mean",
        "raw1_da_mean",
        "timepoint1_da_mean",
        "smooth_k3_da_mean",
        "umsc075_da_mean",
        "elastic_r1_da_mean",
        "elastic_r2_da_mean",
        "elastic_r1_delta_vs_smooth",
        "elastic_r2_delta_vs_smooth",
        "smooth_delta_vs_plain",
        "raw_delta_vs_plain",
        "best_config_by_mean_da",
    ]
    write_tsv(out_dir / "source_group_summary.tsv", source_summary, ["source"] + group_fields)
    write_tsv(out_dir / "target_group_summary.tsv", target_summary, ["target"] + group_fields)
    descriptor_fields = list(pair_rows[0].keys()) if pair_rows else []
    write_tsv(out_dir / "pair_regime_descriptor.tsv", pair_rows, descriptor_fields)
    corr_fields = [
        "descriptor",
        "outcome",
        "n",
        "pearson",
        "group_mean_1",
        "group_mean_0",
        "group_mean_diff_1_minus_0",
        "note",
    ]
    write_tsv(out_dir / "descriptor_outcome_correlation.tsv", corr_rows, corr_fields)
    write_summary(
        out_dir / "regime_diagnostic_summary.md",
        input_counts,
        matrix,
        rank_rows,
        oracle_rows,
        source_summary,
        target_summary,
        pair_rows,
        corr_rows,
    )

    print("Wrote:", out_dir / "task_config_matrix.tsv")
    print("Wrote:", out_dir / "task_config_rank.tsv")
    print("Wrote:", out_dir / "task_oracle_summary.tsv")
    print("Wrote:", out_dir / "source_group_summary.tsv")
    print("Wrote:", out_dir / "target_group_summary.tsv")
    print("Wrote:", out_dir / "pair_regime_descriptor.tsv")
    print("Wrote:", out_dir / "descriptor_outcome_correlation.tsv")
    print("Wrote:", out_dir / "regime_diagnostic_summary.md")


if __name__ == "__main__":
    main()
