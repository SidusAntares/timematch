#!/usr/bin/env python3
import sys
from pathlib import Path

from summarize_v243b_structure_target_stage2b import (
    enrich_deltas,
    mean,
    parse_log,
    stdev,
    summarize,
    values,
    write_tsv,
)


def config_label(config):
    if config == "plain":
        return "plain"
    if config == "trainable_s003_reg000":
        return "reshaper"
    if config.startswith("raw_global_"):
        return "raw_detached" if config.endswith("_detached") else "raw"
    if config.startswith("trainable_s003_raw_global_"):
        return "reshaper_raw_detached" if config.endswith("_detached") else "reshaper_raw"
    return None


def rows_by_task_seed_label(rows):
    mapping = {}
    for row in rows:
        if row["status"] != "ok" or row["da_f1"] is None:
            continue
        label = config_label(row["config"])
        if label is None:
            continue
        mapping[(row["task"], row["seed"], label)] = row
    return mapping


def diff(left, right):
    if left is None or right is None:
        return None
    return left - right


def config_value(mapping, task, seed, key):
    row = mapping.get((task, seed, key))
    return None if row is None else row["da_f1"]


def make_contrast_rows(rows):
    mapping = rows_by_task_seed_label(rows)
    task_seeds = sorted({(row["task"], row["seed"]) for row in rows})
    out = []
    for task, seed in task_seeds:
        plain = config_value(mapping, task, seed, "plain")
        raw_detached = config_value(mapping, task, seed, "raw_detached")
        raw = config_value(mapping, task, seed, "raw")
        reshaper = config_value(mapping, task, seed, "reshaper")
        reshaper_raw_detached = config_value(mapping, task, seed, "reshaper_raw_detached")
        reshaper_raw = config_value(mapping, task, seed, "reshaper_raw")
        if plain is None:
            continue
        raw_grad_effect = diff(raw, raw_detached)
        reshaper_direct_effect = diff(reshaper, plain)
        reshaper_loss_value_effect = diff(reshaper_raw_detached, reshaper)
        combo_grad_effect = diff(reshaper_raw, reshaper_raw_detached)
        interaction_like = diff(combo_grad_effect, raw_grad_effect)
        out.append(
            {
                "task": task,
                "seed": seed,
                "plain": plain,
                "raw_detached": raw_detached,
                "raw": raw,
                "reshaper": reshaper,
                "reshaper_raw_detached": reshaper_raw_detached,
                "reshaper_raw": reshaper_raw,
                "raw_loss_value_effect": diff(raw_detached, plain),
                "raw_grad_effect": raw_grad_effect,
                "raw_total_effect": diff(raw, plain),
                "reshaper_direct_effect": reshaper_direct_effect,
                "reshaper_loss_value_effect": reshaper_loss_value_effect,
                "reshaper_combo_grad_effect": combo_grad_effect,
                "reshaper_combo_total_effect": diff(reshaper_raw, plain),
                "interaction_like": interaction_like,
                "complete_core": int(
                    all(
                        value is not None
                        for value in [
                            raw_detached,
                            raw,
                            reshaper,
                            reshaper_raw_detached,
                            reshaper_raw,
                        ]
                    )
                ),
            }
        )
    return out


def summarize_contrasts(contrast_rows):
    grouped = {}
    for row in contrast_rows:
        grouped.setdefault(row["task"], []).append(row)
    summary = []
    effect_fields = [
        "raw_loss_value_effect",
        "raw_grad_effect",
        "raw_total_effect",
        "reshaper_direct_effect",
        "reshaper_loss_value_effect",
        "reshaper_combo_grad_effect",
        "reshaper_combo_total_effect",
        "interaction_like",
    ]
    for task, group in sorted(grouped.items()):
        out = {
            "task": task,
            "n": len(group),
            "complete_core": sum(row["complete_core"] for row in group),
        }
        for field in effect_fields:
            vals = values(group, field)
            out[f"{field}_mean"] = mean(vals)
            out[f"{field}_std"] = stdev(vals)
            out[f"{field}_pos"] = sum(value > 0 for value in vals)
        summary.append(out)
    return summary


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_v243b_counterfactual_chain.py LOG_DIR")
    root = Path(sys.argv[1])
    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]
    enrich_deltas(rows)

    row_fields = [
        "task",
        "seed",
        "config",
        "mechanism",
        "structure_target",
        "detached",
        "compact_weight",
        "source_self_f1",
        "source_on_target_f1",
        "da_f1",
        "da_gain",
        "delta_vs_plain",
        "source_loss",
        "source_cls_loss",
        "source_compact_loss",
        "source_compact_raw_loss",
        "source_compact_reshaped_loss",
        "source_spatial_delta",
        "source_temporal_delta",
        "status",
        "log",
    ]
    write_tsv(root / "summary.tsv", rows, row_fields)

    config_fields = [
        "task",
        "config",
        "n",
        "ok_count",
        "da_mean",
        "da_std",
        "source_on_target_mean",
        "source_on_target_std",
        "da_gain_mean",
        "delta_vs_plain_mean",
        "delta_vs_plain_std",
        "delta_vs_plain_pos",
        "compact_loss_mean",
        "compact_raw_loss_mean",
        "compact_reshaped_loss_mean",
        "spatial_delta_mean",
        "temporal_delta_mean",
    ]
    write_tsv(root / "config_summary.tsv", summarize(rows, ["task", "config"]), config_fields)

    contrast_rows = make_contrast_rows(rows)
    contrast_fields = [
        "task",
        "seed",
        "plain",
        "raw_detached",
        "raw",
        "reshaper",
        "reshaper_raw_detached",
        "reshaper_raw",
        "raw_loss_value_effect",
        "raw_grad_effect",
        "raw_total_effect",
        "reshaper_direct_effect",
        "reshaper_loss_value_effect",
        "reshaper_combo_grad_effect",
        "reshaper_combo_total_effect",
        "interaction_like",
        "complete_core",
    ]
    write_tsv(root / "counterfactual_contrasts.tsv", contrast_rows, contrast_fields)

    contrast_summary_fields = ["task", "n", "complete_core"]
    for field in [
        "raw_loss_value_effect",
        "raw_grad_effect",
        "raw_total_effect",
        "reshaper_direct_effect",
        "reshaper_loss_value_effect",
        "reshaper_combo_grad_effect",
        "reshaper_combo_total_effect",
        "interaction_like",
    ]:
        contrast_summary_fields.extend([f"{field}_mean", f"{field}_std", f"{field}_pos"])
    write_tsv(
        root / "counterfactual_contrast_summary.tsv",
        summarize_contrasts(contrast_rows),
        contrast_summary_fields,
    )

    print("Wrote:", root / "summary.tsv")
    print("Wrote:", root / "config_summary.tsv")
    print("Wrote:", root / "counterfactual_contrasts.tsv")
    print("Wrote:", root / "counterfactual_contrast_summary.tsv")


if __name__ == "__main__":
    main()
