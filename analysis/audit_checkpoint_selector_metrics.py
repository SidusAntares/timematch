import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict


ORACLE_F1_BY_TASK_EPOCH = {
    "30TXT->31TCJ": {30: 0.7904, 50: 0.8020, 70: 0.7982, 100: 0.7976},
    "30TXT->32VNH": {30: 0.6455, 50: 0.6490, 70: 0.6570, 100: 0.6444},
    "30TXT->33UVP": {30: 0.6986, 50: 0.7605, 70: 0.7388, 100: 0.7637},
    "31TCJ->30TXT": {30: 0.7672, 50: 0.7553, 70: 0.7164, 100: 0.7334},
    "31TCJ->32VNH": {30: 0.5724, 50: 0.6957, 70: 0.5004, 100: 0.5526},
    "31TCJ->33UVP": {30: 0.6676, 50: 0.6027, 70: 0.5888, 100: 0.6509},
    "32VNH->30TXT": {30: 0.5856, 50: 0.5834, 70: 0.5927, 100: 0.5807},
    "32VNH->31TCJ": {30: 0.5270, 50: 0.5474, 70: 0.5295, 100: 0.5361},
    "32VNH->33UVP": {30: 0.6674, 50: 0.4531, 70: 0.5654, 100: 0.5789},
    "33UVP->30TXT": {30: 0.6694, 50: 0.6786, 70: 0.6815, 100: 0.6780},
    "33UVP->31TCJ": {30: 0.6081, 50: 0.6158, 70: 0.6501, 100: 0.6357},
    "33UVP->32VNH": {30: 0.7765, 50: 0.7311, 70: 0.7034, 100: 0.7047},
}

METRIC_GROUPS = {
    "robust_perturbation": [
        ("selection_temporal_perturbation_score", "max"),
        ("selection_perturbation_score", "max"),
    ],
    "distribution_health": [
        ("selection_class_entropy", "max"),
        ("selection_high_conf_class_entropy", "max"),
        ("selection_effective_class_fraction", "max"),
        ("selection_high_conf_effective_class_fraction", "max"),
        ("selection_max_class_fraction", "min"),
        ("selection_high_conf_max_class_fraction", "min"),
    ],
    "confidence_agreement": [
        ("selection_coverage", "max"),
        ("selection_mean_confidence", "max"),
        ("selection_teacher_student_agreement", "max"),
        ("selection_prediction_entropy", "min"),
    ],
    "source_prior_shift": [
        ("selection_source_prior_js", "min"),
        ("selection_shift_stability", "max"),
    ],
    "trajectory": [
        ("selection_trajectory_base_score", "max"),
        ("selection_trajectory_late_gain_ratio", "min"),
        ("selection_trajectory_multiplier", "max"),
        ("selection_trajectory_excess_late_gain", "min"),
    ],
    "metric_bank_existing": [
        ("metric_bank_rank_score", "max"),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dirs",
        nargs="+",
        required=True,
        help="one or more log directories containing selection logs",
    )
    parser.add_argument("--log_pattern", default="*.log", help="log glob pattern inside each directory")
    parser.add_argument("--output_json", required=True, help="path to write audit JSON")
    parser.add_argument("--output_md", required=True, help="path to write audit markdown")
    parser.add_argument("--near_gap", type=float, default=0.01, help="near-hit threshold against oracle F1")
    args = parser.parse_args()

    summaries = []
    for log_dir in args.log_dirs:
        summaries.extend(extract_summaries_from_dir(log_dir, args.log_pattern))

    rows = build_candidate_rows(summaries)
    if not rows:
        raise RuntimeError("No aligned candidate rows found. Check log_dirs/log_pattern and oracle table coverage.")

    metric_names = sorted({metric for row in rows for metric in row["metrics"]})
    single_results = []
    for metric_name in metric_names:
        single_results.append(evaluate_metric(rows, metric_name, "max", args.near_gap))
        single_results.append(evaluate_metric(rows, metric_name, "min", args.near_gap))
    single_results = [item for item in single_results if item["tasks"] > 0]
    single_results.sort(key=lambda item: (item["avg_selected_f1"], item["hit_rate"], item["near_rate"]), reverse=True)

    group_results = []
    for group_name, specs in METRIC_GROUPS.items():
        result = evaluate_group(rows, group_name, specs, args.near_gap)
        if result["tasks"] > 0:
            group_results.append(result)
    group_results.sort(key=lambda item: (item["avg_selected_f1"], item["hit_rate"], item["near_rate"]), reverse=True)

    pipelines = []
    pipelines.extend({"kind": "single", **item} for item in single_results)
    pipelines.extend({"kind": "group", **item} for item in group_results)
    loto = leave_one_task_out(rows, pipelines, args.near_gap)
    variance = compute_variance_report(rows, metric_names)

    payload = {
        "log_dirs": args.log_dirs,
        "num_rows": len(rows),
        "tasks": sorted({row["task"] for row in rows}),
        "top_single_metrics": single_results[:20],
        "group_results": group_results,
        "leave_one_task_out": loto,
        "variance_report": variance,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload, args.near_gap)
    print(json.dumps(payload, indent=2))


def extract_summaries_from_dir(log_dir, pattern):
    summaries = []
    for path in sorted(glob.glob(os.path.join(log_dir, pattern))):
        text = read_text(path)
        fallback_task = extract_task_from_log(text, path)
        objects = extract_json_objects(text)
        for index, (obj, _start, end) in enumerate(objects):
            if isinstance(obj, dict) and "all_candidates" in obj and "best_weights_checkpoint" in obj:
                next_start = objects[index + 1][1] if index + 1 < len(objects) else len(text)
                task = extract_task_from_log(text[end:next_start], path) or fallback_task
                obj["_log_path"] = path
                obj["_task"] = task
                summaries.append(obj)
    return summaries


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def extract_task_from_log(text, path):
    match = re.search(r"SELECTION_RESULT\|([^|]+)\|([^|]+)\|", text)
    if match:
        return f"{match.group(1)}->{match.group(2)}"
    name = os.path.basename(path)
    match = re.search(r"([0-9A-Z]{5})_to_([0-9A-Z]{5})", name)
    if match:
        return f"{match.group(1)}->{match.group(2)}"
    return None


def extract_json_objects(text):
    decoder = json.JSONDecoder()
    objects = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        objects.append((obj, start, start + end))
        index = start + end
    return objects


def build_candidate_rows(summaries):
    rows = []
    seen = set()
    for summary in summaries:
        task = summary.get("_task")
        if task not in ORACLE_F1_BY_TASK_EPOCH:
            continue
        oracle_by_epoch = ORACLE_F1_BY_TASK_EPOCH[task]
        oracle_epoch, oracle_f1 = max(oracle_by_epoch.items(), key=lambda pair: pair[1])
        for candidate in summary.get("all_candidates", []):
            epoch = parse_epoch(candidate.get("weights_checkpoint"))
            if epoch not in oracle_by_epoch:
                continue
            key = (summary.get("_log_path"), task, epoch)
            if key in seen:
                continue
            seen.add(key)
            metrics = numeric_metrics(candidate)
            rows.append(
                {
                    "task": task,
                    "epoch": epoch,
                    "checkpoint": candidate.get("weights_checkpoint"),
                    "oracle_epoch": oracle_epoch,
                    "oracle_f1": oracle_f1,
                    "candidate_f1": oracle_by_epoch[epoch],
                    "gap": oracle_by_epoch[epoch] - oracle_f1,
                    "metrics": metrics,
                    "log_path": summary.get("_log_path"),
                }
            )
    return rows


def parse_epoch(checkpoint):
    match = re.search(r"epoch[_-](\d+)", str(checkpoint))
    if not match:
        return None
    return int(match.group(1))


def numeric_metrics(candidate):
    metrics = {}
    for key, value in candidate.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[key] = float(value)
    return metrics


def evaluate_metric(rows, metric_name, direction, near_gap):
    selected = []
    for task, task_rows in group_rows_by_task(rows).items():
        available = [row for row in task_rows if metric_name in row["metrics"]]
        if not available:
            continue
        reverse = direction == "max"
        best = sorted(available, key=lambda row: row["metrics"][metric_name], reverse=reverse)[0]
        selected.append(best)
    return summarize_selection(
        name=f"{metric_name}:{direction}",
        selected_rows=selected,
        near_gap=near_gap,
        details={"metric": metric_name, "direction": direction},
    )


def evaluate_group(rows, group_name, specs, near_gap):
    selected = []
    for task, task_rows in group_rows_by_task(rows).items():
        scored = score_rows_by_rank(task_rows, specs)
        if not scored:
            continue
        best = max(scored, key=lambda item: item[1])[0]
        selected.append(best)
    return summarize_selection(
        name=group_name,
        selected_rows=selected,
        near_gap=near_gap,
        details={"group": group_name, "specs": specs},
    )


def score_rows_by_rank(rows, specs):
    scores = {id(row): 0.0 for row in rows}
    used = 0
    for metric_name, direction in specs:
        available = [row for row in rows if metric_name in row["metrics"]]
        if len(available) < 2:
            continue
        reverse = direction == "max"
        ordered = sorted(available, key=lambda row: row["metrics"][metric_name], reverse=reverse)
        denom = max(len(ordered) - 1, 1)
        for rank, row in enumerate(ordered):
            scores[id(row)] += 1.0 - rank / denom
        used += 1
    if used == 0:
        return []
    return [(row, scores[id(row)] / used) for row in rows if id(row) in scores]


def summarize_selection(name, selected_rows, near_gap, details):
    tasks = len(selected_rows)
    if tasks == 0:
        return {"name": name, "tasks": 0, **details}
    hits = [row["epoch"] == row["oracle_epoch"] for row in selected_rows]
    near_hits = [abs(row["gap"]) <= near_gap for row in selected_rows]
    avg_selected_f1 = sum(row["candidate_f1"] for row in selected_rows) / tasks
    avg_oracle_f1 = sum(row["oracle_f1"] for row in selected_rows) / tasks
    return {
        "name": name,
        "tasks": tasks,
        "hits": int(sum(hits)),
        "hit_rate": float(sum(hits) / tasks),
        "near_hits": int(sum(near_hits)),
        "near_rate": float(sum(near_hits) / tasks),
        "avg_selected_f1": float(avg_selected_f1),
        "avg_oracle_f1": float(avg_oracle_f1),
        "avg_gap": float(avg_selected_f1 - avg_oracle_f1),
        "selected": [
            {
                "task": row["task"],
                "selected_epoch": row["epoch"],
                "oracle_epoch": row["oracle_epoch"],
                "selected_f1": row["candidate_f1"],
                "oracle_f1": row["oracle_f1"],
                "gap": row["gap"],
            }
            for row in sorted(selected_rows, key=lambda item: item["task"])
        ],
        **details,
    }


def leave_one_task_out(rows, pipelines, near_gap):
    task_map = group_rows_by_task(rows)
    tasks = sorted(task_map)
    heldout = []
    for held_task in tasks:
        train_rows = [row for row in rows if row["task"] != held_task]
        test_rows = task_map[held_task]
        train_scores = []
        for pipeline in pipelines:
            result = rerun_pipeline(train_rows, pipeline, near_gap)
            if result["tasks"] == 0:
                continue
            train_scores.append((pipeline, result))
        if not train_scores:
            continue
        train_scores.sort(
            key=lambda pair: (pair[1]["avg_selected_f1"], pair[1]["hit_rate"], pair[1]["near_rate"]),
            reverse=True,
        )
        chosen_pipeline = train_scores[0][0]
        test_result = rerun_pipeline(test_rows, chosen_pipeline, near_gap)
        heldout.append(
            {
                "heldout_task": held_task,
                "chosen_pipeline": chosen_pipeline["name"],
                "chosen_kind": chosen_pipeline.get("kind"),
                "train_avg_selected_f1": train_scores[0][1]["avg_selected_f1"],
                "train_avg_gap": train_scores[0][1]["avg_gap"],
                "test_result": test_result,
            }
        )
    selected_rows = []
    for item in heldout:
        selected_rows.extend(item["test_result"].get("selected", []))
    if heldout:
        hits = sum(1 for item in heldout if item["test_result"].get("hits", 0) > 0)
        near_hits = sum(1 for item in heldout if item["test_result"].get("near_hits", 0) > 0)
        avg_gap = sum(item["test_result"].get("avg_gap", 0.0) for item in heldout) / len(heldout)
    else:
        hits = 0
        near_hits = 0
        avg_gap = 0.0
    return {
        "tasks": len(heldout),
        "hits": hits,
        "hit_rate": float(hits / len(heldout)) if heldout else 0.0,
        "near_hits": near_hits,
        "near_rate": float(near_hits / len(heldout)) if heldout else 0.0,
        "avg_gap": float(avg_gap),
        "heldout": heldout,
    }


def rerun_pipeline(rows, pipeline, near_gap):
    if pipeline.get("kind") == "single":
        return evaluate_metric(rows, pipeline["metric"], pipeline["direction"], near_gap)
    if pipeline.get("kind") == "group":
        return evaluate_group(rows, pipeline["group"], pipeline["specs"], near_gap)
    raise ValueError(f"Unknown pipeline kind: {pipeline.get('kind')}")


def compute_variance_report(rows, metric_names):
    report = {}
    for task, task_rows in group_rows_by_task(rows).items():
        low_variance = []
        available_count = 0
        for metric_name in metric_names:
            values = [row["metrics"][metric_name] for row in task_rows if metric_name in row["metrics"]]
            if len(values) < 2:
                continue
            available_count += 1
            value_range = max(values) - min(values)
            if value_range < 1e-6:
                low_variance.append(metric_name)
        report[task] = {
            "candidate_count": len(task_rows),
            "available_metric_count": available_count,
            "low_variance_metric_count": len(low_variance),
            "low_variance_metrics": low_variance[:20],
        }
    return report


def group_rows_by_task(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    return grouped


def write_json(path, payload):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_markdown(path, payload, near_gap):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append("# v2.5.4b Offline Selector Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- logs: `{', '.join(payload['log_dirs'])}`")
    lines.append(f"- candidate rows: `{payload['num_rows']}`")
    lines.append(f"- tasks: `{len(payload['tasks'])}`")
    lines.append(f"- near-hit threshold: `{near_gap}` macro F1 gap")
    lines.append("")
    lines.append("## Top Single Metrics")
    lines.append("")
    lines.append("| rank | selector | hit | near | selected avg | oracle avg | gap |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for rank, item in enumerate(payload["top_single_metrics"][:15], start=1):
        lines.append(format_result_row(rank, item))
    lines.append("")
    lines.append("## Metric Groups")
    lines.append("")
    lines.append("| rank | selector | hit | near | selected avg | oracle avg | gap |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for rank, item in enumerate(payload["group_results"], start=1):
        lines.append(format_result_row(rank, item))
    lines.append("")
    lines.append("## Leave-One-Task-Out")
    lines.append("")
    loto = payload["leave_one_task_out"]
    lines.append(
        f"- held-out hit: `{loto['hits']}/{loto['tasks']}`"
        f" ({loto['hit_rate']:.3f})"
    )
    lines.append(
        f"- held-out near-hit: `{loto['near_hits']}/{loto['tasks']}`"
        f" ({loto['near_rate']:.3f})"
    )
    lines.append(f"- held-out avg gap: `{loto['avg_gap']:.4f}`")
    lines.append("")
    lines.append("| heldout task | chosen pipeline | selected | oracle | gap |")
    lines.append("|---|---|---:|---:|---:|")
    for item in loto["heldout"]:
        selected = item["test_result"].get("selected", [])
        if selected:
            row = selected[0]
            lines.append(
                f"| `{item['heldout_task']}` | `{item['chosen_pipeline']}` | "
                f"`epoch{row['selected_epoch']}` | `epoch{row['oracle_epoch']}` | {row['gap']:.4f} |"
            )
    lines.append("")
    lines.append("## Variance Check")
    lines.append("")
    lines.append("| task | candidates | metrics | low-variance metrics |")
    lines.append("|---|---:|---:|---:|")
    for task, item in sorted(payload["variance_report"].items()):
        lines.append(
            f"| `{task}` | {item['candidate_count']} | {item['available_metric_count']} | "
            f"{item['low_variance_metric_count']} |"
        )
    lines.append("")
    lines.append("## Decision Rule")
    lines.append("")
    lines.append(
        "If leave-one-task-out hit rate is not clearly above `0.50`, "
        "do not launch another GPU selector run based only on these metrics."
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def format_result_row(rank, item):
    return (
        f"| {rank} | `{item['name']}` | {item['hits']}/{item['tasks']} | "
        f"{item['near_hits']}/{item['tasks']} | {item['avg_selected_f1']:.4f} | "
        f"{item['avg_oracle_f1']:.4f} | {item['avg_gap']:.4f} |"
    )


if __name__ == "__main__":
    main()
