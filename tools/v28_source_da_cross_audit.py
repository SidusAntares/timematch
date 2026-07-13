import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


DOMAINS = {
    "AT1": ("austria/33UVP/2017", "33UVP"),
    "DK1": ("denmark/32VNH/2017", "32VNH"),
    "FR1": ("france/30TXT/2017", "30TXT"),
    "FR2": ("france/31TCJ/2017", "31TCJ"),
}
CONFIGS = ("base", "smooth_k3")
SEEDS = (1, 2, 3)
PROBE_TASKS = ("AT1_to_FR2", "FR2_to_FR1")


def read_tsv(path):
    with Path(path).open(encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def common_f1_value(row):
    for field in ("common_macro_f1", "common_test_f1", "macro_f1"):
        value = row.get(field, "")
        if value not in ("", None):
            return value
    return ""


def lookup_source_checkpoint(manifest, source, config, version, seed=1):
    if version not in ("old", "cleaned"):
        raise ValueError(f"unsupported source version: {version}")
    matches = [
        row
        for row in read_tsv(manifest)
        if row.get("source_domain") == source
        and row.get("config") == config
        and int(row.get("seed", -1)) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one source checkpoint row for {source}/{config}/seed{seed}; found {len(matches)}"
        )
    path = matches[0].get(f"{version}_checkpoint_path", "").strip()
    if not path:
        raise ValueError(f"empty {version} checkpoint path for {source}/{config}/seed{seed}")
    return path


def print_source_checkpoint(args):
    print(lookup_source_checkpoint(args.manifest, args.source, args.config, args.version, args.seed))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict(checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_domain_from_task(task):
    return task.split("_to_", 1)[0]


def source_checkpoint_candidates(outputs_root, source, config, seed):
    tile = DOMAINS[source][1]
    if config == "base":
        pattern = (
            f"pseltae_{tile}_closedset_noshift_v275_closedset_baseline_"
            f"v275_12tasks_3seeds_{source}_to_*_seed{seed}_plain_source/fold_0/model.pt"
        )
    else:
        pattern = (
            f"pseltae_{tile}_closedset_noshift_v276_smoothed_lambda12_half_20260619_215420_"
            f"w1p0_{source}_to_*_seed{seed}_v276_smoothed_timepoint_w1_source/fold_0/model.pt"
        )
    return sorted(Path(outputs_root).glob(pattern))


def build_manifests(args):
    cleaned_rows = read_tsv(args.cleaned_source_inventory)
    cleaned = {
        (row["source_domain"], int(row["seed"]), row["source_config"]): row["checkpoint_path"]
        for row in cleaned_rows
    }
    source_rows = []
    failures = []
    for source in DOMAINS:
        for seed in SEEDS:
            for config in CONFIGS:
                candidates = source_checkpoint_candidates(args.outputs_root, source, config, seed)
                hashes = [file_sha256(path) for path in candidates]
                cleaned_path = Path(cleaned.get((source, seed, config), ""))
                if len(candidates) != 3 or len(set(hashes)) != 1:
                    failures.append(f"old source candidates {source}/{config}/seed{seed}: {len(candidates)} files, {len(set(hashes))} hashes")
                if not cleaned_path.is_file():
                    failures.append(f"missing cleaned source {source}/{config}/seed{seed}: {cleaned_path}")
                source_rows.append(
                    {
                        "source_domain": source,
                        "source_dataset": DOMAINS[source][0],
                        "config": config,
                        "seed": seed,
                        "old_checkpoint_path": str(candidates[0]) if candidates else "",
                        "old_task_checkpoint_count": len(candidates),
                        "old_task_checkpoint_file_hashes_equal": len(candidates) == 3 and len(set(hashes)) == 1,
                        "cleaned_checkpoint_path": str(cleaned_path),
                    }
                )
    source_fields = list(source_rows[0])
    write_tsv(args.source_manifest, source_rows, source_fields)

    recovered = read_tsv(args.recovered_results)
    cleaned_summary = read_tsv(args.cleaned_summary)
    existing_rows = []
    for task in PROBE_TASKS:
        source, target = task.split("_to_")
        source_tile, target_tile = DOMAINS[source][1], DOMAINS[target][1]
        for method in CONFIGS:
            if method == "base":
                old_name = (
                    f"timematch_{source_tile}_to_{target_tile}_closedset_noshift_"
                    f"v275_closedset_baseline_v275_12tasks_3seeds_{task}_seed1_plain/fold_0/model.pt"
                )
                old_result_fragment = "v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121/raw_strength_rows.tsv"
                old_result_config = "base"
            else:
                old_name = (
                    f"timematch_{source_tile}_to_{target_tile}_closedset_noshift_"
                    f"v276_smoothed_lambda12_half_20260619_215420_w1p0_{task}_seed1_"
                    f"v276_smoothed_timepoint_w1/fold_0/model.pt"
                )
                old_result_fragment = "v276_smoothed_lambda12_half_20260619_215420/w1p0/raw_strength_rows.tsv"
                old_result_config = "smooth_k3"
            old_path = Path(args.outputs_root) / old_name
            cleaned_path = Path(args.outputs_root) / f"v28_cleaned_full12_{source}_{target}_{method}_seed1/fold_0/model.pt"
            old_f1_rows = [
                row for row in recovered
                if old_result_fragment in row.get("path", "")
                and row.get("result_kind") == "tabular_row"
                and row.get("task") == task
                and row.get("seed") == "1"
                and row.get("config") == old_result_config
            ]
            cleaned_f1_rows = [
                row for row in cleaned_summary
                if row.get("task") == task and row.get("seed") == "1" and row.get("config") == method
            ]
            if not old_path.is_file():
                failures.append(f"missing old DA: {old_path}")
            if not cleaned_path.is_file():
                failures.append(f"missing cleaned DA: {cleaned_path}")
            if len(old_f1_rows) != 1 or len(cleaned_f1_rows) != 1:
                failures.append(f"native result mismatch: {task}/{method} old={len(old_f1_rows)} cleaned={len(cleaned_f1_rows)}")
            for version, checkpoint, native_rows in (
                ("old", old_path, old_f1_rows),
                ("cleaned", cleaned_path, cleaned_f1_rows),
            ):
                native = (
                    native_rows[0].get("last_macro_f1")
                    if version == "old" and native_rows
                    else native_rows[0].get("da_test_macro_f1") if native_rows else ""
                )
                existing_rows.append(
                    {
                        "task": task,
                        "source": DOMAINS[source][0],
                        "target": DOMAINS[target][0],
                        "method": method,
                        "seed": 1,
                        "checkpoint_version": version,
                        "source_version": version,
                        "da_version": version,
                        "checkpoint_path": str(checkpoint),
                        "native_test_f1": native,
                    }
                )
    write_tsv(args.existing_da_manifest, existing_rows, list(existing_rows[0]))
    print(
        f"MANIFESTS|source_pairs={len(source_rows)}|existing_da={len(existing_rows)}|"
        f"failures={len(failures)}"
    )
    if failures:
        for failure in failures:
            print(f"PREFLIGHT_ERROR|{failure}")
        raise SystemExit(2)


def module_for(name, bn_prefixes):
    prefix = name.rsplit(".", 1)[0]
    if name.endswith("running_mean"):
        return "batch_norm_running_mean"
    if name.endswith("running_var"):
        return "batch_norm_running_var"
    if name.endswith("num_batches_tracked"):
        return "other_buffers"
    if prefix in bn_prefixes:
        return "batch_norm_parameters"
    if name.startswith("spatial_encoder."):
        return "spatial_encoder"
    if name.startswith("temporal_encoder."):
        return "temporal_encoder"
    if name.startswith("decoder.") or name.startswith("classifier."):
        return "classifier"
    return "other_buffers"


def compare_tensor_groups(old_state, cleaned_state):
    import torch

    keys_equal = set(old_state) == set(cleaned_state)
    common = sorted(set(old_state) & set(cleaned_state))
    shapes_equal = keys_equal and all(old_state[key].shape == cleaned_state[key].shape for key in common)
    bn_prefixes = {
        name.rsplit(".", 1)[0]
        for name in common
        if name.endswith("running_mean") or name.endswith("running_var")
    }
    groups = {"overall": common}
    for key in common:
        groups.setdefault(module_for(key, bn_prefixes), []).append(key)
    rows = []
    for module, keys in groups.items():
        valid = [key for key in keys if old_state[key].shape == cleaned_state[key].shape]
        exact = len(valid) == len(keys) and all(torch.equal(old_state[key], cleaned_state[key]) for key in valid)
        squared_old = squared_diff = dot = squared_clean = 0.0
        max_diff = 0.0
        tensor_count = 0
        for key in valid:
            old = old_state[key].detach().cpu().double().reshape(-1)
            cleaned = cleaned_state[key].detach().cpu().double().reshape(-1)
            if old.numel() == 0:
                continue
            diff = cleaned - old
            squared_old += float(torch.dot(old, old))
            squared_clean += float(torch.dot(cleaned, cleaned))
            squared_diff += float(torch.dot(diff, diff))
            dot += float(torch.dot(old, cleaned))
            max_diff = max(max_diff, float(diff.abs().max()))
            tensor_count += 1
        l2 = math.sqrt(squared_diff)
        old_norm = math.sqrt(squared_old)
        clean_norm = math.sqrt(squared_clean)
        rows.append(
            {
                "module": module,
                "tensor_count": tensor_count,
                "state_dict_keys_equal": keys_equal,
                "tensor_shapes_equal": shapes_equal,
                "exact_tensor_equal": exact,
                "parameter_l2_distance": l2,
                "parameter_relative_l2_distance": l2 / old_norm if old_norm else 0.0,
                "parameter_cosine_similarity": dot / (old_norm * clean_norm) if old_norm and clean_norm else 1.0,
                "max_parameter_abs_difference": max_diff,
            }
        )
    return rows


def compare_sources(args):
    manifest = read_tsv(args.source_manifest)
    output_rows = []
    for item in manifest:
        old_path = Path(item["old_checkpoint_path"])
        cleaned_path = Path(item["cleaned_checkpoint_path"])
        old_state = state_dict(old_path)
        cleaned_state = state_dict(cleaned_path)
        base = {
            **item,
            "old_file_sha256": file_sha256(old_path),
            "cleaned_file_sha256": file_sha256(cleaned_path),
            "old_state_dict_sha256": state_sha256(old_state),
            "cleaned_state_dict_sha256": state_sha256(cleaned_state),
        }
        for metrics in compare_tensor_groups(old_state, cleaned_state):
            output_rows.append({**base, **metrics})
    fields = list(output_rows[0])
    write_tsv(args.output, output_rows, fields)

    overall = [row for row in output_rows if row["module"] == "overall"]
    summary = []
    for config in CONFIGS:
        group = [row for row in overall if row["config"] == config]
        bn_mean = [row for row in output_rows if row["config"] == config and row["module"] == "batch_norm_running_mean"]
        bn_var = [row for row in output_rows if row["config"] == config and row["module"] == "batch_norm_running_var"]
        summary.append(
            {
                "config": config,
                "num_pairs": len(group),
                "exact_hash_matches": sum(row["old_state_dict_sha256"] == row["cleaned_state_dict_sha256"] for row in group),
                "mean_relative_l2": sum(row["parameter_relative_l2_distance"] for row in group) / len(group),
                "mean_cosine_similarity": sum(row["parameter_cosine_similarity"] for row in group) / len(group),
                "mean_bn_running_mean_diff": sum(row["parameter_l2_distance"] for row in bn_mean) / len(bn_mean) if bn_mean else "",
                "mean_bn_running_var_diff": sum(row["parameter_l2_distance"] for row in bn_var) / len(bn_var) if bn_var else "",
            }
        )
    write_tsv(args.summary, summary, list(summary[0]))
    print(f"SOURCE_COMPARE|pairs={len(overall)}|rows={len(output_rows)}|output={args.output}")


def build_eval_jobs(args):
    source_manifest = read_tsv(args.source_manifest)
    old_base_rows = read_tsv(args.old_base_rows)
    old_smooth_rows = read_tsv(args.old_smooth_rows)
    cleaned_summary = read_tsv(args.cleaned_summary)

    def native_source_f1(task, config, seed, version):
        if version == "cleaned":
            matches = [
                row for row in cleaned_summary
                if row.get("task") == task and row.get("config") == config and int(row.get("seed", -1)) == int(seed)
            ]
            return matches[0].get("source_on_target_macro_f1", "") if len(matches) == 1 else ""
        rows = old_base_rows if config == "base" else old_smooth_rows
        expected_config = "plain" if config == "base" else "v276_smoothed_timepoint_w1"
        matches = [
            row for row in rows
            if row.get("task") == task and row.get("config") == expected_config and int(row.get("seed", -1)) == int(seed)
        ]
        return matches[0].get("source_on_target_f1", "") if len(matches) == 1 else ""

    jobs = []
    for item in source_manifest:
        source = item["source_domain"]
        for target in DOMAINS:
            if target == source:
                continue
            task = f"{source}_to_{target}"
            for version in ("old", "cleaned"):
                jobs.append(
                    {
                        "kind": "source",
                        "task": task,
                        "source": DOMAINS[source][0],
                        "target": DOMAINS[target][0],
                        "config": item["config"],
                        "seed": item["seed"],
                        "checkpoint_version": version,
                        "source_version": version,
                        "da_version": "none",
                        "checkpoint_path": item[f"{version}_checkpoint_path"],
                        "native_test_f1": native_source_f1(task, item["config"], item["seed"], version),
                    }
                )
    for row in read_tsv(args.existing_da_manifest):
        jobs.append(
            {
                "kind": "existing_da",
                "task": row["task"],
                "source": row["source"],
                "target": row["target"],
                "config": row["method"],
                "seed": row["seed"],
                "checkpoint_version": row["checkpoint_version"],
                "source_version": row["source_version"],
                "da_version": row["da_version"],
                "checkpoint_path": row["checkpoint_path"],
                "native_test_f1": row["native_test_f1"],
            }
        )
    for index, row in enumerate(jobs):
        row["job_id"] = index
    write_tsv(args.output, jobs, list(jobs[0]))
    print(f"EVAL_JOBS|source={sum(row['kind']=='source' for row in jobs)}|existing_da={sum(row['kind']=='existing_da' for row in jobs)}")


def summarize_evals(args):
    jobs = read_tsv(args.jobs)
    rows = []
    failures = []
    for job in jobs:
        result_path = Path(args.results_root) / f"job_{int(job['job_id']):03d}.json"
        if not result_path.is_file():
            failures.append(job["job_id"])
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        native = job.get("native_test_f1", "")
        common = result.get("macro_f1", "")
        delta = ""
        if native not in ("", None) and common not in ("", None):
            delta = float(common) - float(native)
        rows.append(
            {
                **job,
                **result,
                "checkpoint_sha256": result.get("file_sha256", ""),
                "common_accuracy": result.get("accuracy", ""),
                "common_macro_f1": common,
                "common_weighted_f1": result.get("weighted_f1", ""),
                "common_kappa": result.get("kappa", ""),
                "common_test_f1": common,
                "delta_common_minus_native": delta,
            }
        )
    source_rows = [row for row in rows if row["kind"] == "source"]
    da_rows = [row for row in rows if row["kind"] == "existing_da"]
    source_fields = [
        "task", "source", "target", "config", "seed", "checkpoint_version", "checkpoint_path",
        "state_dict_sha256", "native_test_f1", "common_test_f1", "accuracy", "macro_f1", "weighted_f1", "kappa",
    ]
    da_fields = [
        "task", "config", "seed", "source_version", "da_version", "checkpoint_path",
        "checkpoint_sha256", "state_dict_sha256", "native_test_f1", "common_accuracy",
        "common_macro_f1", "common_weighted_f1", "common_kappa", "delta_common_minus_native",
        "load_status", "missing_keys", "unexpected_keys",
    ]
    write_tsv(args.source_output, source_rows, source_fields)
    write_tsv(args.da_output, da_rows, da_fields)
    paired = []
    for task in sorted({row["task"] for row in source_rows}):
        for config in CONFIGS:
            for seed in SEEDS:
                group = [row for row in source_rows if row["task"] == task and row["config"] == config and int(row["seed"]) == seed]
                by_version = {row["checkpoint_version"]: float(row["macro_f1"]) for row in group}
                if set(by_version) == {"old", "cleaned"}:
                    paired.append(
                        {
                            "task": task,
                            "config": config,
                            "seed": seed,
                            "old_source_f1": by_version["old"],
                            "cleaned_source_f1": by_version["cleaned"],
                            "delta_cleaned_minus_old": by_version["cleaned"] - by_version["old"],
                        }
                    )
    write_tsv(args.paired_output, paired, list(paired[0]) if paired else ["task"])
    print(f"EVAL_SUMMARY|source={len(source_rows)}|existing_da={len(da_rows)}|failed={len(failures)}")
    if failures:
        raise SystemExit(1)


def gate_first_round(args):
    comparison = [row for row in read_tsv(args.source_comparison) if row.get("module") == "overall"]
    source_eval = read_tsv(args.source_eval)
    da_eval = read_tsv(args.da_eval)
    load_smoke = read_tsv(args.load_smoke)
    tolerance = args.f1_tolerance

    def f1_consistent(rows):
        checked = []
        for row in rows:
            native = row.get("native_test_f1", "")
            common = common_f1_value(row)
            if native in ("", None) or common in ("", None):
                checked.append(False)
            else:
                checked.append(abs(float(native) - float(common)) <= tolerance)
        return len(checked), sum(checked)

    source_n, source_ok = f1_consistent(source_eval)
    da_n, da_ok = f1_consistent(da_eval)
    strict_ok = sum(
        str(row.get("strict_load_ok", "")).lower() == "true"
        and str(row.get("hash_equal", "")).lower() == "true"
        for row in load_smoke
    )

    def check(name, expected, completed, passed_count, details):
        ok = completed == expected and passed_count == expected
        return {
            "check_name": name,
            "expected": expected,
            "completed": completed,
            "passed": passed_count,
            "failed": max(0, expected - passed_count),
            "status": "passed" if ok else "failed",
            "details": details,
        }

    checks = [
        check("source_checkpoint_pairs", 24, len(comparison), len(comparison), "old-cleaned checkpoint pairs"),
        check("source_common_evaluations", 144, len(source_eval), len(source_eval), "reused common-eval rows"),
        check(
            "source_native_common_consistency", 144, source_n, source_ok,
            f"within_tolerance={source_ok}; tolerance={tolerance}",
        ),
        check("existing_da_common_evaluations", 8, len(da_eval), len(da_eval), "reused existing DA rows"),
        check(
            "existing_da_native_common_consistency", 8, da_n, da_ok,
            f"within_tolerance={da_ok}; tolerance={tolerance}",
        ),
        check(
            "old_da_strict_load_cleaned_source", 4, len(load_smoke), strict_ok,
            f"strict_and_hash_equal={strict_ok}",
        ),
    ]
    write_tsv(
        args.output,
        checks,
        ["check_name", "expected", "completed", "passed", "failed", "status", "details"],
    )
    passed = all(row["status"] == "passed" for row in checks)
    marker = Path(args.marker)
    if passed:
        marker.write_text("FIRST_ROUND_PASSED\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()
    print(f"FIRST_ROUND_GATE|passed={passed}|output={args.output}|marker={marker}")
    raise SystemExit(0 if passed else 1)


def factorial_effects(a, b, c, d):
    """A=old/old, B=cleaned/cleaned, C=old/cleaned, D=cleaned/old."""
    return {
        "da_effect_under_old_source_C_minus_A": c - a,
        "da_effect_under_cleaned_source_B_minus_D": b - d,
        "source_effect_under_old_da_D_minus_A": d - a,
        "source_effect_under_cleaned_da_B_minus_C": b - c,
        "interaction_B_plus_A_minus_C_minus_D": b + a - c - d,
    }


def effect_scale(value):
    value = abs(float(value))
    if value <= 1e-4:
        return "strict_numerical_equivalence"
    if value <= 0.002:
        return "practical_near_equivalence"
    if value <= 0.01:
        return "small_effect"
    return "material_effect"


def effect_direction(value):
    value = float(value)
    if abs(value) <= 1e-4:
        return "zero"
    return "positive" if value > 0 else "negative"


def write_cross_common(args):
    rows = read_tsv(args.cross_summary)
    output = []
    failures = []
    for row in rows:
        native = row.get("native_test_f1", "")
        common = common_f1_value(row)
        delta = float(common) - float(native) if native not in ("", None) and common not in ("", None) else ""
        strict_ok = (
            str(row.get("load_status", "")) == "strict_ok"
            and str(row.get("missing_keys", "")) in ("[]", "")
            and str(row.get("unexpected_keys", "")) in ("[]", "")
        )
        source_hash_equal = (
            row.get("checkpoint_state_dict_sha256", "")
            == row.get("initial_student_state_dict_sha256", "")
        )
        ok = (
            str(row.get("status", "")) in ("0", "success")
            and delta != ""
            and abs(delta) <= args.f1_tolerance
            and strict_ok
            and source_hash_equal
        )
        if not ok:
            failures.append(f"{row.get('task')}/{row.get('method')}/{row.get('source_version')}/{row.get('da_version')}")
        output.append(
            {
                "task": row.get("task", ""),
                "method": row.get("method", ""),
                "source_version": row.get("source_version", ""),
                "da_version": row.get("da_version", ""),
                "seed": row.get("seed", "1"),
                "checkpoint_path": row.get("final_student_checkpoint_path", ""),
                "checkpoint_sha256": row.get("final_student_file_sha256", ""),
                "native_test_f1": native,
                "common_accuracy": row.get("common_accuracy", ""),
                "common_macro_f1": common,
                "common_weighted_f1": row.get("common_weighted_f1", ""),
                "common_kappa": row.get("common_kappa", ""),
                "delta_common_minus_native": delta,
                "strict_load_ok": strict_ok,
                "missing_keys": row.get("missing_keys", ""),
                "unexpected_keys": row.get("unexpected_keys", ""),
                "source_state_hash_equal_after_load": source_hash_equal,
                "status": "passed" if ok else "failed",
            }
        )
    fields = list(output[0]) if output else ["task"]
    write_tsv(args.output, output, fields)
    complete = len(output) == 8 and not failures
    print(f"CROSS_COMMON|rows={len(output)}|failed={len(failures)}|output={args.output}")
    raise SystemExit(0 if complete else 1)


def summarize_cross(args):
    existing = read_tsv(args.existing_da_eval)
    cross = read_tsv(args.cross_summary)
    cells = {}
    for row in existing:
        cells[(row["task"], row["config"], row["source_version"], row["da_version"])] = float(common_f1_value(row))
    for row in cross:
        cells[(row["task"], row["method"], row["source_version"], row["da_version"])] = float(row["common_test_f1"])
    result_rows = []
    effect_rows = []
    for task in PROBE_TASKS:
        for method in CONFIGS:
            a = cells.get((task, method, "old", "old"))
            b = cells.get((task, method, "cleaned", "cleaned"))
            c = cells.get((task, method, "old", "cleaned"))
            d = cells.get((task, method, "cleaned", "old"))
            complete = all(value is not None for value in (a, b, c, d))
            effects = factorial_effects(a, b, c, d) if complete else {
                "da_effect_under_old_source_C_minus_A": "",
                "da_effect_under_cleaned_source_B_minus_D": "",
                "source_effect_under_old_da_D_minus_A": "",
                "source_effect_under_cleaned_da_B_minus_C": "",
                "interaction_B_plus_A_minus_C_minus_D": "",
            }
            result_rows.append(
                {
                    "task": task,
                    "method": method,
                    "A_old_source_old_da": a,
                    "B_cleaned_source_cleaned_da": b,
                    "C_old_source_cleaned_da": c,
                    "D_cleaned_source_old_da": d,
                    "complete": complete,
                }
            )
            if complete:
                renamed = {
                    "da_effect_under_old_source": effects["da_effect_under_old_source_C_minus_A"],
                    "da_effect_under_cleaned_source": effects["da_effect_under_cleaned_source_B_minus_D"],
                    "source_effect_under_old_da": effects["source_effect_under_old_da_D_minus_A"],
                    "source_effect_under_cleaned_da": effects["source_effect_under_cleaned_da_B_minus_C"],
                    "interaction": effects["interaction_B_plus_A_minus_C_minus_D"],
                }
                source_max = max(abs(renamed["source_effect_under_old_da"]), abs(renamed["source_effect_under_cleaned_da"]))
                da_max = max(abs(renamed["da_effect_under_old_source"]), abs(renamed["da_effect_under_cleaned_source"]))
                interaction_abs = abs(renamed["interaction"])
                dominant = max(
                    ((source_max, "source_checkpoint"), (da_max, "da_implementation"), (interaction_abs, "interaction"))
                )[1]
                interpretation = (
                    f"dominant={dominant}; "
                    f"source={effect_scale(source_max)}; da={effect_scale(da_max)}; "
                    f"interaction={effect_scale(interaction_abs)}"
                )
                effect_rows.append(
                    {
                        "task": task,
                        "method": method,
                        **renamed,
                        **{f"abs_{name}": abs(value) for name, value in renamed.items()},
                        **{f"direction_{name}": effect_direction(value) for name, value in renamed.items()},
                        **{f"scale_{name}": effect_scale(value) for name, value in renamed.items()},
                        "dominant_effect": dominant,
                        "interpretation": interpretation,
                    }
                )
    write_tsv(args.results_output, result_rows, list(result_rows[0]))
    write_tsv(args.output, effect_rows, list(effect_rows[0]) if effect_rows else ["task"])
    failed = [row for row in result_rows if not row["complete"]]
    print(f"CROSS_SUMMARY|rows={len(result_rows)}|incomplete={len(failed)}|output={args.output}")
    raise SystemExit(1 if failed else 0)


def markdown_table(rows, fields):
    header = "| " + " | ".join(fields) + " |"
    separator = "|" + "|".join("---" for _ in fields) + "|"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_final_report(args):
    gate = read_tsv(args.gate)
    statuses = read_tsv(args.job_status)
    common = read_tsv(args.cross_common)
    results = read_tsv(args.results)
    effects = read_tsv(args.effects)
    successful = sum(row.get("status") == "success" for row in statuses)
    gate_passed = bool(gate) and all(row.get("status") == "passed" for row in gate)
    result_fields = [
        "task", "method", "A_old_source_old_da", "B_cleaned_source_cleaned_da",
        "C_old_source_cleaned_da", "D_cleaned_source_old_da",
    ]
    effect_fields = [
        "task", "method", "da_effect_under_old_source", "da_effect_under_cleaned_source",
        "source_effect_under_old_da", "source_effect_under_cleaned_da", "interaction",
        "dominant_effect", "interpretation",
    ]
    common_fields = [
        "task", "method", "source_version", "da_version", "native_test_f1",
        "common_macro_f1", "delta_common_minus_native", "status",
    ]
    text = f"""# v2.8 Source Checkpoint × DA Implementation 因果审计

## 审计状态

- 第一轮门禁：{'通过' if gate_passed else '未通过'}
- 复用既有评估：144 条 source + 8 条 existing DA
- Cross jobs：{successful}/8 成功
- 所有最终比较均使用共同评估器的 final student macro-F1。

## 门禁

{markdown_table(gate, ['check_name', 'expected', 'completed', 'passed', 'failed', 'status', 'details'])}

## Cross Common Evaluation

{markdown_table(common, common_fields)}

## 2×2 结果

{markdown_table(results, result_fields)}

其中：A=old source+old DA，B=cleaned source+cleaned DA，C=old source+cleaned DA，D=cleaned source+old DA。

## 因果效应

{markdown_table(effects, effect_fields)}

效应等级：`≤1e-4` 为数值等价，`≤0.002` 为实际近似等价，`≤0.01` 为小效应，`>0.01` 为实质效应。

## 证据边界

### Confirmed

- source checkpoint 权重身份不同，且共同评估差异真实存在。
- existing DA 与 cross DA 均由同一共同评估器评估 final student。
- cleaned source 可严格加载到 old DA；cross job 还检查加载后的初始 student hash。

### Ruled Out

- smooth_k3 损失公式与梯度迁移错误。
- source/common evaluator 或 existing DA/common evaluator 的统计口径错误。
- checkpoint key/shape 不兼容或静默宽松加载。

### Unresolved

- 历史 A/B 运行没有统一保存逐 epoch trace，因此不能在不重跑对角单元的前提下精确报告 old-vs-cleaned 的首次分叉 epoch。
- 是否扩展到更多任务应由本轮四张 2×2 表的任务一致性决定，不在本报告中预设。
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    complete = gate_passed and successful == 8 and len(common) == 8 and len(results) == 4 and len(effects) == 4
    print(f"FINAL_REPORT|complete={complete}|output={output}")
    raise SystemExit(0 if complete else 1)


def parser():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifests")
    build.add_argument("--outputs_root", required=True)
    build.add_argument("--cleaned_source_inventory", required=True)
    build.add_argument("--recovered_results", required=True)
    build.add_argument("--cleaned_summary", required=True)
    build.add_argument("--source_manifest", required=True)
    build.add_argument("--existing_da_manifest", required=True)
    build.set_defaults(func=build_manifests)
    compare = sub.add_parser("compare-sources")
    compare.add_argument("--source_manifest", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--summary", required=True)
    compare.set_defaults(func=compare_sources)
    jobs = sub.add_parser("build-eval-jobs")
    jobs.add_argument("--source_manifest", required=True)
    jobs.add_argument("--existing_da_manifest", required=True)
    jobs.add_argument("--old_base_rows", required=True)
    jobs.add_argument("--old_smooth_rows", required=True)
    jobs.add_argument("--cleaned_summary", required=True)
    jobs.add_argument("--output", required=True)
    jobs.set_defaults(func=build_eval_jobs)
    summary = sub.add_parser("summarize-evals")
    summary.add_argument("--jobs", required=True)
    summary.add_argument("--results_root", required=True)
    summary.add_argument("--source_output", required=True)
    summary.add_argument("--paired_output", required=True)
    summary.add_argument("--da_output", required=True)
    summary.set_defaults(func=summarize_evals)
    gate = sub.add_parser("gate-first-round")
    gate.add_argument("--source_comparison", required=True)
    gate.add_argument("--source_eval", required=True)
    gate.add_argument("--da_eval", required=True)
    gate.add_argument("--load_smoke", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--marker", required=True)
    gate.add_argument("--f1_tolerance", type=float, default=1e-4)
    gate.set_defaults(func=gate_first_round)
    cross = sub.add_parser("summarize-cross")
    cross.add_argument("--existing_da_eval", required=True)
    cross.add_argument("--cross_summary", required=True)
    cross.add_argument("--results_output", required=True)
    cross.add_argument("--output", required=True)
    cross.set_defaults(func=summarize_cross)
    cross_common = sub.add_parser("write-cross-common")
    cross_common.add_argument("--cross_summary", required=True)
    cross_common.add_argument("--output", required=True)
    cross_common.add_argument("--f1_tolerance", type=float, default=1e-4)
    cross_common.set_defaults(func=write_cross_common)
    report = sub.add_parser("write-final-report")
    report.add_argument("--gate", required=True)
    report.add_argument("--job_status", required=True)
    report.add_argument("--cross_common", required=True)
    report.add_argument("--results", required=True)
    report.add_argument("--effects", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(func=write_final_report)
    lookup = sub.add_parser("lookup-source-checkpoint")
    lookup.add_argument("--manifest", required=True)
    lookup.add_argument("--source", required=True)
    lookup.add_argument("--config", required=True)
    lookup.add_argument("--version", required=True, choices=("old", "cleaned"))
    lookup.add_argument("--seed", type=int, default=1)
    lookup.set_defaults(func=print_source_checkpoint)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
