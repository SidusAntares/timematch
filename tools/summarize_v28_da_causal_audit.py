import argparse
import csv
import json
from pathlib import Path
import re


VAL_RE = re.compile(r"Validation result:.*?f1=([0-9.]+)")
TEST_RE = re.compile(r"Test result for [^:]+:\s*accuracy=[-+0-9.eE]+,\s*f1=([-+0-9.eE]+)")


def read_tsv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_log(path):
    path = Path(path)
    if not path.is_file():
        return {"native_val_f1": "", "native_test_f1": ""}
    text = path.read_text(encoding="utf-8", errors="replace")
    val = [float(value) for value in VAL_RE.findall(text)]
    test = [float(value) for value in TEST_RE.findall(text)]
    return {
        "native_val_f1": max(val) if val else "",
        "native_test_f1": test[-1] if test else "",
    }


def parse_diag(path):
    rows = read_tsv(path)
    if not rows:
        return {}
    shifts = [row.get("estimated_shift_t_to_s", row.get("global_shift", "")) for row in rows]
    shifts = [value for value in shifts if value != ""]
    margins = []
    for row in rows:
        for name in ("shift_score_margin", "am_top1_top2_margin", "is_top1_top2_margin"):
            value = row.get(name, "")
            if value not in ("", None):
                margins.append(float(value))
                break
    return {
        "initial_shift": shifts[0] if shifts else "",
        "final_shift": shifts[-1] if shifts else "",
        "number_of_shift_changes": sum(a != b for a, b in zip(shifts, shifts[1:])),
        "mean_shift_margin": sum(margins) / len(margins) if margins else "",
        "pseudo_confidence": rows[-1].get(
            "teacher_pseudo_confidence_mean", rows[-1].get("pseudo_confidence", "")
        ),
        "pseudo_ratio": rows[-1].get(
            "teacher_pseudo_coverage", rows[-1].get("pseudo_ratio", "")
        ),
        "target_loss": rows[-1].get("target_loss", ""),
        "source_loss": rows[-1].get("source_loss", ""),
        "total_loss": rows[-1].get("total_loss", ""),
        "final_val_macro_f1": rows[-1].get("val_macro_f1", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize v2.8 DA causal audit jobs.")
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.log_root)
    statuses = read_tsv(root / "job_status.tsv")
    fields = [
        "job_id",
        "phase",
        "task",
        "method",
        "variant",
        "da_impl",
        "source_version",
        "da_version",
        "seed",
        "gpu",
        "pid",
        "epsilon",
        "status",
        "return_code",
        "start_time",
        "end_time",
        "runtime_s",
        "command",
        "git_commit",
        "python_version",
        "torch_version",
        "cuda_version",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "checkpoint_state_dict_sha256",
        "initial_student_state_dict_sha256",
        "initial_shift",
        "final_shift",
        "number_of_shift_changes",
        "mean_shift_margin",
        "pseudo_confidence",
        "pseudo_ratio",
        "target_loss",
        "source_loss",
        "total_loss",
        "final_val_macro_f1",
        "native_val_f1",
        "native_test_f1",
        "common_accuracy",
        "common_macro_f1",
        "common_weighted_f1",
        "common_kappa",
        "delta_common_minus_native",
        "load_status",
        "missing_keys",
        "unexpected_keys",
        "common_test_f1",
        "final_student_checkpoint_path",
        "final_student_file_sha256",
        "final_student_state_dict_sha256",
        "final_teacher_state_dict_sha256",
        "log_path",
    ]
    output_rows = []
    for status in statuses:
        log_path = Path(status.get("log_path", ""))
        row = {**status, **parse_log(log_path)}
        diag_path = status.get("diag_path", "") or str(log_path.parent / "timematch_diag.tsv")
        row.update(parse_diag(diag_path))
        common_path = status.get("common_eval_path", "")
        if common_path and Path(common_path).is_file():
            common = json.loads(Path(common_path).read_text(encoding="utf-8"))
            row["common_test_f1"] = common.get("macro_f1", "")
            row["common_accuracy"] = common.get("accuracy", "")
            row["common_macro_f1"] = common.get("macro_f1", "")
            row["common_weighted_f1"] = common.get("weighted_f1", "")
            row["common_kappa"] = common.get("kappa", "")
            row["load_status"] = common.get("load_status", "")
            row["missing_keys"] = common.get("missing_keys", "")
            row["unexpected_keys"] = common.get("unexpected_keys", "")
            if row.get("native_test_f1", "") not in ("", None):
                row["delta_common_minus_native"] = (
                    float(row["common_macro_f1"]) - float(row["native_test_f1"])
                )
            row["final_student_file_sha256"] = common.get("file_sha256", "")
            row["final_student_state_dict_sha256"] = common.get("state_dict_sha256", "")
        run_summary_path = status.get("run_summary_path", "")
        if run_summary_path and Path(run_summary_path).is_file():
            run_summary = json.loads(Path(run_summary_path).read_text(encoding="utf-8"))
            row["initial_student_state_dict_sha256"] = run_summary.get(
                "initial_student_state_hash", ""
            )
            row["final_student_state_dict_sha256"] = run_summary.get(
                "final_student_state_hash", row.get("final_student_state_dict_sha256", "")
            )
            row["final_teacher_state_dict_sha256"] = run_summary.get(
                "final_teacher_state_hash", ""
            )
        output_rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    failed = [row for row in output_rows if str(row.get("status")) not in ("0", "success")]
    print(f"SUMMARY|jobs={len(output_rows)}|failed={len(failed)}|output={output}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
