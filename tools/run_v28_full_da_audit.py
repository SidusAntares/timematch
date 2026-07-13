import argparse
import csv
import json
import os
from pathlib import Path
import runpy
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v28_audit_utils import model_sha256


def scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return float(value.detach().item())
    return float(value)


def mean(values):
    return sum(values) / len(values) if values else None


def enclosing_train_locals(frame):
    current = frame.f_back
    while current is not None:
        if current.f_code.co_name == "train_timematch":
            return current.f_locals
        current = current.f_back
    return {}


class FullDATrace:
    def __init__(self, implementation):
        self.implementation = implementation
        self.epochs = {}
        self.seen_steps = set()
        self.shift_by_epoch = {}
        self.current_validation = {}
        self.initial_student_hash = ""
        self.initial_teacher_hash = ""
        self.final_student_hash = ""
        self.final_teacher_hash = ""

    def epoch_row(self, epoch):
        return self.epochs.setdefault(
            int(epoch),
            {
                "epoch": int(epoch) + 1,
                "source_losses": [],
                "target_losses": [],
                "total_losses": [],
                "pseudo_confidences": [],
                "pseudo_ratios": [],
                "global_shift": None,
                "best_shift_score": None,
                "second_shift_score": None,
                "shift_margin": None,
                "val_macro_f1": None,
            },
        )

    traced_functions = {
        "train_timematch",
        "estimate_temporal_shift",
        "estimate_temporal_shift_details",
        "validation",
    }

    def global_trace(self, frame, event, arg):
        if event == "call" and frame.f_code.co_name in self.traced_functions:
            return self.trace
        return None

    def trace(self, frame, event, arg):
        name = frame.f_code.co_name
        if name in {"estimate_temporal_shift", "estimate_temporal_shift_details"} and event == "return":
            if name == "estimate_temporal_shift_details" and isinstance(arg, dict):
                self.capture_shift_details(frame, arg)
            elif name == "estimate_temporal_shift":
                self.capture_shift(frame, arg)
            return self.trace
        if name == "validation" and event == "line":
            local = frame.f_locals
            if "val_f1" in local and "epoch" in local:
                self.epoch_row(local["epoch"])["val_macro_f1"] = float(local["val_f1"])
            return self.trace
        if name != "train_timematch":
            return self.trace
        local = frame.f_locals
        if event == "call":
            self.initial_student_hash = model_sha256(local["student"])
            return self.trace
        if event == "line":
            if "teacher" in local and not self.initial_teacher_hash:
                self.initial_student_hash = model_sha256(local["student"])
                self.initial_teacher_hash = model_sha256(local["teacher"])
            epoch = local.get("epoch")
            step = local.get("step")
            if epoch is not None:
                row = self.epoch_row(epoch)
                row["global_shift"] = int(local.get("target_to_source_shift", 0))
                shift = self.shift_by_epoch.get(int(epoch))
                if shift:
                    row.update(shift)
            key = (epoch, step)
            loss_source = local.get("loss_source")
            loss_target = local.get("loss_target")
            loss = local.get("loss")
            pseudo_conf = local.get("pseudo_conf")
            pseudo_mask = local.get("pseudo_mask")
            if (
                epoch is not None
                and step is not None
                and key not in self.seen_steps
                and loss_source is not None
                and loss is not None
                and pseudo_conf is not None
                and pseudo_mask is not None
            ):
                self.seen_steps.add(key)
                row = self.epoch_row(epoch)
                row["source_losses"].append(scalar(loss_source))
                row["target_losses"].append(scalar(loss_target) if loss_target is not None else 0.0)
                row["total_losses"].append(scalar(loss))
                row["pseudo_confidences"].append(float(pseudo_conf.detach().float().mean().item()))
                row["pseudo_ratios"].append(float(pseudo_mask.detach().float().mean().item()))
            return self.trace
        if event == "return":
            self.final_student_hash = model_sha256(local["student"])
            self.final_teacher_hash = model_sha256(local["teacher"])
        return self.trace

    def capture_shift(self, frame, selected_shift):
        local = frame.f_locals
        parent = enclosing_train_locals(frame)
        epoch = parent.get("epoch")
        if epoch is None:
            return
        estimator = local.get("shift_estimator")
        scores = None
        maximize = False
        if estimator == "IS" and "inception_score" in local:
            scores, maximize = local["inception_score"], True
        elif estimator == "ENT" and "entropy_score" in local:
            scores = local["entropy_score"]
        elif estimator == "AM" and "am" in local:
            scores = local["am"]
        elif estimator == "ACC" and "shift_acc_scores" in local:
            scores, maximize = local["shift_acc_scores"], True
        best = second = margin = None
        if scores is not None:
            values = [float(value) for value in scores]
            ranked = sorted(values, reverse=maximize)
            if ranked:
                best = ranked[0]
            if len(ranked) > 1:
                second = ranked[1]
                margin = (best - second) if maximize else (second - best)
        self.shift_by_epoch[int(epoch)] = {
            "global_shift": int(selected_shift),
            "best_shift_score": best,
            "second_shift_score": second,
            "shift_margin": margin,
        }

    def capture_shift_details(self, frame, details):
        parent = enclosing_train_locals(frame)
        epoch = parent.get("epoch")
        if epoch is None:
            return
        estimator = str(frame.f_locals.get("shift_estimator", "IS")).upper()
        score_key = {
            "IS": "is_scores",
            "ENT": "entropy_scores",
            "AM": "am_scores",
            "ACC": "acc_scores",
            "F1": "f1_scores",
        }.get(estimator)
        maximize = estimator in {"IS", "ACC", "F1"}
        values = [float(value) for value in details.get(score_key, [])] if score_key else []
        ranked = sorted(values, reverse=maximize)
        best = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        margin = None
        if best is not None and second is not None:
            margin = best - second if maximize else second - best
        self.shift_by_epoch[int(epoch)] = {
            "global_shift": int(details["best_shift"]),
            "best_shift_score": best,
            "second_shift_score": second,
            "shift_margin": margin,
        }

    def rows(self):
        output = []
        for epoch in sorted(self.epochs):
            row = dict(self.epochs[epoch])
            row["source_loss"] = mean(row.pop("source_losses"))
            row["target_loss"] = mean(row.pop("target_losses"))
            row["total_loss"] = mean(row.pop("total_losses"))
            row["pseudo_confidence"] = mean(row.pop("pseudo_confidences"))
            row["pseudo_ratio"] = mean(row.pop("pseudo_ratios"))
            output.append(row)
        return output


def write_tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "global_shift",
        "best_shift_score",
        "second_shift_score",
        "shift_margin",
        "pseudo_confidence",
        "pseudo_ratio",
        "source_loss",
        "target_loss",
        "total_loss",
        "val_macro_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run one full old/cleaned TimeMatch DA causal-audit job.")
    parser.add_argument("--implementation", required=True, choices=("old", "cleaned"))
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--diag_output", required=True)
    parser.add_argument("--run_summary", required=True)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    repo_root = Path(args.repo_root).resolve()
    Path(args.diag_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.run_summary).parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    old_argv = sys.argv
    old_cwd = Path.cwd()
    trace = FullDATrace(args.implementation)
    started = time.time()
    status = 0
    error = ""
    try:
        os.chdir(repo_root)
        sys.argv = [str(repo_root / "train.py"), *train_args]
        sys.settrace(trace.global_trace)
        runpy.run_path(str(repo_root / "train.py"), run_name="__main__")
    except SystemExit as exc:
        status = int(exc.code or 0)
        if status:
            error = f"SystemExit({status})"
    except Exception as exc:
        status = 1
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.settrace(None)
        sys.argv = old_argv
        os.chdir(old_cwd)
        write_tsv(args.diag_output, trace.rows())
        summary = {
            "implementation": args.implementation,
            "status": status,
            "runtime_s": time.time() - started,
            "initial_student_state_hash": trace.initial_student_hash,
            "initial_teacher_state_hash": trace.initial_teacher_hash,
            "final_student_state_hash": trace.final_student_hash,
            "final_teacher_state_hash": trace.final_teacher_hash,
            "epochs_recorded": len(trace.rows()),
            "error": error,
        }
        Path(args.run_summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"FULL_DA_AUDIT|implementation={args.implementation}|status={status}|"
        f"epochs={len(trace.rows())}|student={trace.final_student_hash}|teacher={trace.final_teacher_hash}"
    )
    raise SystemExit(status)


if __name__ == "__main__":
    main()
