import argparse
import csv
import hashlib
import os
from pathlib import Path
import runpy
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v28_audit_utils import file_sha256, model_sha256


def nested_hash(value):
    import torch

    digest = hashlib.sha256()

    def visit(item, prefix=""):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(prefix.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(item[key], f"{prefix}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}/{index}")
        else:
            digest.update(f"{prefix}:{item!r}".encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def scalar(value):
    if value is None:
        return ""
    if hasattr(value, "detach"):
        return float(value.detach().item())
    return float(value)


class SourceTrace:
    function_names = {
        "train_supervised",
        "train_supervised_source_phase_compactness",
    }

    def __init__(self):
        self.row = {
            "initial_state_hash": "",
            "first_batch_hash": "",
            "first_step_classification_loss": "",
            "first_step_structure_loss": "",
            "first_step_total_loss": "",
            "final_state_dict_hash": "",
        }

    def global_trace(self, frame, event, arg):
        if event == "call" and frame.f_code.co_name in self.function_names:
            return self.trace
        return None

    def trace(self, frame, event, arg):
        local = frame.f_locals
        if event == "call":
            self.row["initial_state_hash"] = model_sha256(local["model"])
            return self.trace
        if event == "line":
            if local.get("step") == 0 and "sample" in local:
                if not self.row["first_batch_hash"]:
                    self.row["first_batch_hash"] = nested_hash(local["sample"])
                classification = local.get("cls_loss_raw", local.get("loss"))
                structure = local.get("compact_loss")
                total = local.get("loss")
                if classification is not None:
                    self.row["first_step_classification_loss"] = scalar(classification)
                if structure is not None:
                    self.row["first_step_structure_loss"] = scalar(structure)
                elif frame.f_code.co_name == "train_supervised":
                    self.row["first_step_structure_loss"] = 0.0
                if total is not None:
                    self.row["first_step_total_loss"] = scalar(total)
            return self.trace
        if event == "return":
            self.row["final_state_dict_hash"] = model_sha256(local["model"])
        return self.trace


def main():
    parser = argparse.ArgumentParser(description="Trace one deterministic v2.8 source smoke run.")
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", required=True, choices=("base", "smooth_k3"))
    parser.add_argument("--output", required=True)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    old_argv = sys.argv
    old_cwd = Path.cwd()
    trace = SourceTrace()
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

    checkpoint = ""
    if "--output_dir" in train_args and "-e" in train_args:
        output_dir = Path(train_args[train_args.index("--output_dir") + 1])
        experiment = train_args[train_args.index("-e") + 1]
        checkpoint = output_dir / experiment / "fold_0" / "model.pt"
    row = {
        "task": args.task,
        "config": args.config,
        "status": status,
        "runtime_s": time.time() - started,
        **trace.row,
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_hash": file_sha256(checkpoint) if checkpoint and checkpoint.is_file() else "",
        "error": error,
    }
    fields = list(row)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(
        f"SOURCE_SMOKE_RESULT|task={args.task}|config={args.config}|status={status}|"
        f"initial={row['initial_state_hash']}|final={row['final_state_dict_hash']}|"
        f"checkpoint={row['checkpoint_file_hash']}"
    )
    raise SystemExit(status)


if __name__ == "__main__":
    main()
