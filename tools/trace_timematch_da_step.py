import argparse
import csv
import hashlib
import json
import linecache
from pathlib import Path
import random
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v28_audit_utils import checkpoint_hashes, model_sha256, tensor_sha256


TRACE_FIELDS = [
    "implementation",
    "step",
    "source_checkpoint_file_sha256",
    "source_checkpoint_state_dict_sha256",
    "source_batch_hash",
    "target_weak_batch_hash",
    "target_strong_batch_hash",
    "student_state_hash_before",
    "teacher_state_hash_before",
    "optimizer_lr",
    "scheduler_state",
    "source_logits_mean",
    "source_logits_std",
    "source_logits_hash",
    "teacher_target_logits_mean",
    "teacher_target_logits_std",
    "teacher_target_logits_hash",
    "pseudo_softmax_mean",
    "pseudo_confidence_mean",
    "pseudo_ratio",
    "pseudo_label_distribution",
    "pseudo_label_hash",
    "pseudo_mask_hash",
    "global_shift",
    "shift_score_best",
    "shift_score_second",
    "shift_score_margin",
    "shift_score_top5",
    "target_positions_before_min",
    "target_positions_before_max",
    "target_positions_after_min",
    "target_positions_after_max",
    "target_positions_after_hash",
    "student_target_logits_mean",
    "student_target_logits_std",
    "student_target_logits_hash",
    "source_loss",
    "target_loss",
    "total_loss",
    "student_grad_norm",
    "student_state_hash_after",
    "teacher_state_hash_after",
]


def clone_cpu(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    return value


def nested_tensor_hash(value):
    import torch

    digest = hashlib.sha256()

    def visit(item, prefix=""):
        if torch.is_tensor(item):
            digest.update(prefix.encode("utf-8"))
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(item[key], f"{prefix}/{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}/{index}")

    visit(value)
    return digest.hexdigest()


class ReplayDataset:
    def __init__(self, labels):
        self._labels = labels

    def get_labels(self):
        return self._labels


class ReplayLoader:
    def __init__(self, batches, labels=None):
        self.batches = batches
        self.dataset = ReplayDataset(labels) if labels is not None else None

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def take_batches(loader, count):
    iterator = iter(loader)
    return [clone_cpu(next(iterator)) for _ in range(count)]


def replay_loaders(payload):
    return (
        ReplayLoader(payload["source_batches"]),
        ReplayLoader(payload["target_no_aug_batches"], payload["target_labels"]),
        ReplayLoader(payload["target_train_batches"]),
    )


def capture_rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def write_rows(path, rows, fields=TRACE_FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=True, sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field, "")
                    for field in fields
                }
            )


def scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return float(value.detach().item())
    return float(value)


def tensor_summary(value):
    if value is None:
        return {"mean": None, "std": None, "hash": ""}
    detached = value.detach()
    return {
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "hash": tensor_sha256(detached),
    }


def gradient_norm(model):
    import torch

    total = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        term = parameter.grad.detach().pow(2).sum()
        total = term if total is None else total + term
    return float(torch.sqrt(total).item()) if total is not None else 0.0


class TrainTraceCollector:
    def __init__(self, implementation, checkpoint_info):
        self.implementation = implementation
        self.checkpoint_info = checkpoint_info
        self.pending = {}
        self.rows = []
        self.shift_info = {}

    def trace(self, frame, event, arg):
        name = frame.f_code.co_name
        if name in {"estimate_temporal_shift", "estimate_temporal_shift_details"} and event == "return":
            self._capture_shift(frame, arg)
            return self.trace
        if name != "train_timematch":
            return self.trace
        if event != "line":
            return self.trace

        source_line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
        local = frame.f_locals
        step = int(local.get("step", -1))
        if "pixels_t_weak, mask_t_weak" in source_line and step >= 0:
            student = local.get("student")
            teacher = local.get("teacher")
            self.pending[step] = {
                "student_state_hash_before": model_sha256(student),
                "teacher_state_hash_before": model_sha256(teacher),
                "source_batch_hash": nested_tensor_hash(local.get("sample_source")),
                "target_weak_batch_hash": nested_tensor_hash(local.get("sample_target_weak")),
                "target_strong_batch_hash": nested_tensor_hash(local.get("sample_target_strong")),
            }
        elif "optimizer.zero_grad()" in source_line and step >= 0:
            self._capture_pre_update(step, local)
        elif "global_step += 1" in source_line and step >= 0:
            self._capture_post_update(step, local)
        return self.trace

    def _capture_shift(self, frame, returned):
        local = frame.f_locals
        if isinstance(returned, dict):
            estimator = str(local.get("estimator", local.get("shift_estimator", "IS"))).upper()
            key = {"IS": "is_scores", "AM": "am_scores", "ENT": "entropy_scores", "ACC": "acc_scores", "F1": "f1_scores"}.get(estimator)
            scores = returned.get(key) if key else None
            indices = returned.get("topk_indices", [])
            if scores is not None and indices:
                values = [float(scores[index]) for index in indices[:5]]
            else:
                values = []
            best = values[0] if values else None
            second = values[1] if len(values) > 1 else None
            self.shift_info = {
                "global_shift": int(returned.get("best_shift", 0)),
                "shift_score_best": best,
                "shift_score_second": second,
                "shift_score_margin": abs(best - second) if best is not None and second is not None else None,
                "shift_score_top5": values,
            }
            return

        estimator = str(local.get("shift_estimator", "IS")).upper()
        if estimator == "IS":
            scores = local.get("inception_score")
            descending = True
        elif estimator == "AM":
            scores = local.get("am")
            descending = False
        elif estimator == "ENT":
            scores = local.get("entropy_score")
            descending = False
        else:
            scores = local.get("shift_acc_scores")
            descending = True
        if scores is None:
            return
        import numpy as np

        values_array = np.asarray(scores, dtype=float)
        ranked = np.argsort(values_array)
        if descending:
            ranked = ranked[::-1]
        values = [float(values_array[index]) for index in ranked[:5]]
        best = values[0] if values else None
        second = values[1] if len(values) > 1 else None
        self.shift_info = {
            "global_shift": int(returned),
            "shift_score_best": best,
            "shift_score_second": second,
            "shift_score_margin": abs(best - second) if best is not None and second is not None else None,
            "shift_score_top5": values,
        }

    def _capture_pre_update(self, step, local):
        import torch

        row = self.pending.setdefault(step, {})
        source = tensor_summary(local.get("logits_source"))
        target = tensor_summary(local.get("logits_target"))
        teacher_logits = local.get("teacher_logits")
        if teacher_logits is None:
            with torch.no_grad():
                teacher_logits = local["teacher"].forward(
                    local["pixels_t_weak"],
                    local["mask_t_weak"],
                    local["position_t_weak"] + int(local.get("target_to_source_shift", 0)),
                    local["extra_t_weak"],
                )
        teacher = tensor_summary(teacher_logits)
        pseudo_probs = local.get("teacher_preds")
        pseudo_conf = local.get("pseudo_conf")
        pseudo_targets = local.get("pseudo_targets")
        pseudo_mask = local.get("pseudo_mask")
        positions = local.get("position_t_weak")
        shift = int(local.get("target_to_source_shift", self.shift_info.get("global_shift", 0)))
        optimizer = local.get("optimizer")
        scheduler = local.get("scheduler")
        counts = []
        if pseudo_targets is not None:
            num_classes = int(local.get("config").num_classes)
            counts = torch.bincount(pseudo_targets.detach().cpu(), minlength=num_classes).tolist()
        row.update(
            {
                "implementation": self.implementation,
                "step": step,
                "source_checkpoint_file_sha256": self.checkpoint_info["file_sha256"],
                "source_checkpoint_state_dict_sha256": self.checkpoint_info["state_dict_sha256"],
                "optimizer_lr": float(optimizer.param_groups[0]["lr"]),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else {},
                "source_logits_mean": source["mean"],
                "source_logits_std": source["std"],
                "source_logits_hash": source["hash"],
                "teacher_target_logits_mean": teacher["mean"],
                "teacher_target_logits_std": teacher["std"],
                "teacher_target_logits_hash": teacher["hash"],
                "pseudo_softmax_mean": float(pseudo_probs.detach().mean().item()),
                "pseudo_confidence_mean": float(pseudo_conf.detach().mean().item()),
                "pseudo_ratio": float(pseudo_mask.float().mean().item()),
                "pseudo_label_distribution": counts,
                "pseudo_label_hash": tensor_sha256(pseudo_targets),
                "pseudo_mask_hash": tensor_sha256(pseudo_mask),
                "global_shift": shift,
                **self.shift_info,
                "target_positions_before_min": float(positions.min().item()),
                "target_positions_before_max": float(positions.max().item()),
                "target_positions_after_min": float((positions + shift).min().item()),
                "target_positions_after_max": float((positions + shift).max().item()),
                "target_positions_after_hash": tensor_sha256(positions + shift),
                "student_target_logits_mean": target["mean"],
                "student_target_logits_std": target["std"],
                "student_target_logits_hash": target["hash"],
                "source_loss": scalar(local.get("loss_source")),
                "target_loss": scalar(local.get("loss_target")),
                "total_loss": scalar(local.get("loss")),
            }
        )

    def _capture_post_update(self, step, local):
        row = self.pending.setdefault(step, {})
        row.update(
            {
                "student_grad_norm": gradient_norm(local["student"]),
                "student_state_hash_after": model_sha256(local["student"]),
                "teacher_state_hash_after": model_sha256(local["teacher"]),
            }
        )
        self.rows.append(row)


def parse_weights_path(train_args):
    if "--weights" not in train_args:
        raise ValueError("forwarded train.py arguments must include --weights")
    root = Path(train_args[train_args.index("--weights") + 1])
    checkpoint = root / "fold_0" / "model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def run_trace(args, train_args):
    import importlib
    import torch

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    timematch = importlib.import_module("timematch")
    train_function = timematch.train_timematch
    implementation_module = importlib.import_module(train_function.__module__)
    original_get_data_loaders = implementation_module.get_data_loaders

    if args.mode == "replay":
        payload = torch.load(args.trace_inputs, map_location="cpu", weights_only=False)

        def audited_get_data_loaders(*_unused, **_unused_kwargs):
            restore_rng_state(payload["rng_state_after_capture"])
            return replay_loaders(payload)
    else:
        payload_holder = {}

        def audited_get_data_loaders(splits, config, balance_source=True):
            loaders = original_get_data_loaders(splits, config, balance_source)
            source_loader, target_no_aug_loader, target_train_loader = loaders
            payload = {
                "source_batches": take_batches(source_loader, args.steps),
                "target_no_aug_batches": take_batches(target_no_aug_loader, args.shift_batches),
                "target_train_batches": take_batches(target_train_loader, args.steps),
                "target_labels": clone_cpu(target_no_aug_loader.dataset.get_labels()),
            }
            payload["source_batch_hashes"] = [nested_tensor_hash(batch) for batch in payload["source_batches"]]
            payload["target_train_batch_hashes"] = [nested_tensor_hash(batch) for batch in payload["target_train_batches"]]
            payload["rng_state_after_capture"] = capture_rng_state()
            payload_holder.update(payload)
            Path(args.trace_inputs).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.trace_inputs)
            return replay_loaders(payload)

    implementation_module.get_data_loaders = audited_get_data_loaders
    checkpoint_info = checkpoint_hashes(parse_weights_path(train_args))
    collector = TrainTraceCollector(args.implementation, checkpoint_info)
    old_argv = sys.argv[:]
    old_trace = sys.gettrace()
    try:
        sys.argv = [str(repo_root / "train.py"), *train_args]
        sys.settrace(collector.trace)
        runpy.run_path(str(repo_root / "train.py"), run_name="__main__")
    finally:
        sys.settrace(old_trace)
        sys.argv = old_argv
        implementation_module.get_data_loaders = original_get_data_loaders
    write_rows(args.trace_output, collector.rows)
    if len(collector.rows) != args.steps:
        raise RuntimeError(f"expected {args.steps} trace rows, got {len(collector.rows)}")
    print(f"TRACE_OK|implementation={args.implementation}|steps={len(collector.rows)}|output={args.trace_output}")


def classify_field(field):
    if "shift" in field:
        return "shift_scoring_difference"
    if "pseudo" in field or "teacher_target" in field:
        return "pseudo_label_difference"
    if "loss" in field or "logits" in field:
        return "loss_difference"
    if "grad" in field or "student_state" in field or "scheduler" in field or "optimizer" in field:
        return "optimizer_difference"
    if "teacher_state" in field:
        return "EMA_difference"
    return "floating_point_only"


def compare_traces(old_path, cleaned_path, output):
    def load(path):
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    old_rows = load(old_path)
    cleaned_rows = load(cleaned_path)
    if len(old_rows) != len(cleaned_rows):
        raise ValueError(f"trace row count differs: old={len(old_rows)}, cleaned={len(cleaned_rows)}")
    diff_fields = ["field", "old_value", "cleaned_value", "abs_diff", "first_divergent_step", "classification"]
    differences = []
    for field in TRACE_FIELDS:
        if field in {"implementation"}:
            continue
        first = None
        old_value = cleaned_value = ""
        abs_diff = ""
        for old_row, cleaned_row in zip(old_rows, cleaned_rows):
            old_value = old_row.get(field, "")
            cleaned_value = cleaned_row.get(field, "")
            if old_value == cleaned_value:
                continue
            try:
                abs_diff = abs(float(old_value) - float(cleaned_value))
                if abs_diff <= 1e-10:
                    continue
            except (TypeError, ValueError):
                abs_diff = ""
            first = old_row.get("step", "")
            break
        differences.append(
            {
                "field": field,
                "old_value": old_value if first is not None else "",
                "cleaned_value": cleaned_value if first is not None else "",
                "abs_diff": abs_diff if first is not None else 0.0,
                "first_divergent_step": first if first is not None else "",
                "classification": classify_field(field) if first is not None else "identical",
            }
        )
    write_rows(output, differences, diff_fields)
    print(f"TRACE_DIFF_OK|rows={len(differences)}|output={output}")


def main():
    argv = sys.argv[1:]
    train_args = []
    if "--" in argv:
        split = argv.index("--")
        train_args = argv[split + 1 :]
        argv = argv[:split]
    parser = argparse.ArgumentParser(description="Capture/replay and compare deterministic TimeMatch DA steps.")
    parser.add_argument("--mode", choices=["capture", "replay", "compare"], required=True)
    parser.add_argument("--implementation", choices=["old", "cleaned"], default="cleaned")
    parser.add_argument("--repo_root", default=str(ROOT))
    parser.add_argument("--trace_inputs", default="")
    parser.add_argument("--trace_output", default="")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--shift_batches", type=int, default=3)
    parser.add_argument("--old_trace", default="")
    parser.add_argument("--cleaned_trace", default="")
    args = parser.parse_args(argv)

    if args.mode == "compare":
        compare_traces(args.old_trace, args.cleaned_trace, args.trace_output)
        return
    if not args.trace_inputs or not args.trace_output or not train_args:
        parser.error("capture/replay requires --trace_inputs, --trace_output and forwarded train.py args after --")
    run_trace(args, train_args)


if __name__ == "__main__":
    main()
