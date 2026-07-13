import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from methods.source_structure.losses import compute_source_raw_global_compactness_loss


VERSION = "v276_raw_smoothed_timepoint_compactness"


def load_old_module(commit, old_source_file=""):
    if old_source_file:
        source_path = Path(old_source_file)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = source_path.read_text(encoding="utf-8")
        source_name = str(source_path)
    else:
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit}:ideas/source_raw_compactness.py"],
            text=True,
            encoding="utf-8",
        )
        source_name = "ideas/source_raw_compactness.py"
    module = types.ModuleType("v28_old_source_raw_compactness")
    exec(compile(source, source_name, "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description="Compare old and cleaned smooth_k3 loss and gradients.")
    parser.add_argument(
        "--old_commit",
        default="89d9df4e52744cb955168b0d203a2ddd61c3199e",
    )
    parser.add_argument(
        "--old_source_file",
        default="",
        help="path to exported old ideas/source_raw_compactness.py; avoids requiring .git on server",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    old_module = load_old_module(args.old_commit, args.old_source_file)
    torch.manual_seed(args.seed)
    old_features = torch.randn(10, 7, 5, dtype=torch.float32, requires_grad=True)
    new_features = old_features.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    common = {
        "version": VERSION,
        "intra_trade_off": 1.0,
        "compact_distance": "mse",
        "norm_preserve_trade_off": 0.0,
    }

    old_loss, _ = old_module.compute_source_raw_global_compactness_loss(
        old_features,
        labels,
        **common,
    )
    new_loss, _ = compute_source_raw_global_compactness_loss(
        new_features,
        labels,
        time_smooth_kernel_size=3,
        **common,
    )
    old_loss.backward()
    new_loss.backward()

    result = {
        "old_commit": args.old_commit,
        "seed": args.seed,
        "kernel": 3,
        "lambda": 1.0,
        "detach": False,
        "distance": "mse",
        "old_loss": float(old_loss.detach()),
        "cleaned_loss": float(new_loss.detach()),
        "loss_abs_diff": abs(float(old_loss.detach()) - float(new_loss.detach())),
        "old_grad_norm": float(old_features.grad.norm()),
        "cleaned_grad_norm": float(new_features.grad.norm()),
        "grad_max_abs_diff": float((old_features.grad - new_features.grad).abs().max()),
    }
    result["passed"] = (
        result["loss_abs_diff"] <= args.tolerance
        and result["grad_max_abs_diff"] <= args.tolerance
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(f"{key}={value}" for key, value in result.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
