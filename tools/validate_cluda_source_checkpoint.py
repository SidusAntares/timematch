"""Strict preflight for a plain_psecludatcn checkpoint used by full CLUDA."""

import argparse
import json
from pathlib import Path

import torch

from methods.cluda.config import CLUDAConfig
from methods.cluda.model import CLUDATCNClassifier


def validate(run_dir, source, target, seed):
    run_dir = Path(run_dir)
    config_path = run_dir / "train_config.json"
    checkpoint_path = run_dir / "fold_0" / "model.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required = {
        "model": "cludatcn",
        "input_dim": 10,
        "with_extra": False,
        "with_shift_aug": False,
        "cluda_channels": "64-64-64-64-64",
        "cluda_hidden_dim": 256,
        "cluda_kernel_size": 3,
        "cluda_dilation_factor": 2,
        "cluda_dropout": 0.0,
        "cluda_max_temporal_shift": 100,
        "closed_set": True,
        "combine_spring_and_winter": False,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"source checkpoint architecture mismatch: {mismatches}")
    identity = {"source": source, "target": target, "seed": seed}
    identity_mismatches = {
        key: (config.get(key), expected)
        for key, expected in identity.items()
        if config.get(key) != expected
    }
    if identity_mismatches:
        raise ValueError(f"source checkpoint identity mismatch: {identity_mismatches}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    num_classes = state_dict["predictor.output_fc.weight"].shape[0]
    classes = config.get("classes")
    if not isinstance(classes, list) or len(classes) != num_classes or len(set(classes)) != len(classes):
        raise ValueError("source checkpoint is missing a valid ordered classes mapping")
    method = CLUDAConfig.from_namespace(argparse.Namespace(**config))
    classifier = CLUDATCNClassifier(
        input_dim=config["input_dim"],
        num_classes=num_classes,
        channels=method.channels,
        hidden_dim=method.hidden_dim,
        kernel_size=method.kernel_size,
        stride=method.stride,
        dilation_factor=method.dilation_factor,
        dropout=method.dropout,
        max_temporal_shift=method.max_temporal_shift,
    )
    classifier.load_state_dict(state_dict, strict=True)
    print(f"CLUDA_SOURCE_PREFLIGHT_OK|checkpoint={checkpoint_path}|strict=true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    validate(args.run_dir, args.source, args.target, args.seed)
