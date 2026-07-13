import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dataset import PixelSetData, count_pixelset_samples, create_evaluation_loaders
from evaluation import evaluation
from models.stclassifier import PseGru, PseLTae, PseTae, PseTempCNN
from train import create_train_val_test_folds
from utils import label_utils
from utils.train_utils import bool_flag
from tools.v28_audit_utils import checkpoint_hashes


def build_model(name, input_dim, num_classes, with_extra):
    models = {
        "pseltae": PseLTae,
        "psetae": PseTae,
        "psetcnn": PseTempCNN,
        "psegru": PseGru,
    }
    return models[name](input_dim=input_dim, num_classes=num_classes, with_extra=with_extra)


def main():
    parser = argparse.ArgumentParser(description="Evaluate old/cleaned TimeMatch checkpoints with one evaluator.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--closed_set", type=bool_flag, default=True)
    parser.add_argument("--combine_spring_and_winter", type=bool_flag, default=False)
    parser.add_argument("--sample_pixels_val", type=bool_flag, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_loader_timeout", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_pixels", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--model", default="pseltae", choices=["pseltae", "psetae", "psetcnn", "psegru"])
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--with_extra", type=bool_flag, default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source_classes = label_utils.get_classes(
        args.source.split("/")[0],
        combine_spring_and_winter=args.combine_spring_and_winter,
    )
    if args.closed_set:
        source_classes = [name for name in source_classes if name != "unknown"]
    source_data = PixelSetData(args.data_root, args.source, source_classes, closed_set=args.closed_set)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[index] for index in labels[counts >= 200]]
    indices = {
        args.source: len(source_data),
        args.target: count_pixelset_samples(
            args.data_root,
            args.target,
            source_classes,
            closed_set=args.closed_set,
        ),
    }
    splits = create_train_val_test_folds(
        [args.source, args.target],
        1,
        indices,
        args.val_ratio,
        args.test_ratio,
    )[0]
    args.classes = source_classes
    args.num_classes = len(source_classes)
    _, test_loader = create_evaluation_loaders(
        args.target,
        splits,
        args,
        sample_pixels_val=args.sample_pixels_val,
    )

    model = build_model(args.model, args.input_dim, args.num_classes, args.with_extra).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=True)
    metrics = evaluation(model, test_loader, args.device, source_classes, mode="test")
    hashes = checkpoint_hashes(args.checkpoint)
    result = {
        **hashes,
        "load_status": "strict_ok",
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        "closed_set": args.closed_set,
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "kappa": float(metrics["kappa"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
