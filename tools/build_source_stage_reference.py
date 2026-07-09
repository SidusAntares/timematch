"""Build source class-stage references for v3.2.1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from distutils.util import strtobool

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch.utils import data
from torchvision import transforms

from dataset import GroupByShapesBatchSampler, PixelSetData
from methods.local_shift.source_reference import build_source_stage_reference
from models.stclassifier import PseGru, PseLTae, PseTae, PseTempCNN
from transforms import Normalize, RandomSamplePixels, ToTensor
from utils import label_utils


def bool_flag(value):
    return bool(strtobool(str(value)))


def build_model(args, num_classes):
    model_kwargs = {
        "input_dim": args.input_dim,
        "with_extra": args.with_extra,
        "num_classes": num_classes,
        "max_temporal_shift": args.max_temporal_shift,
    }
    if args.model == "pseltae":
        return PseLTae(**model_kwargs)
    if args.model == "psetae":
        return PseTae(**model_kwargs)
    if args.model == "psegru":
        return PseGru(**model_kwargs)
    if args.model == "psetcnn":
        return PseTempCNN(**model_kwargs, seq_len=args.seq_length)
    raise ValueError(f"unsupported model: {args.model}")


def infer_num_classes_from_checkpoint(weights_path: str) -> int:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    decoder_weights = []
    for key, value in state_dict.items():
        if key.startswith("decoder.") and key.endswith(".weight") and torch.is_tensor(value) and value.ndim == 2:
            parts = key.split(".")
            try:
                layer_idx = int(parts[1])
            except (IndexError, ValueError):
                layer_idx = -1
            decoder_weights.append((layer_idx, key, int(value.shape[0])))
    if not decoder_weights:
        raise ValueError(f"could not infer num_classes from checkpoint decoder weights: {weights_path}")
    decoder_weights.sort(key=lambda item: item[0])
    return decoder_weights[-1][2]


def infer_classes(args, num_classes):
    if args.classes:
        return [item.strip() for item in args.classes.split(",") if item.strip()]
    country = args.source.split("/")[0]
    classes = label_utils.get_classes(country, combine_spring_and_winter=args.combine_spring_and_winter)
    if args.closed_set:
        classes = [cls for cls in classes if cls != "unknown"]
    if args.source_min_count > 0:
        source_data = PixelSetData(
            args.data_root,
            args.source,
            classes,
            closed_set=args.closed_set,
        )
        labels, counts = torch.as_tensor(source_data.get_labels()).unique(return_counts=True)
        keep = {int(label.item()) for label, count in zip(labels, counts) if int(count.item()) >= args.source_min_count}
        classes = [cls for idx, cls in enumerate(classes) if idx in keep]
    if num_classes > 0 and len(classes) != num_classes:
        raise ValueError(
            f"inferred {len(classes)} source classes but checkpoint expects {num_classes}. "
            "Pass --classes explicitly if this source checkpoint used a custom class list."
        )
    return classes


def load_checkpoint(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v3.2.1 source stage reference.")
    parser.add_argument("--weights", required=True, help="source checkpoint path, usually fold_0/model.pt")
    parser.add_argument("--source", required=True, help="source dataset name, e.g. france/31TCJ/2017")
    parser.add_argument("--output", required=True, help="output .pt path")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--classes", default="", help="optional comma-separated class names")
    parser.add_argument("--num_classes", type=int, default=0, help="<=0 infers from checkpoint decoder")
    parser.add_argument("--source_min_count", type=int, default=200, help="match train.py source class filtering")
    parser.add_argument("--model", default="pseltae", choices=["pseltae", "psetae", "psegru", "psetcnn"])
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--with_extra", type=bool_flag, default=True)
    parser.add_argument("--closed_set", type=bool_flag, default=True)
    parser.add_argument("--combine_spring_and_winter", type=bool_flag, default=False)
    parser.add_argument("--max_temporal_shift", type=int, default=100)
    parser.add_argument("--seq_length", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_pixels", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--min_stage_len", type=int, default=3)
    parser.add_argument("--change_threshold", type=float, default=None)
    parser.add_argument("--change_quantile", type=float, default=0.75)
    parser.add_argument("--nms_radius", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    num_classes = args.num_classes if args.num_classes > 0 else infer_num_classes_from_checkpoint(args.weights)
    classes = infer_classes(args, num_classes)

    transform = transforms.Compose([RandomSamplePixels(args.num_pixels), Normalize(), ToTensor()])
    dataset = PixelSetData(
        args.data_root,
        args.source,
        classes,
        transform=transform,
        closed_set=args.closed_set,
    )
    loader = data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_sampler=GroupByShapesBatchSampler(dataset, args.batch_size, by_pixel_dim=True),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args, num_classes=len(classes))
    model = load_checkpoint(model, args.weights, device).to(device)
    reference = build_source_stage_reference(
        model=model,
        source_loader=loader,
        device=device,
        num_classes=len(classes),
        kmax=args.kmax,
        min_stage_len=args.min_stage_len,
        change_threshold=args.change_threshold,
        change_quantile=args.change_quantile,
        nms_radius=args.nms_radius,
        max_batches=args.max_batches if args.max_batches > 0 else None,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(reference, args.output)
    summary = reference.get("summary", {})
    print("SOURCE_STAGE_REFERENCE_SUMMARY|" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"SAVED|{args.output}")


if __name__ == "__main__":
    main()
