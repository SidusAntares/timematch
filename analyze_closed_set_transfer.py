import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import GroupByShapesBatchSampler, PixelSetData
from models.stclassifier import PseGru, PseLTae, PseTae, PseTempCNN
from transforms import Normalize, ToTensor
from utils import label_utils


TAG_TO_DATASET = {
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
    "DK1": "denmark/32VNH/2017",
    "AT1": "austria/33UVP/2017",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute task-level closed-set structure metrics for baseline TimeMatch runs."
    )
    parser.add_argument(
        "--data_root",
        default="/data/user/DBL/timematch_data",
        type=str,
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--result_root",
        default="result/baseline",
        type=str,
        help="Path to closed-set baseline result directory.",
    )
    parser.add_argument(
        "--outputs_root",
        default="outputs",
        type=str,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--output_csv",
        default="result/baseline_analysis/closed_set_transfer_metrics.csv",
        type=str,
        help="Destination CSV for aggregated metrics.",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument(
        "--model",
        default="pseltae",
        choices=["pseltae", "psetae", "psetcnn", "psegru"],
        help="Backbone used by the baseline experiments.",
    )
    parser.add_argument(
        "--with_extra",
        default=False,
        action="store_true",
        help="Whether geometric features are enabled in the model.",
    )
    parser.add_argument(
        "--max_mmd_samples",
        default=2048,
        type=int,
        help="Maximum number of features per domain used for MMD/CORAL.",
    )
    parser.add_argument(
        "--max_acf_lag",
        default=10,
        type=int,
        help="Maximum lag used to compute ACF distance.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_closed_set_classes(data_root, source_dataset):
    source_country = source_dataset.split("/")[0]
    source_classes = label_utils.get_classes(source_country, combine_spring_and_winter=False)
    source_data = PixelSetData(data_root, source_dataset, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    source_classes = [class_name for class_name in source_classes if class_name != "unknown"]
    return source_classes


def build_dataset(data_root, dataset_name, classes):
    transform = transforms.Compose([Normalize(), ToTensor()])
    return PixelSetData(
        data_root=data_root,
        dataset_name=dataset_name,
        classes=classes,
        transform=transform,
        with_extra=False,
    )


def build_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=GroupByShapesBatchSampler(dataset, batch_size),
        pin_memory=torch.cuda.is_available(),
    )


def build_model(model_name, num_classes, with_extra, device):
    if model_name == "pseltae":
        model = PseLTae(input_dim=10, num_classes=num_classes, with_extra=with_extra)
    elif model_name == "psetae":
        model = PseTae(input_dim=10, num_classes=num_classes, with_extra=with_extra)
    elif model_name == "psetcnn":
        model = PseTempCNN(input_dim=10, num_classes=num_classes, with_extra=with_extra)
    elif model_name == "psegru":
        model = PseGru(input_dim=10, num_classes=num_classes, with_extra=with_extra)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def compute_batch_acf(parcel_series, max_lag, eps=1e-6):
    # parcel_series: [B, T, C]
    batch_size, seq_len, num_channels = parcel_series.shape
    max_lag = min(max_lag, seq_len - 1)
    if max_lag <= 0:
        return parcel_series.new_zeros((num_channels, 0)), 0

    acf_values = []
    for lag in range(1, max_lag + 1):
        lhs = parcel_series[:, :-lag, :]
        rhs = parcel_series[:, lag:, :]

        lhs = lhs - lhs.mean(dim=1, keepdim=True)
        rhs = rhs - rhs.mean(dim=1, keepdim=True)

        numerator = (lhs * rhs).sum(dim=1)
        denominator = torch.sqrt((lhs.pow(2).sum(dim=1) * rhs.pow(2).sum(dim=1)).clamp_min(eps))
        corr = numerator / denominator
        acf_values.append(corr.mean(dim=0))

    acf_matrix = torch.stack(acf_values, dim=1)
    return acf_matrix, batch_size


@torch.no_grad()
def extract_domain_statistics(model, loader, device, max_acf_lag):
    feature_batches = []
    label_batches = []
    acf_sum = None
    acf_count = 0

    for sample in tqdm(loader, desc="Extracting", leave=False):
        labels = sample["label"]
        pixels = sample["pixels"].to(device=device, non_blocking=True)
        valid_pixels = sample["valid_pixels"].to(device=device, non_blocking=True)
        positions = sample["positions"].to(device=device, non_blocking=True)
        extra = sample["extra"].to(device=device, non_blocking=True)

        logits, feats = model.forward(pixels, valid_pixels, positions, extra, return_feats=True)
        del logits

        feature_batches.append(feats.detach().cpu())
        label_batches.append(labels.cpu())

        parcel_series = pixels.mean(dim=-1).permute(0, 1, 2)
        batch_acf, batch_size = compute_batch_acf(parcel_series, max_lag=max_acf_lag)
        batch_acf = batch_acf.detach().cpu()
        if acf_sum is None:
            acf_sum = batch_acf * batch_size
        else:
            acf_sum += batch_acf * batch_size
        acf_count += batch_size

    features = torch.cat(feature_batches, dim=0).float()
    labels = torch.cat(label_batches, dim=0).long()
    acf_mean = acf_sum / max(acf_count, 1)
    return {
        "features": features,
        "labels": labels,
        "acf": acf_mean,
    }


def maybe_subsample(features, labels, max_samples, seed):
    if features.shape[0] <= max_samples:
        return features, labels
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(features.shape[0], generator=generator)[:max_samples]
    return features[indices], labels[indices]


def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).t()
    return (x_norm + y_norm - 2.0 * x @ y.t()).clamp_min(0.0)


def rbf_kernel(x, y, gamma):
    return torch.exp(-gamma * pairwise_sq_dists(x, y))


def compute_mmd_rbf(source_feats, target_feats, eps=1e-6):
    combined = torch.cat([source_feats, target_feats], dim=0)
    if combined.shape[0] < 2:
        return float("nan")

    with torch.no_grad():
        dists = torch.pdist(combined, p=2)
        median_dist = torch.median(dists)
        sigma = median_dist.item() if torch.isfinite(median_dist) and median_dist.item() > eps else 1.0
        gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = rbf_kernel(source_feats, source_feats, gamma)
    k_yy = rbf_kernel(target_feats, target_feats, gamma)
    k_xy = rbf_kernel(source_feats, target_feats, gamma)
    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(mmd.item())


def compute_coral(source_feats, target_feats):
    source_centered = source_feats - source_feats.mean(dim=0, keepdim=True)
    target_centered = target_feats - target_feats.mean(dim=0, keepdim=True)

    source_cov = (source_centered.t() @ source_centered) / max(source_feats.shape[0] - 1, 1)
    target_cov = (target_centered.t() @ target_centered) / max(target_feats.shape[0] - 1, 1)
    dim = source_feats.shape[1]
    coral = ((source_cov - target_cov) ** 2).sum() / (4.0 * dim * dim)
    return float(coral.item())


def compute_prototype_distance(source_feats, source_labels, target_feats, target_labels):
    shared_classes = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    if not shared_classes:
        return float("nan"), 0

    distances = []
    for class_id in shared_classes:
        source_proto = source_feats[source_labels == class_id].mean(dim=0)
        target_proto = target_feats[target_labels == class_id].mean(dim=0)
        distances.append(torch.norm(source_proto - target_proto, p=2))

    distance = torch.stack(distances).mean()
    return float(distance.item()), len(shared_classes)


def compute_relation_structure_distance(source_feats, source_labels, target_feats, target_labels, eps=1e-6):
    shared_classes = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    if len(shared_classes) < 2:
        return float("nan"), len(shared_classes)

    source_prototypes = []
    target_prototypes = []
    for class_id in shared_classes:
        source_prototypes.append(source_feats[source_labels == class_id].mean(dim=0))
        target_prototypes.append(target_feats[target_labels == class_id].mean(dim=0))

    source_proto = torch.stack(source_prototypes, dim=0)
    target_proto = torch.stack(target_prototypes, dim=0)

    source_proto = torch.nn.functional.normalize(source_proto, dim=1)
    target_proto = torch.nn.functional.normalize(target_proto, dim=1)

    source_rel = source_proto @ source_proto.t()
    target_rel = target_proto @ target_proto.t()
    source_rel.fill_diagonal_(0.0)
    target_rel.fill_diagonal_(0.0)

    source_norm = source_rel.norm(p="fro")
    target_norm = target_rel.norm(p="fro")
    if source_norm > eps:
        source_rel = source_rel / source_norm
    if target_norm > eps:
        target_rel = target_rel / target_norm

    relation_distance = torch.mean((source_rel - target_rel) ** 2)
    return float(relation_distance.item()), len(shared_classes)


def compute_acf_distance(source_acf, target_acf):
    return float(torch.mean(torch.abs(source_acf - target_acf)).item())


def find_latest_result_dirs(result_root):
    latest = {}
    for source_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        for target_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            run_dirs = sorted(p for p in target_dir.iterdir() if p.is_dir())
            if not run_dirs:
                continue
            latest[(source_dir.name, target_dir.name)] = run_dirs[-1]
    return latest


def read_result_row(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return next(reader)


def ensure_parent_dir(path_str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    args = parse_args()
    set_seed(args.seed)

    result_root = Path(args.result_root)
    outputs_root = Path(args.outputs_root)
    output_csv = ensure_parent_dir(args.output_csv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    latest_result_dirs = find_latest_result_dirs(result_root)
    rows = []

    for (source_tag, target_tag), run_dir in sorted(latest_result_dirs.items()):
        source_dataset = TAG_TO_DATASET[source_tag]
        target_dataset = TAG_TO_DATASET[target_tag]
        result_row = read_result_row(run_dir / "results.csv")
        experiment_name = result_row["timematch_experiment"]
        checkpoint_path = outputs_root / experiment_name / "fold_0" / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {source_tag}->{target_tag}: {checkpoint_path}")

        closed_set_classes = get_closed_set_classes(args.data_root, source_dataset)
        source_dataset_obj = build_dataset(args.data_root, source_dataset, closed_set_classes)
        target_dataset_obj = build_dataset(args.data_root, target_dataset, closed_set_classes)
        source_loader = build_loader(source_dataset_obj, args.batch_size, args.num_workers)
        target_loader = build_loader(target_dataset_obj, args.batch_size, args.num_workers)

        model = build_model(
            model_name=args.model,
            num_classes=len(closed_set_classes),
            with_extra=args.with_extra,
            device=device,
        )
        model = load_checkpoint(model, checkpoint_path, device)

        print(f"\n[INFO] Computing metrics for {source_tag} -> {target_tag}")
        source_stats = extract_domain_statistics(model, source_loader, device, args.max_acf_lag)
        target_stats = extract_domain_statistics(model, target_loader, device, args.max_acf_lag)

        source_feats_for_global, source_labels_for_global = maybe_subsample(
            source_stats["features"], source_stats["labels"], args.max_mmd_samples, args.seed
        )
        target_feats_for_global, target_labels_for_global = maybe_subsample(
            target_stats["features"], target_stats["labels"], args.max_mmd_samples, args.seed
        )

        mmd_value = compute_mmd_rbf(source_feats_for_global, target_feats_for_global)
        coral_value = compute_coral(source_feats_for_global, target_feats_for_global)
        proto_dist, shared_classes = compute_prototype_distance(
            source_stats["features"],
            source_stats["labels"],
            target_stats["features"],
            target_stats["labels"],
        )
        relation_dist, relation_shared_classes = compute_relation_structure_distance(
            source_stats["features"],
            source_stats["labels"],
            target_stats["features"],
            target_stats["labels"],
        )
        acf_dist = compute_acf_distance(source_stats["acf"], target_stats["acf"])

        rows.append(
            {
                "source": source_tag,
                "target": target_tag,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "closed_set_num_classes": len(closed_set_classes),
                "shared_classes_for_proto": shared_classes,
                "timematch_experiment": experiment_name,
                "target_f1": float(result_row["timematch_macro_f1"]),
                "target_accuracy": float(result_row["timematch_accuracy"]),
                "mmd": mmd_value,
                "coral": coral_value,
                "prototype_distance": proto_dist,
                "relation_structure_distance": relation_dist,
                "acf_distance": acf_dist,
                "source_samples": len(source_dataset_obj),
                "target_samples": len(target_dataset_obj),
                "result_dir": str(run_dir),
                "checkpoint_path": str(checkpoint_path),
            }
        )

    fieldnames = [
        "source",
        "target",
        "source_dataset",
        "target_dataset",
        "closed_set_num_classes",
        "shared_classes_for_proto",
        "timematch_experiment",
        "target_f1",
        "target_accuracy",
        "mmd",
        "coral",
        "prototype_distance",
        "relation_structure_distance",
        "acf_distance",
        "source_samples",
        "target_samples",
        "result_dir",
        "checkpoint_path",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Saved transfer analysis table to: {output_csv}")


if __name__ == "__main__":
    main()
