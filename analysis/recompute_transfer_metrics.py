import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import GroupByShapesBatchSampler, PixelSetData
from models.stclassifier import PseGru, PseLTae, PseTae, PseTempCNN
from transforms import Normalize, ToTensor
from utils import label_utils
from utils.train_utils import bool_flag


TAG_TO_DATASET = {
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
    "DK1": "denmark/32VNH/2017",
    "AT1": "austria/33UVP/2017",
}

DATASET_TO_TAG = {value: key for key, value in TAG_TO_DATASET.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute compact transferability metrics for TimeMatch baselines."
    )
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data", type=str)
    parser.add_argument("--outputs_root", default="outputs", type=str)
    parser.add_argument(
        "--output_csv",
        default="result/baseline_analysis/open_set_transfer_metrics_recomputed.csv",
        type=str,
    )
    parser.add_argument("--closed_set", default=False, type=bool_flag)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument(
        "--model",
        default="pseltae",
        choices=["pseltae", "psetae", "psetcnn", "psegru"],
    )
    parser.add_argument("--with_extra", default=False, type=bool_flag)
    parser.add_argument("--max_feature_samples", default=2048, type=int)
    parser.add_argument("--max_acf_lag", default=10, type=int)
    parser.add_argument("--temporal_grid_size", default=30, type=int)
    parser.add_argument("--max_shift_steps", default=5, type=int)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent_dir(path_str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(text)


def row_key(source_tag, target_tag, closed_set):
    return source_tag, target_tag, str(bool(closed_set))


def load_existing_rows(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row_key(row["source"], row["target"], row.get("closed_set", False)): row
        for row in rows
    }


def get_task_classes(data_root, source_dataset, closed_set):
    source_country = source_dataset.split("/")[0]
    source_classes = label_utils.get_classes(source_country, combine_spring_and_winter=False)
    source_data = PixelSetData(data_root, source_dataset, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    if closed_set:
        source_classes = [class_name for class_name in source_classes if class_name != "unknown"]
    return source_classes


def build_dataset(data_root, dataset_name, classes, closed_set, with_extra):
    transform = transforms.Compose([Normalize(), ToTensor()])
    return PixelSetData(
        data_root=data_root,
        dataset_name=dataset_name,
        classes=classes,
        transform=transform,
        with_extra=with_extra,
        closed_set=closed_set,
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


def compute_batch_acf(parcel_series, max_lag, eps=1e-6):
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


def interpolate_curve(curve, positions, grid_size):
    positions = np.asarray(positions, dtype=np.float32)
    curve = np.asarray(curve, dtype=np.float32)
    if len(positions) == 0:
        return np.zeros((grid_size, curve.shape[-1]), dtype=np.float32)
    if len(positions) == 1:
        return np.repeat(curve, grid_size, axis=0)

    start = float(positions[0])
    end = float(positions[-1])
    if end <= start:
        end = start + 1.0
    grid = np.linspace(start, end, num=grid_size, dtype=np.float32)

    interpolated = np.zeros((grid_size, curve.shape[1]), dtype=np.float32)
    for dim in range(curve.shape[1]):
        interpolated[:, dim] = np.interp(grid, positions, curve[:, dim])
    return interpolated


@torch.no_grad()
def extract_domain_statistics(model, loader, device, num_classes, temporal_grid_size, max_acf_lag):
    feature_batches = []
    label_batches = []
    acf_sum = None
    acf_count = 0
    curve_sums = None
    curve_counts = torch.zeros(num_classes, dtype=torch.long)
    domain_positions = None
    raw_curve_sums = None
    raw_curve_counts = torch.zeros(num_classes, dtype=torch.long)
    raw_global_curve_sum = None
    raw_global_curve_count = 0
    ndvi_values = [[] for _ in range(num_classes)]

    for sample in tqdm(loader, desc="Extracting", leave=False):
        labels = sample["label"]
        pixels = sample["pixels"].to(device=device, non_blocking=True)
        valid_pixels = sample["valid_pixels"].to(device=device, non_blocking=True)
        positions = sample["positions"].to(device=device, non_blocking=True)
        extra = sample["extra"].to(device=device, non_blocking=True)

        logits, features = model.forward(pixels, valid_pixels, positions, extra, return_feats=True)
        del logits

        spatial_feats = model.spatial_encoder(pixels, valid_pixels, extra)
        spatial_feats = spatial_feats.detach().cpu()
        positions_cpu = sample["positions"].cpu()
        raw_series = pixels.mean(dim=-1).detach().cpu()  # [B, T, C]

        if domain_positions is None:
            domain_positions = positions_cpu[0].numpy()

        if curve_sums is None:
            curve_sums = torch.zeros(
                num_classes,
                temporal_grid_size,
                spatial_feats.shape[-1],
                dtype=torch.float32,
            )
        if raw_curve_sums is None:
            raw_curve_sums = torch.zeros(
                num_classes,
                temporal_grid_size,
                raw_series.shape[-1],
                dtype=torch.float32,
            )
        if raw_global_curve_sum is None:
            raw_global_curve_sum = torch.zeros(
                temporal_grid_size,
                raw_series.shape[-1],
                dtype=torch.float32,
            )

        for batch_idx in range(spatial_feats.shape[0]):
            class_id = int(labels[batch_idx].item())
            interpolated = interpolate_curve(
                spatial_feats[batch_idx].numpy(),
                positions_cpu[batch_idx].numpy(),
                temporal_grid_size,
            )
            curve_sums[class_id] += torch.from_numpy(interpolated)
            curve_counts[class_id] += 1

            raw_interpolated = interpolate_curve(
                raw_series[batch_idx].numpy(),
                positions_cpu[batch_idx].numpy(),
                temporal_grid_size,
            )
            raw_curve_sums[class_id] += torch.from_numpy(raw_interpolated)
            raw_curve_counts[class_id] += 1
            raw_global_curve_sum += torch.from_numpy(raw_interpolated)
            raw_global_curve_count += 1

            red = raw_series[batch_idx, :, 2]
            nir = raw_series[batch_idx, :, 6]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi_values[class_id].append(ndvi)

        feature_batches.append(features.detach().cpu())
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

    mean_curves = torch.zeros_like(curve_sums)
    valid_curve_mask = curve_counts > 0
    mean_curves[valid_curve_mask] = curve_sums[valid_curve_mask] / curve_counts[valid_curve_mask].unsqueeze(-1).unsqueeze(-1)

    raw_mean_curves = torch.zeros_like(raw_curve_sums)
    valid_raw_curve_mask = raw_curve_counts > 0
    raw_mean_curves[valid_raw_curve_mask] = (
        raw_curve_sums[valid_raw_curve_mask] / raw_curve_counts[valid_raw_curve_mask].unsqueeze(-1).unsqueeze(-1)
    )

    global_raw_curve = raw_global_curve_sum / max(raw_global_curve_count, 1)
    ndvi_values = [
        torch.cat(class_values, dim=0) if class_values else torch.empty(0, dtype=torch.float32)
        for class_values in ndvi_values
    ]

    return {
        "features": features,
        "labels": labels,
        "acf": acf_mean,
        "mean_curves": mean_curves,
        "curve_counts": curve_counts,
        "raw_mean_curves": raw_mean_curves,
        "raw_curve_counts": raw_curve_counts,
        "global_raw_curve": global_raw_curve,
        "ndvi_values": ndvi_values,
        "positions": domain_positions,
    }


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

    source_proto = F.normalize(source_proto, dim=1)
    target_proto = F.normalize(target_proto, dim=1)

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


def compute_phase_curve_distances(source_stats, target_stats):
    shared_classes = torch.nonzero(
        (source_stats["curve_counts"] > 0) & (target_stats["curve_counts"] > 0),
        as_tuple=False,
    ).flatten()
    if len(shared_classes) == 0:
        return {
            "pse_early_curve_distance": float("nan"),
            "pse_mid_curve_distance": float("nan"),
            "pse_late_curve_distance": float("nan"),
            "pse_trend_curve_distance": float("nan"),
            "shared_classes_for_curves": 0,
        }

    grid_size = source_stats["mean_curves"].shape[1]
    splits = np.array_split(np.arange(grid_size), 3)
    phase_names = [
        ("pse_early_curve_distance", splits[0]),
        ("pse_mid_curve_distance", splits[1]),
        ("pse_late_curve_distance", splits[2]),
    ]

    results = {}
    for metric_name, phase_indices in phase_names:
        phase_distances = []
        for class_id in shared_classes.tolist():
            source_phase = source_stats["mean_curves"][class_id, phase_indices].mean(dim=0)
            target_phase = target_stats["mean_curves"][class_id, phase_indices].mean(dim=0)
            phase_distances.append(torch.norm(source_phase - target_phase, p=2))
        results[metric_name] = float(torch.stack(phase_distances).mean().item())

    trend_distances = []
    for class_id in shared_classes.tolist():
        source_curve = source_stats["mean_curves"][class_id]
        target_curve = target_stats["mean_curves"][class_id]
        source_trend = source_curve[1:] - source_curve[:-1]
        target_trend = target_curve[1:] - target_curve[:-1]
        trend_distances.append(torch.norm(source_trend.flatten() - target_trend.flatten(), p=2))
    results["pse_trend_curve_distance"] = float(torch.stack(trend_distances).mean().item())
    results["shared_classes_for_curves"] = int(len(shared_classes))
    return results


def compute_source_curve_spread(domain_stats):
    valid_classes = torch.nonzero(domain_stats["curve_counts"] > 0, as_tuple=False).flatten()
    if len(valid_classes) < 2:
        return float("nan")
    curves = domain_stats["mean_curves"][valid_classes].reshape(len(valid_classes), -1)
    return float(torch.pdist(curves, p=2).mean().item())


def compute_source_curve_activity_range(domain_stats):
    valid_classes = torch.nonzero(domain_stats["curve_counts"] > 0, as_tuple=False).flatten()
    if len(valid_classes) == 0:
        return float("nan")
    activity = []
    for class_id in valid_classes.tolist():
        curve = domain_stats["mean_curves"][class_id]
        activity.append(torch.norm(curve.max(dim=0).values - curve.min(dim=0).values, p=2))
    return float(torch.stack(activity).mean().item())


def compute_source_fisher_ratio(features, labels, eps=1e-6):
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        return float("nan")

    global_mean = features.mean(dim=0)
    between = 0.0
    within = 0.0
    for class_id in classes:
        class_feats = features[labels == class_id]
        class_mean = class_feats.mean(dim=0)
        between += class_feats.shape[0] * torch.sum((class_mean - global_mean) ** 2)
        within += torch.sum((class_feats - class_mean) ** 2)

    ratio = between / (within + eps)
    return float(ratio.item())


def compute_global_curve_shift_mse(source_stats, target_stats):
    source_curve = source_stats["global_raw_curve"]
    target_curve = target_stats["global_raw_curve"]
    return float(torch.mean((source_curve - target_curve) ** 2).item())


def compute_domain_bandwise_curve_mse(source_stats, target_stats):
    source_curve = source_stats["global_raw_curve"]
    target_curve = target_stats["global_raw_curve"]
    bandwise = torch.mean((source_curve - target_curve) ** 2, dim=0)
    return float(torch.mean(bandwise).item())


def compute_best_shifted_curve_mse(source_stats, target_stats, max_shift_steps):
    source_curve = source_stats["global_raw_curve"]
    target_curve = target_stats["global_raw_curve"]
    best = None
    for shift in range(-max_shift_steps, max_shift_steps + 1):
        if shift == 0:
            src = source_curve
            tgt = target_curve
        elif shift > 0:
            src = source_curve[shift:]
            tgt = target_curve[:-shift]
        else:
            src = source_curve[:shift]
            tgt = target_curve[-shift:]
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            continue
        mse = torch.mean((src - tgt) ** 2).item()
        if best is None or mse < best:
            best = mse
    return float(best if best is not None else float("nan"))


def compute_classwise_curve_shift_mse_mean(source_stats, target_stats):
    shared_classes = torch.nonzero(
        (source_stats["raw_curve_counts"] > 0) & (target_stats["raw_curve_counts"] > 0),
        as_tuple=False,
    ).flatten()
    if len(shared_classes) == 0:
        return float("nan")
    mses = []
    for class_id in shared_classes.tolist():
        source_curve = source_stats["raw_mean_curves"][class_id]
        target_curve = target_stats["raw_mean_curves"][class_id]
        mses.append(torch.mean((source_curve - target_curve) ** 2))
    return float(torch.stack(mses).mean().item())


def compute_class_relative_curve_structure_metrics(source_stats, target_stats, eps=1e-6):
    shared_classes = torch.nonzero(
        (source_stats["raw_curve_counts"] > 0) & (target_stats["raw_curve_counts"] > 0),
        as_tuple=False,
    ).flatten()
    if len(shared_classes) < 2:
        return float("nan"), float("nan"), int(len(shared_classes))

    source_curves = source_stats["raw_mean_curves"][shared_classes].reshape(len(shared_classes), -1)
    target_curves = target_stats["raw_mean_curves"][shared_classes].reshape(len(shared_classes), -1)
    source_d = pairwise_sq_dists(source_curves, source_curves).sqrt()
    target_d = pairwise_sq_dists(target_curves, target_curves).sqrt()

    structure_mse = torch.mean((source_d - target_d) ** 2).item()

    tri_idx = torch.triu_indices(source_d.shape[0], source_d.shape[1], offset=1)
    source_vec = source_d[tri_idx[0], tri_idx[1]]
    target_vec = target_d[tri_idx[0], tri_idx[1]]
    source_centered = source_vec - source_vec.mean()
    target_centered = target_vec - target_vec.mean()
    denom = torch.sqrt((source_centered.pow(2).sum() * target_centered.pow(2).sum()).clamp_min(eps))
    structure_corr = float((source_centered * target_centered).sum().item() / denom.item())
    return float(structure_mse), structure_corr, int(len(shared_classes))


def compute_source_ndvi_q90_metrics(source_stats):
    ranges = []
    for ndvi in source_stats["ndvi_values"]:
        if ndvi.numel() == 0:
            continue
        q95 = torch.quantile(ndvi, 0.95)
        q05 = torch.quantile(ndvi, 0.05)
        ranges.append(float((q95 - q05).item()))
    if not ranges:
        return float("nan"), float("nan")
    return float(np.mean(ranges)), float(np.std(ranges))


def compute_source_class_curve_variance_mean(source_stats):
    valid_classes = torch.nonzero(source_stats["raw_curve_counts"] > 0, as_tuple=False).flatten()
    if len(valid_classes) == 0:
        return float("nan")
    variances = []
    for class_id in valid_classes.tolist():
        curve = source_stats["raw_mean_curves"][class_id]
        variances.append(torch.var(curve, dim=0, unbiased=False).mean())
    return float(torch.stack(variances).mean().item())


def compute_domain_ndvi_curve_mse(source_stats, target_stats):
    source_curve = source_stats["global_raw_curve"]
    target_curve = target_stats["global_raw_curve"]
    source_red = source_curve[:, 2]
    source_nir = source_curve[:, 6]
    target_red = target_curve[:, 2]
    target_nir = target_curve[:, 6]
    source_ndvi = (source_nir - source_red) / (source_nir + source_red + 1e-6)
    target_ndvi = (target_nir - target_red) / (target_nir + target_red + 1e-6)
    return float(torch.mean((source_ndvi - target_ndvi) ** 2).item())


def read_target_metrics(outputs_root, experiment_name, target_dataset):
    target_name = target_dataset.replace("/", "_")
    overall_path = outputs_root / experiment_name / f"overall_{target_name}.json"
    if overall_path.exists():
        with open(overall_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        return float(np.mean(metrics["macro_f1"])), float(np.mean(metrics["accuracy"])), len(metrics["macro_f1"])

    fold_metrics = []
    fold_accuracies = []
    fold_dirs = sorted((outputs_root / experiment_name).glob("fold_*"))
    for fold_dir in fold_dirs:
        metrics_path = fold_dir / f"test_metrics_{target_name}.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)
            fold_metrics.append(float(metrics["macro_f1"]))
            fold_accuracies.append(float(metrics["accuracy"]))
    if not fold_metrics:
        raise FileNotFoundError(f"Missing evaluation metrics for {experiment_name} -> {target_name}")
    return float(np.mean(fold_metrics)), float(np.mean(fold_accuracies)), len(fold_metrics)


def iter_task_pairs():
    tags = list(TAG_TO_DATASET.keys())
    for source_tag in tags:
        for target_tag in tags:
            if source_tag != target_tag:
                yield source_tag, target_tag


def get_experiment_name(source_dataset, target_dataset, closed_set):
    source_tile = source_dataset.split("/")[1]
    target_tile = target_dataset.split("/")[1]
    if closed_set:
        return f"timematch_{source_tile}_to_{target_tile}_closedset_noshift"
    return f"timematch_{source_tile}_to_{target_tile}"


def get_fold_checkpoint_paths(outputs_root, experiment_name):
    experiment_dir = outputs_root / experiment_name
    fold_dirs = sorted(experiment_dir.glob("fold_*"))
    checkpoints = [fold_dir / "model.pt" for fold_dir in fold_dirs if (fold_dir / "model.pt").exists()]
    if checkpoints:
        return checkpoints
    raise FileNotFoundError(f"Missing checkpoints under {experiment_dir}")


def aggregate_metric_dicts(metric_dicts):
    keys = metric_dicts[0].keys()
    aggregated = {}
    for key in keys:
        values = [metric[key] for metric in metric_dicts]
        if isinstance(values[0], (int, np.integer)):
            aggregated[key] = int(round(float(np.mean(values))))
        elif isinstance(values[0], float):
            aggregated[key] = float(np.mean(values))
        else:
            aggregated[key] = values[0]
    return aggregated


def compute_metric_correlations(rows, metric_fields):
    y = [safe_float(row["target_f1"]) for row in rows]
    correlations = {}
    for field in metric_fields:
        pairs = [
            (safe_float(row.get(field)), target)
            for row, target in zip(rows, y)
            if safe_float(row.get(field)) is not None and target is not None
        ]
        if len(pairs) < 2:
            correlations[field] = float("nan")
            continue
        xs = np.array([p[0] for p in pairs], dtype=np.float64)
        ys = np.array([p[1] for p in pairs], dtype=np.float64)
        if np.std(xs) == 0 or np.std(ys) == 0:
            correlations[field] = float("nan")
            continue
        correlations[field] = float(np.corrcoef(xs, ys)[0, 1])
    return correlations


def main():
    args = parse_args()
    set_seed(args.seed)

    outputs_root = Path(args.outputs_root)
    output_csv = ensure_parent_dir(args.output_csv)
    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    existing_rows = load_existing_rows(output_csv)

    rows = []
    metric_fields = [
        "prototype_distance",
        "relation_structure_distance",
        "pse_early_curve_distance",
        "pse_mid_curve_distance",
        "pse_late_curve_distance",
        "pse_trend_curve_distance",
        "source_curve_spread",
        "source_curve_activity_range",
        "source_fisher_ratio",
        "mmd",
        "coral",
        "acf_distance",
        "global_curve_shift_mse",
        "domain_bandwise_curve_mse",
        "best_shifted_curve_mse",
        "classwise_curve_shift_mse_mean",
        "class_relative_curve_structure_mse",
        "class_relative_curve_structure_corr",
        "source_ndvi_q90_range_mean",
        "source_ndvi_q90_range_std",
        "source_class_curve_variance_mean",
        "domain_ndvi_curve_mse",
    ]
    for source_tag, target_tag in iter_task_pairs():
        source_dataset = TAG_TO_DATASET[source_tag]
        target_dataset = TAG_TO_DATASET[target_tag]
        experiment_name = get_experiment_name(source_dataset, target_dataset, args.closed_set)
        checkpoint_paths = get_fold_checkpoint_paths(outputs_root, experiment_name)
        target_f1, target_accuracy, fold_count = read_target_metrics(outputs_root, experiment_name, target_dataset)
        existing_row = existing_rows.get(row_key(source_tag, target_tag, args.closed_set))

        task_classes = get_task_classes(args.data_root, source_dataset, args.closed_set)
        source_dataset_obj = build_dataset(
            args.data_root,
            source_dataset,
            task_classes,
            closed_set=args.closed_set,
            with_extra=args.with_extra,
        )
        target_dataset_obj = build_dataset(
            args.data_root,
            target_dataset,
            task_classes,
            closed_set=args.closed_set,
            with_extra=args.with_extra,
        )
        source_loader = build_loader(source_dataset_obj, args.batch_size, args.num_workers)
        target_loader = build_loader(target_dataset_obj, args.batch_size, args.num_workers)

        fold_metrics = []
        print(f"\n[INFO] Recomputing metrics for {source_tag} -> {target_tag} ({experiment_name})")
        for fold_idx, checkpoint_path in enumerate(checkpoint_paths):
            print(f"[INFO] Fold {fold_idx}: {checkpoint_path}")
            model = build_model(
                model_name=args.model,
                num_classes=len(task_classes),
                with_extra=args.with_extra,
                device=device,
            )
            load_checkpoint(model, checkpoint_path, device)

            source_stats = extract_domain_statistics(
                model,
                source_loader,
                device,
                num_classes=len(task_classes),
                temporal_grid_size=args.temporal_grid_size,
                max_acf_lag=args.max_acf_lag,
            )
            target_stats = extract_domain_statistics(
                model,
                target_loader,
                device,
                num_classes=len(task_classes),
                temporal_grid_size=args.temporal_grid_size,
                max_acf_lag=args.max_acf_lag,
            )

            source_feats_global, source_labels_global = maybe_subsample(
                source_stats["features"],
                source_stats["labels"],
                args.max_feature_samples,
                args.seed + fold_idx,
            )
            target_feats_global, target_labels_global = maybe_subsample(
                target_stats["features"],
                target_stats["labels"],
                args.max_feature_samples,
                args.seed + 100 + fold_idx,
            )

            proto_dist, shared_proto = compute_prototype_distance(
                source_stats["features"],
                source_stats["labels"],
                target_stats["features"],
                target_stats["labels"],
            )
            relation_dist, shared_rel = compute_relation_structure_distance(
                source_stats["features"],
                source_stats["labels"],
                target_stats["features"],
                target_stats["labels"],
            )
            phase_metrics = compute_phase_curve_distances(source_stats, target_stats)
            class_relative_mse, class_relative_corr, shared_curve_structure_classes = compute_class_relative_curve_structure_metrics(
                source_stats, target_stats
            )
            source_ndvi_q90_mean, source_ndvi_q90_std = compute_source_ndvi_q90_metrics(source_stats)

            fold_metric = {
                "mmd": safe_float(existing_row.get("mmd")) if existing_row and safe_float(existing_row.get("mmd")) is not None else compute_mmd_rbf(source_feats_global, target_feats_global),
                "coral": safe_float(existing_row.get("coral")) if existing_row and safe_float(existing_row.get("coral")) is not None else compute_coral(source_feats_global, target_feats_global),
                "prototype_distance": safe_float(existing_row.get("prototype_distance")) if existing_row and safe_float(existing_row.get("prototype_distance")) is not None else proto_dist,
                "relation_structure_distance": safe_float(existing_row.get("relation_structure_distance")) if existing_row and safe_float(existing_row.get("relation_structure_distance")) is not None else relation_dist,
                "acf_distance": safe_float(existing_row.get("acf_distance")) if existing_row and safe_float(existing_row.get("acf_distance")) is not None else compute_acf_distance(source_stats["acf"], target_stats["acf"]),
                "source_curve_spread": safe_float(existing_row.get("source_curve_spread")) if existing_row and safe_float(existing_row.get("source_curve_spread")) is not None else compute_source_curve_spread(source_stats),
                "source_curve_activity_range": safe_float(existing_row.get("source_curve_activity_range")) if existing_row and safe_float(existing_row.get("source_curve_activity_range")) is not None else compute_source_curve_activity_range(source_stats),
                "source_fisher_ratio": safe_float(existing_row.get("source_fisher_ratio")) if existing_row and safe_float(existing_row.get("source_fisher_ratio")) is not None else compute_source_fisher_ratio(
                    source_stats["features"], source_stats["labels"]
                ),
                "shared_classes_for_proto": shared_proto,
                "shared_classes_for_relation": shared_rel,
                "shared_classes_for_curves": phase_metrics["shared_classes_for_curves"],
                "pse_early_curve_distance": safe_float(existing_row.get("pse_early_curve_distance")) if existing_row and safe_float(existing_row.get("pse_early_curve_distance")) is not None else phase_metrics["pse_early_curve_distance"],
                "pse_mid_curve_distance": safe_float(existing_row.get("pse_mid_curve_distance")) if existing_row and safe_float(existing_row.get("pse_mid_curve_distance")) is not None else phase_metrics["pse_mid_curve_distance"],
                "pse_late_curve_distance": safe_float(existing_row.get("pse_late_curve_distance")) if existing_row and safe_float(existing_row.get("pse_late_curve_distance")) is not None else phase_metrics["pse_late_curve_distance"],
                "pse_trend_curve_distance": safe_float(existing_row.get("pse_trend_curve_distance")) if existing_row and safe_float(existing_row.get("pse_trend_curve_distance")) is not None else phase_metrics["pse_trend_curve_distance"],
                "global_curve_shift_mse": compute_global_curve_shift_mse(source_stats, target_stats),
                "domain_bandwise_curve_mse": compute_domain_bandwise_curve_mse(source_stats, target_stats),
                "best_shifted_curve_mse": compute_best_shifted_curve_mse(source_stats, target_stats, args.max_shift_steps),
                "classwise_curve_shift_mse_mean": compute_classwise_curve_shift_mse_mean(source_stats, target_stats),
                "class_relative_curve_structure_mse": class_relative_mse,
                "class_relative_curve_structure_corr": class_relative_corr,
                "source_ndvi_q90_range_mean": source_ndvi_q90_mean,
                "source_ndvi_q90_range_std": source_ndvi_q90_std,
                "source_class_curve_variance_mean": compute_source_class_curve_variance_mean(source_stats),
                "domain_ndvi_curve_mse": compute_domain_ndvi_curve_mse(source_stats, target_stats),
                "shared_classes_for_curve_structure": shared_curve_structure_classes,
            }
            fold_metrics.append(fold_metric)

        metrics = aggregate_metric_dicts(fold_metrics)
        row = {
            "source": source_tag,
            "target": target_tag,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "closed_set": bool(args.closed_set),
            "num_classes": len(task_classes),
            "fold_count": fold_count,
            "timematch_experiment": experiment_name,
            "target_f1": target_f1,
            "target_accuracy": target_accuracy,
            "shared_classes_for_proto": metrics["shared_classes_for_proto"],
            "shared_classes_for_relation": metrics["shared_classes_for_relation"],
            "shared_classes_for_curves": metrics["shared_classes_for_curves"],
            "shared_classes_for_curve_structure": metrics["shared_classes_for_curve_structure"],
            "source_samples": len(source_dataset_obj),
            "target_samples": len(target_dataset_obj),
            "checkpoint_dir": str((outputs_root / experiment_name).resolve()),
        }
        row.update({field: metrics[field] for field in metric_fields})
        rows.append(row)

    metric_correlations = compute_metric_correlations(rows, metric_fields)
    sorted_metric_fields = sorted(
        metric_fields,
        key=lambda field: (
            1 if np.isnan(metric_correlations[field]) else 0,
            0 if np.isnan(metric_correlations[field]) else -abs(metric_correlations[field]),
            field,
        ),
    )

    metadata_fields = [
        "source",
        "target",
        "source_dataset",
        "target_dataset",
        "closed_set",
        "num_classes",
        "fold_count",
        "timematch_experiment",
        "target_f1",
        "target_accuracy",
    ]
    auxiliary_fields = [
        "shared_classes_for_proto",
        "shared_classes_for_relation",
        "shared_classes_for_curves",
        "shared_classes_for_curve_structure",
        "source_samples",
        "target_samples",
        "checkpoint_dir",
    ]
    fieldnames = metadata_fields + sorted_metric_fields + auxiliary_fields

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Saved transfer analysis table to: {output_csv}")


if __name__ == "__main__":
    main()
