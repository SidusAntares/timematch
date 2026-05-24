import argparse
import csv
import json
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import GroupByShapesBatchSampler
from data_adapters.factory import (
    build_dataset,
    get_classes_for_config,
    get_dataset_length,
    is_har,
    make_eval_transform,
)
from ideas.source_feature_reshaper import build_source_feature_reshaper
from ideas.source_phase_compactness import (
    _segment_masks_from_spec,
)
from models.stclassifier import PseGru, PseLTae, PseTae, PseTempCNN
from timematch import estimate_temporal_shift
from utils.train_utils import bool_flag, to_cuda


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "v2.7.2a offline diagnostic: discover shift-aligned, "
            "ambiguity-conditioned adaptive temporal segments."
        )
    )
    parser.add_argument("--run_dir", required=True, help="Source training run directory containing train_config.json.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to run_dir/fold_0/model.pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default="", help="Override data_root from train_config.json.")
    parser.add_argument("--source_max_batches", default=64, type=int)
    parser.add_argument("--target_max_batches", default=64, type=int)
    parser.add_argument("--shift_sample_size", default=40, type=int)
    parser.add_argument("--target_to_source_shift", default=None, type=int)
    parser.add_argument("--shift_jitter", default=3, type=int)
    parser.add_argument("--atomic_bins", default=12, type=int)
    parser.add_argument("--top_k_segments", default=8, type=int)
    parser.add_argument("--segment_score_quantile", default=0.75, type=float)
    parser.add_argument("--min_segment_score", default=0.0, type=float)
    parser.add_argument("--min_segment_ratio", default=1.0, type=float)
    parser.add_argument("--max_margin", default=0.20, type=float)
    parser.add_argument("--min_top2_mass", default=0.35, type=float)
    parser.add_argument("--prototype_temperature", default=1.0, type=float)
    parser.add_argument("--baseline_pairs_per_sample", default=4, type=int)
    parser.add_argument("--baseline_mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--apply_source_reshaper", default=False, type=bool_flag)
    parser.add_argument("--shuffle_pair_baseline", default=True, type=bool_flag)
    parser.add_argument("--seed", default=None, type=int)
    return parser.parse_args()


def load_config(run_dir, data_root_override, device_override, seed_override):
    config_path = os.path.join(run_dir, "train_config.json")
    with open(config_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if data_root_override:
        data["data_root"] = data_root_override
    if device_override:
        data["device"] = device_override
    if seed_override is not None:
        data["seed"] = int(seed_override)
    return Namespace(**data)


def build_model(config):
    model_name = str(config.model).lower()
    if model_name == "pseltae":
        return PseLTae(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    if model_name == "psetae":
        return PseTae(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    if model_name == "psetcnn":
        return PseTempCNN(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    if model_name == "psegru":
        return PseGru(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
    raise ValueError(f"Unsupported model: {config.model}")


def load_model_and_reshaper(config, checkpoint_path, device):
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    reshaper = build_source_feature_reshaper(
        kind=getattr(config, "source_feature_reshaper", "none"),
        feature_dim=model.spatial_encoder.output_dim,
        strength=getattr(config, "source_feature_reshaper_strength", 0.10),
        kernel_size=getattr(config, "source_feature_reshaper_kernel_size", 3),
    )
    if reshaper is not None:
        reshaper.to(device)
        if "source_feature_reshaper_state_dict" in checkpoint:
            reshaper.load_state_dict(checkpoint["source_feature_reshaper_state_dict"])
        reshaper.eval()
    return model, reshaper


def make_splits(config):
    classes = get_classes_for_config(config)
    config.classes = classes
    config.num_classes = len(classes)
    indices = {
        config.source: get_dataset_length(config, config.source, split="train"),
        config.target: get_dataset_length(config, config.target, split="train"),
    }
    return create_train_val_test_folds_local(
        [config.source, config.target],
        config.num_folds,
        indices,
        config.val_ratio,
        config.test_ratio,
    )[0]


def create_train_val_test_folds_local(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset_name in datasets:
            indices = list(range(num_indices[dataset_name]))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val
            random.shuffle(indices)
            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:]) if n_test > 0 else set()
            splits[dataset_name] = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
        folds.append(splits)
    return folds


def create_diagnostic_dataset(config, dataset_name, splits):
    transform = make_eval_transform(config, sample_pixels=True, sample_time=False)
    return build_dataset(
        config,
        dataset_name,
        config.classes,
        transform=transform,
        indices=splits[dataset_name]["train"],
        split="train",
    )


def create_diagnostic_loader(dataset, config):
    if is_har(config):
        return data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return data.DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(
            dataset,
            config.batch_size,
            by_pixel_dim=False,
        ),
    )


def build_atomic_partition_spec(date_positions, atomic_bins):
    sorted_positions = np.asarray(sorted({int(pos) for pos in date_positions}), dtype=np.int64)
    if sorted_positions.size == 0:
        raise ValueError("Cannot build atomic partition without date positions.")
    bins = max(1, min(int(atomic_bins), int(sorted_positions.size)))
    chunks = np.array_split(sorted_positions, bins)
    intervals = [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size > 0]
    return {
        "mode": "adaptive_atomic",
        "phase_count": len(intervals),
        "segment_count": len(intervals),
        "intervals": intervals,
        "date_positions": sorted_positions.tolist(),
        "atomic_bins": int(bins),
    }


def describe_atomic_partition_spec(spec):
    intervals = spec.get("intervals") or []
    interval_text = ", ".join([f"[{start},{end}]" for start, end in intervals])
    return (
        f"mode={spec.get('mode', 'adaptive_atomic')}, "
        f"atomic_bins={spec.get('atomic_bins', len(intervals))}, "
        f"segment_count={len(intervals)}, intervals={interval_text}"
    )


def get_spatial_features(model, reshaper, sample, device, apply_reshaper, labels=None):
    pixels, mask, positions, extra = to_cuda(sample, device)
    feats = model.spatial_encoder(pixels, mask, extra)
    if reshaper is not None and apply_reshaper:
        feats = reshaper(feats, positions=positions, labels=labels)
    return feats, positions, extra


@torch.no_grad()
def compute_source_segment_prototypes(
    model,
    reshaper,
    loader,
    phase_partition_spec,
    num_classes,
    device,
    max_batches,
    apply_reshaper,
):
    sums = None
    sq_sums = None
    counts = None
    feature_dim = None

    for batch_idx, sample in enumerate(tqdm(loader, desc="source prototypes")):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        labels = sample["label"].to(device=device, non_blocking=True)
        feats, positions, _ = get_spatial_features(
            model,
            reshaper,
            sample,
            device,
            apply_reshaper=apply_reshaper,
            labels=labels,
        )
        if feature_dim is None:
            feature_dim = feats.shape[-1]
            segment_count = int(phase_partition_spec["phase_count"])
            sums = torch.zeros(num_classes, segment_count, feature_dim, device=device)
            sq_sums = torch.zeros_like(sums)
            counts = torch.zeros(num_classes, segment_count, device=device)

        masks = _segment_masks_from_spec(positions, phase_partition_spec)
        for seg_idx, mask in enumerate(masks):
            valid = mask.sum(dim=1) > 0
            if not bool(valid.any().item()):
                continue
            weights = mask.to(dtype=feats.dtype).unsqueeze(-1)
            pooled = (feats * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            pooled = pooled[valid]
            pooled_labels = labels[valid]
            for class_id in pooled_labels.unique(sorted=True):
                class_mask = pooled_labels == class_id
                class_feats = pooled[class_mask]
                idx = int(class_id.item())
                sums[idx, seg_idx] += class_feats.sum(dim=0)
                sq_sums[idx, seg_idx] += class_feats.pow(2).sum(dim=0)
                counts[idx, seg_idx] += class_feats.shape[0]

    if feature_dim is None:
        raise RuntimeError("No source batches were collected.")
    prototypes = sums / counts.clamp_min(1.0).unsqueeze(-1)
    variances = sq_sums / counts.clamp_min(1.0).unsqueeze(-1) - prototypes.pow(2)
    variances = variances.clamp_min(0.0).mean(dim=-1)
    return prototypes, variances, counts


def pair_key(a, b):
    a, b = int(a), int(b)
    if a <= b:
        return f"{a}:{b}", a, b
    return f"{b}:{a}", b, a


def safe_exp_score(distance, scale, temperature):
    denom = max(float(scale) * float(temperature), 1e-6)
    return float(math.exp(-float(distance) / denom))


def compute_pair_components(
    target_feat,
    prototypes,
    variances,
    counts,
    seg_idx,
    class_a,
    class_b,
    prototype_temperature,
):
    if counts[class_a, seg_idx] < 2 or counts[class_b, seg_idx] < 2:
        return None
    proto_a = prototypes[class_a, seg_idx]
    proto_b = prototypes[class_b, seg_idx]
    source_sep_raw = torch.norm(proto_a - proto_b, p=2).item()
    source_scale = float(
        torch.sqrt(variances[class_a, seg_idx]).item()
        + torch.sqrt(variances[class_b, seg_idx]).item()
        + 1e-6
    )
    source_separability = source_sep_raw / source_scale
    dist_a = torch.norm(target_feat - proto_a, p=2).item()
    dist_b = torch.norm(target_feat - proto_b, p=2).item()
    min_pair_dist = min(dist_a, dist_b)
    explainability = safe_exp_score(
        min_pair_dist,
        source_scale + source_sep_raw,
        prototype_temperature,
    )
    return {
        "source_sep_raw": source_sep_raw,
        "source_scale": source_scale,
        "source_separability": source_separability,
        "target_explainability": explainability,
        "min_pair_dist": min_pair_dist,
    }


def sample_baseline_pairs(num_classes, class_a, class_b, count):
    candidates = [idx for idx in range(num_classes) if idx not in {int(class_a), int(class_b)}]
    if not candidates:
        return []
    pairs = []
    for _ in range(max(1, int(count))):
        other = random.choice(candidates)
        anchor = int(class_a) if random.random() < 0.5 else int(class_b)
        pairs.append(pair_key(anchor, other)[1:])
    return pairs


@torch.no_grad()
def discover_target_segments(
    model,
    reshaper,
    loader,
    prototypes,
    variances,
    counts,
    phase_partition_spec,
    config,
    device,
    target_to_source_shift,
    max_batches,
    max_margin,
    min_top2_mass,
    prototype_temperature,
    shift_jitter,
    shuffle_pair_baseline=True,
    baseline_pairs_per_sample=4,
    baseline_mode="mean",
):
    num_classes, segment_count, _ = prototypes.shape
    segment_stats = defaultdict(lambda: defaultdict(float))
    pair_stats = defaultdict(lambda: defaultdict(float))
    baseline_segment_stats = defaultdict(lambda: defaultdict(float))
    accepted = 0
    seen = 0

    shifts = [int(target_to_source_shift)]
    if int(shift_jitter) > 0:
        shifts.extend([int(target_to_source_shift) - int(shift_jitter), int(target_to_source_shift) + int(shift_jitter)])

    for batch_idx, sample in enumerate(tqdm(loader, desc="target ambiguity")):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        pixels, mask, positions, extra = to_cuda(sample, device)
        feats_for_segments = model.spatial_encoder(pixels, mask, extra)

        logits_by_shift = []
        for shift in shifts:
            logits_by_shift.append(model.decoder(model.temporal_encoder(feats_for_segments, positions + int(shift))))
        probs = F.softmax(logits_by_shift[0], dim=1)
        top2_prob, top2_idx = torch.topk(probs, k=min(2, num_classes), dim=1)
        if top2_prob.shape[1] < 2:
            continue

        stable_pair = torch.ones(probs.shape[0], dtype=torch.bool, device=device)
        if len(logits_by_shift) > 1:
            main_pairs = torch.sort(top2_idx, dim=1).values
            for logits in logits_by_shift[1:]:
                pair = torch.topk(F.softmax(logits, dim=1), k=2, dim=1).indices
                pair = torch.sort(pair, dim=1).values
                stable_pair &= (pair == main_pairs).all(dim=1)

        margins = top2_prob[:, 0] - top2_prob[:, 1]
        top2_mass = top2_prob.sum(dim=1)
        ambiguous = (margins <= float(max_margin)) & (top2_mass >= float(min_top2_mass))
        accepted_mask = ambiguous & stable_pair
        seen += int(probs.shape[0])
        accepted += int(accepted_mask.sum().item())
        if not bool(accepted_mask.any().item()):
            continue

        masks = _segment_masks_from_spec(positions, phase_partition_spec)
        for seg_idx, seg_mask in enumerate(masks):
            valid_segment = seg_mask.sum(dim=1) > 0
            valid = accepted_mask & valid_segment
            if not bool(valid.any().item()):
                continue
            weights = seg_mask.to(dtype=feats_for_segments.dtype).unsqueeze(-1)
            pooled = (feats_for_segments * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

            for sample_idx in torch.nonzero(valid, as_tuple=False).flatten():
                class_a = int(top2_idx[sample_idx, 0].item())
                class_b = int(top2_idx[sample_idx, 1].item())
                key, class_a, class_b = pair_key(class_a, class_b)
                if counts[class_a, seg_idx] < 2 or counts[class_b, seg_idx] < 2:
                    continue

                target_feat = pooled[sample_idx]
                actual_components = compute_pair_components(
                    target_feat,
                    prototypes,
                    variances,
                    counts,
                    seg_idx,
                    class_a,
                    class_b,
                    prototype_temperature,
                )
                if actual_components is None:
                    continue
                ambiguity = float((float(max_margin) - float(margins[sample_idx].item())) / max(float(max_margin), 1e-6))
                ambiguity = max(0.0, min(1.0, ambiguity))
                source_separability = actual_components["source_separability"]
                explainability = actual_components["target_explainability"]
                actual_raw_score = ambiguity * source_separability * explainability

                baseline_raw_scores = []
                baseline_separabilities = []
                baseline_explainabilities = []
                if shuffle_pair_baseline:
                    for base_a, base_b in sample_baseline_pairs(
                        num_classes,
                        class_a,
                        class_b,
                        baseline_pairs_per_sample,
                    ):
                        base_components = compute_pair_components(
                            target_feat,
                            prototypes,
                            variances,
                            counts,
                            seg_idx,
                            base_a,
                            base_b,
                            prototype_temperature,
                        )
                        if base_components is None:
                            continue
                        base_raw = (
                            ambiguity
                            * base_components["source_separability"]
                            * base_components["target_explainability"]
                        )
                        baseline_raw_scores.append(base_raw)
                        baseline_separabilities.append(base_components["source_separability"])
                        baseline_explainabilities.append(base_components["target_explainability"])

                if baseline_raw_scores:
                    if str(baseline_mode).lower() == "max":
                        baseline_raw_score = max(baseline_raw_scores)
                    else:
                        baseline_raw_score = float(sum(baseline_raw_scores) / len(baseline_raw_scores))
                    baseline_sep = float(sum(baseline_separabilities) / len(baseline_separabilities))
                    baseline_expl = float(sum(baseline_explainabilities) / len(baseline_explainabilities))
                else:
                    baseline_raw_score = 0.0
                    baseline_sep = 0.0
                    baseline_expl = 0.0

                _accumulate_score(
                    segment_stats[seg_idx],
                    actual_raw_score,
                    source_separability,
                    explainability,
                    ambiguity,
                    actual_raw_score=actual_raw_score,
                    baseline_raw_score=baseline_raw_score,
                    baseline_separability=baseline_sep,
                    baseline_explainability=baseline_expl,
                )
                _accumulate_score(
                    pair_stats[key],
                    actual_raw_score,
                    source_separability,
                    explainability,
                    ambiguity,
                    actual_raw_score=actual_raw_score,
                    baseline_raw_score=baseline_raw_score,
                    baseline_separability=baseline_sep,
                    baseline_explainability=baseline_expl,
                )
                segment_stats[seg_idx][f"pair_{key}_score_sum"] += actual_raw_score
                segment_stats[seg_idx][f"pair_{key}_count"] += 1.0
                _accumulate_score(
                    baseline_segment_stats[seg_idx],
                    baseline_raw_score,
                    baseline_sep,
                    baseline_expl,
                    ambiguity,
                    actual_raw_score=actual_raw_score,
                    baseline_raw_score=baseline_raw_score,
                    baseline_separability=baseline_sep,
                    baseline_explainability=baseline_expl,
                )

    segment_rows = _stats_to_segment_rows(segment_stats, phase_partition_spec, segment_count)
    baseline_rows = _stats_to_segment_rows(baseline_segment_stats, phase_partition_spec, segment_count)
    pair_rows = []
    for key, stats in pair_stats.items():
        count = max(stats.get("count", 0.0), 1.0)
        pair_rows.append(
            {
                "pair": key,
                "score": stats.get("score_sum", 0.0) / count,
                "support_count": int(stats.get("count", 0.0)),
                "source_separability": stats.get("source_separability_sum", 0.0) / count,
                "target_explainability": stats.get("target_explainability_sum", 0.0) / count,
                "ambiguity": stats.get("ambiguity_sum", 0.0) / count,
            }
        )

    segment_rows_sorted = sorted(segment_rows, key=lambda row: row["score"], reverse=True)
    pair_rows.sort(key=lambda row: row["score"], reverse=True)
    return {
        "seen_target_samples": seen,
        "accepted_ambiguous_samples": accepted,
        "accepted_fraction": accepted / max(seen, 1),
        "segment_rows": segment_rows_sorted,
        "segment_rows_by_index": segment_rows,
        "baseline_segment_rows": sorted(baseline_rows, key=lambda row: row["score"], reverse=True),
        "pair_rows": pair_rows,
    }


def _accumulate_score(
    store,
    score,
    source_separability,
    explainability,
    ambiguity,
    actual_raw_score=0.0,
    baseline_raw_score=0.0,
    baseline_separability=0.0,
    baseline_explainability=0.0,
):
    store["score_sum"] += score
    store["source_separability_sum"] += source_separability
    store["target_explainability_sum"] += explainability
    store["ambiguity_sum"] += ambiguity
    store["actual_raw_score_sum"] += actual_raw_score
    store["baseline_raw_score_sum"] += baseline_raw_score
    store["baseline_separability_sum"] += baseline_separability
    store["baseline_explainability_sum"] += baseline_explainability
    store["count"] += 1.0


def _best_pair_for_segment(stats):
    best_pair = ""
    best_score = 0.0
    for key, value in stats.items():
        if not key.startswith("pair_") or not key.endswith("_score_sum"):
            continue
        pair = key[len("pair_"):-len("_score_sum")]
        count = max(stats.get(f"pair_{pair}_count", 0.0), 1.0)
        score = value / count
        if score > best_score:
            best_score = score
            best_pair = pair
    return best_pair, best_score


def _stats_to_segment_rows(segment_stats, phase_partition_spec, segment_count):
    intervals = phase_partition_spec.get("intervals") or [(idx + 1, idx + 1) for idx in range(segment_count)]
    rows = []
    for seg_idx in range(segment_count):
        stats = segment_stats[seg_idx]
        count = max(stats.get("count", 0.0), 1.0)
        best_pair, best_pair_score = _best_pair_for_segment(stats)
        start, end = intervals[seg_idx] if seg_idx < len(intervals) else (seg_idx + 1, seg_idx + 1)
        actual_mean = stats.get("actual_raw_score_sum", 0.0) / count
        baseline_mean = stats.get("baseline_raw_score_sum", 0.0) / count
        relative_mean = actual_mean - baseline_mean
        ratio_mean = actual_mean / max(baseline_mean, 1e-6)
        score = max(relative_mean, 0.0)
        rows.append(
            {
                "segment": seg_idx + 1,
                "start": int(start),
                "end": int(end),
                "score": score,
                "support_count": int(stats.get("count", 0.0)),
                "source_separability": stats.get("source_separability_sum", 0.0) / count,
                "target_explainability": stats.get("target_explainability_sum", 0.0) / count,
                "ambiguity": stats.get("ambiguity_sum", 0.0) / count,
                "actual_raw_score": actual_mean,
                "baseline_raw_score": baseline_mean,
                "relative_score": relative_mean,
                "ratio_score": ratio_mean,
                "baseline_separability": stats.get("baseline_separability_sum", 0.0) / count,
                "baseline_explainability": stats.get("baseline_explainability_sum", 0.0) / count,
                "best_pair": best_pair,
                "best_pair_score": best_pair_score,
            }
        )
    return rows


def construct_adaptive_segments(rows_by_index, score_quantile, min_score, min_ratio):
    positive_scores = [
        row["score"]
        for row in rows_by_index
        if row["score"] > 0.0 and row["ratio_score"] >= float(min_ratio) and row["support_count"] > 0
    ]
    if positive_scores:
        threshold = float(np.quantile(np.asarray(positive_scores), float(score_quantile)))
        threshold = max(threshold, float(min_score))
    else:
        threshold = float(min_score)

    segments = []
    current = None
    for row in rows_by_index:
        keep = (
            row["score"] >= threshold
            and row["ratio_score"] >= float(min_ratio)
            and row["support_count"] > 0
        )
        if not keep:
            if current is not None:
                segments.append(current)
                current = None
            continue
        if current is None:
            current = {
                "start": row["start"],
                "end": row["end"],
                "atomic_segments": [row["segment"]],
                "score_sum": row["score"],
                "support_count": row["support_count"],
                "source_separability_sum": row["source_separability"],
                "target_explainability_sum": row["target_explainability"],
                "ambiguity_sum": row["ambiguity"],
                "actual_raw_score_sum": row["actual_raw_score"],
                "baseline_raw_score_sum": row["baseline_raw_score"],
                "relative_score_sum": row["relative_score"],
                "ratio_score_sum": row["ratio_score"],
                "best_pairs": [row["best_pair"]] if row["best_pair"] else [],
            }
        else:
            current["end"] = row["end"]
            current["atomic_segments"].append(row["segment"])
            current["score_sum"] += row["score"]
            current["support_count"] += row["support_count"]
            current["source_separability_sum"] += row["source_separability"]
            current["target_explainability_sum"] += row["target_explainability"]
            current["ambiguity_sum"] += row["ambiguity"]
            current["actual_raw_score_sum"] += row["actual_raw_score"]
            current["baseline_raw_score_sum"] += row["baseline_raw_score"]
            current["relative_score_sum"] += row["relative_score"]
            current["ratio_score_sum"] += row["ratio_score"]
            if row["best_pair"]:
                current["best_pairs"].append(row["best_pair"])
    if current is not None:
        segments.append(current)

    merged = []
    for segment in segments:
        n = max(len(segment["atomic_segments"]), 1)
        pair_counts = defaultdict(int)
        for pair in segment["best_pairs"]:
            pair_counts[pair] += 1
        best_pair = max(pair_counts.items(), key=lambda item: item[1])[0] if pair_counts else ""
        merged.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "atomic_segments": segment["atomic_segments"],
                "score": segment["score_sum"] / n,
                "support_count": int(segment["support_count"]),
                "source_separability": segment["source_separability_sum"] / n,
                "target_explainability": segment["target_explainability_sum"] / n,
                "ambiguity": segment["ambiguity_sum"] / n,
                "actual_raw_score": segment["actual_raw_score_sum"] / n,
                "baseline_raw_score": segment["baseline_raw_score_sum"] / n,
                "relative_score": segment["relative_score_sum"] / n,
                "ratio_score": segment["ratio_score_sum"] / n,
                "dominant_pair": best_pair,
            }
        )
    merged.sort(key=lambda row: row["score"], reverse=True)
    return threshold, merged


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    config = load_config(args.run_dir, args.data_root, args.device, args.seed)
    seed = int(getattr(config, "seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    splits = make_splits(config)
    source_dataset = create_diagnostic_dataset(config, config.source, splits)
    target_dataset = create_diagnostic_dataset(config, config.target, splits)
    source_loader = create_diagnostic_loader(source_dataset, config)
    target_loader = create_diagnostic_loader(target_dataset, config)

    checkpoint_path = args.checkpoint or os.path.join(args.run_dir, "fold_0", "model.pt")
    model, reshaper = load_model_and_reshaper(config, checkpoint_path, device)

    phase_partition_spec = build_atomic_partition_spec(source_dataset.date_positions, args.atomic_bins)
    print("atomic partition:", describe_atomic_partition_spec(phase_partition_spec))

    if args.target_to_source_shift is None:
        max_shift = int(getattr(config, "max_temporal_shift", 60))
        shift = estimate_temporal_shift(
            model,
            target_loader,
            device,
            min_shift=-max_shift,
            max_shift=max_shift,
            sample_size=args.shift_sample_size,
            shift_estimator="IS",
        )
    else:
        shift = int(args.target_to_source_shift)
    print(f"target_to_source_shift={shift}")

    prototypes, variances, counts = compute_source_segment_prototypes(
        model,
        reshaper,
        source_loader,
        phase_partition_spec,
        config.num_classes,
        device,
        max_batches=args.source_max_batches,
        apply_reshaper=bool(args.apply_source_reshaper),
    )
    discovery = discover_target_segments(
        model,
        reshaper if bool(args.apply_source_reshaper) else None,
        target_loader,
        prototypes,
        variances,
        counts,
        phase_partition_spec,
        config,
        device,
        target_to_source_shift=shift,
        max_batches=args.target_max_batches,
        max_margin=args.max_margin,
        min_top2_mass=args.min_top2_mass,
        prototype_temperature=args.prototype_temperature,
        shift_jitter=args.shift_jitter,
        shuffle_pair_baseline=bool(args.shuffle_pair_baseline),
        baseline_pairs_per_sample=args.baseline_pairs_per_sample,
        baseline_mode=args.baseline_mode,
    )
    threshold, adaptive_segments = construct_adaptive_segments(
        discovery["segment_rows_by_index"],
        score_quantile=args.segment_score_quantile,
        min_score=args.min_segment_score,
        min_ratio=args.min_segment_ratio,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "source": config.source,
        "target": config.target,
        "run_dir": args.run_dir,
        "checkpoint": checkpoint_path,
        "target_to_source_shift": int(shift),
        "segment_partition": phase_partition_spec,
        "accepted_fraction": discovery["accepted_fraction"],
        "seen_target_samples": discovery["seen_target_samples"],
        "accepted_ambiguous_samples": discovery["accepted_ambiguous_samples"],
        "score_threshold": threshold,
        "min_segment_ratio": float(args.min_segment_ratio),
        "top_segments": discovery["segment_rows"][: int(args.top_k_segments)],
        "top_pairs": discovery["pair_rows"][: int(args.top_k_segments)],
        "adaptive_segments": adaptive_segments,
        "baseline_top_segments": discovery["baseline_segment_rows"][: int(args.top_k_segments)],
    }
    json_path = os.path.join(args.output_dir, "adaptive_segments.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    write_csv(
        os.path.join(args.output_dir, "atomic_bin_scores.csv"),
        discovery["segment_rows"],
        [
            "segment",
            "start",
            "end",
            "score",
            "support_count",
            "source_separability",
            "target_explainability",
            "ambiguity",
            "actual_raw_score",
            "baseline_raw_score",
            "relative_score",
            "ratio_score",
            "baseline_separability",
            "baseline_explainability",
            "best_pair",
            "best_pair_score",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "baseline_atomic_bin_scores.csv"),
        discovery["baseline_segment_rows"],
        [
            "segment",
            "start",
            "end",
            "score",
            "support_count",
            "source_separability",
            "target_explainability",
            "ambiguity",
            "actual_raw_score",
            "baseline_raw_score",
            "relative_score",
            "ratio_score",
            "baseline_separability",
            "baseline_explainability",
            "best_pair",
            "best_pair_score",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "pair_scores.csv"),
        discovery["pair_rows"],
        ["pair", "score", "support_count", "source_separability", "target_explainability", "ambiguity"],
    )
    print(f"Saved {json_path}")
    print(f"accepted_fraction={discovery['accepted_fraction']:.4f}, threshold={threshold:.4f}")
    for row in adaptive_segments[: int(args.top_k_segments)]:
        print(
            "ADAPTIVE "
            f"[{row['start']},{row['end']}]: score={row['score']:.4f}, "
            f"atoms={row['atomic_segments']}, n={row['support_count']}, pair={row['dominant_pair']}"
            f", raw={row['actual_raw_score']:.4f}, base={row['baseline_raw_score']:.4f}, ratio={row['ratio_score']:.3f}"
        )
    for row in summary["top_segments"]:
        print(
            "ATOM "
            f"{row['segment']}[{row['start']},{row['end']}]: score={row['score']:.4f}, "
            f"n={row['support_count']}, sep={row['source_separability']:.3f}, "
            f"expl={row['target_explainability']:.3f}, amb={row['ambiguity']:.3f}, "
            f"raw={row['actual_raw_score']:.4f}, base={row['baseline_raw_score']:.4f}, ratio={row['ratio_score']:.3f}"
        )


if __name__ == "__main__":
    main()
