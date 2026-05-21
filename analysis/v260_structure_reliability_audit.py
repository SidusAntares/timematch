import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils import data

from data_adapters.factory import (
    build_dataset,
    get_classes_for_config,
    get_dataset_length,
    make_eval_transform,
)
from data_adapters.har_dataset import get_adatime_input_dim
from dataset import GroupByShapesBatchSampler
from models.stclassifier import PseLTae, PseGru, PseTae, PseTempCNN
from train import create_train_val_test_folds
from utils.train_utils import to_cuda


REMOTE_DOMAINS = {
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
    "DK1": "denmark/32VNH/2017",
    "AT1": "austria/33UVP/2017",
}

REMOTE_TASKS_12 = [
    ("FR1", "FR2"),
    ("FR1", "DK1"),
    ("FR1", "AT1"),
    ("FR2", "FR1"),
    ("FR2", "DK1"),
    ("FR2", "AT1"),
    ("DK1", "FR1"),
    ("DK1", "FR2"),
    ("DK1", "AT1"),
    ("AT1", "FR1"),
    ("AT1", "FR2"),
    ("AT1", "DK1"),
]

HAR_TASKS = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16")]
HHAR_TASKS = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5")]


def parse_task(text):
    if "->" in text:
        source, target = text.split("->", 1)
    elif ":" in text:
        source, target = text.split(":", 1)
    else:
        raise ValueError(f"Task must look like SRC->TGT, got: {text}")
    return source.strip(), target.strip()


def expand_tasks(args):
    if args.source and args.target:
        return [(args.dataset_name, args.source, args.target)]
    if args.task:
        source, target = parse_task(args.task)
        return [(args.dataset_name, source, target)]

    tasks = []
    presets = args.preset.split(",")
    for preset in presets:
        preset = preset.strip().lower()
        if preset == "remote_quick":
            tasks.extend([("remote", "FR1", "FR2"), ("remote", "FR2", "FR1"), ("remote", "DK1", "FR1"), ("remote", "FR2", "DK1")])
        elif preset == "remote12":
            tasks.extend(("remote", s, t) for s, t in REMOTE_TASKS_12)
        elif preset == "har":
            tasks.extend(("HAR", s, t) for s, t in HAR_TASKS)
        elif preset == "hhar":
            tasks.extend(("HHAR_SA", s, t) for s, t in HHAR_TASKS)
        elif preset == "har_hhar":
            tasks.extend(("HAR", s, t) for s, t in HAR_TASKS)
            tasks.extend(("HHAR_SA", s, t) for s, t in HHAR_TASKS)
        elif preset == "full":
            tasks.extend(("remote", s, t) for s, t in REMOTE_TASKS_12)
            tasks.extend(("HAR", s, t) for s, t in HAR_TASKS)
            tasks.extend(("HHAR_SA", s, t) for s, t in HHAR_TASKS)
        else:
            raise ValueError(f"Unsupported preset: {preset}")
    return tasks


def resolve_dataset_config(dataset_name, source, target, args):
    if dataset_name.lower() == "remote":
        source_name = REMOTE_DOMAINS.get(source, source)
        target_name = REMOTE_DOMAINS.get(target, target)
        return SimpleNamespace(
            dataset_label="remote",
            dataset_type="remote_sensing",
            data_root=args.remote_data_root,
            har_dataset_name="HAR",
            source=source_name,
            target=target_name,
            task=f"{source}->{target}",
            input_dim=10,
            num_pixels=args.remote_num_pixels,
            seq_length=args.remote_seq_length,
            closed_set=True,
            combine_spring_and_winter=False,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            num_folds=1,
            sample_pixels_val=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model=args.model,
            with_extra=False,
            har_label_offset="auto",
            source_feature_reshaper="none",
        )

    dataset = dataset_name.upper()
    data_root = args.har_data_root if dataset == "HAR" else args.hhar_data_root
    input_dim = get_adatime_input_dim(dataset)
    return SimpleNamespace(
        dataset_label=dataset,
        dataset_type="har",
        data_root=data_root,
        har_dataset_name=dataset,
        source=str(source),
        target=str(target),
        task=f"{source}->{target}",
        input_dim=input_dim,
        num_pixels=1,
        seq_length=args.har_seq_length,
        closed_set=True,
        combine_spring_and_winter=False,
        val_ratio=args.val_ratio,
        test_ratio=0.0,
        num_folds=1,
        sample_pixels_val=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=args.model,
        with_extra=False,
        har_label_offset="auto",
        source_feature_reshaper="none",
    )


def build_model(config, num_classes, device):
    kwargs = dict(input_dim=config.input_dim, num_classes=num_classes, with_extra=config.with_extra)
    if config.model == "pseltae":
        model = PseLTae(**kwargs)
    elif config.model == "psetae":
        model = PseTae(**kwargs)
    elif config.model == "psetcnn":
        model = PseTempCNN(**kwargs)
    elif config.model == "psegru":
        model = PseGru(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {config.model}")
    model.to(device)
    return model


def maybe_load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "fold_0" / "model.pt"
    if not path.exists():
        print(f"[warn] checkpoint not found, using raw_input mode: {path}")
        return False
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {path}")
    return True


def normalize_token(text):
    return str(text).lower().replace("/", "_").replace("\\", "_").replace("-", "_")


def checkpoint_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def checkpoint_is_compatible(path, config):
    try:
        state_dict = checkpoint_state_dict(path)
    except Exception as exc:
        print(f"[warn] skip unreadable checkpoint: {path} ({exc})")
        return False

    pse_weight = state_dict.get("spatial_encoder.mlp1.0.linear.weight")
    decoder_weight = state_dict.get("decoder.2.weight")
    if pse_weight is not None and int(pse_weight.shape[1]) != int(config.input_dim):
        return False
    if decoder_weight is not None and int(decoder_weight.shape[0]) != int(config.num_classes):
        return False
    return True


def remote_tile_token(domain):
    parts = str(domain).split("/")
    return normalize_token(parts[1]) if len(parts) >= 2 else normalize_token(domain)


def checkpoint_name_score(path, config, source_alias, target_alias, args):
    text = normalize_token(path)
    source_alias = normalize_token(source_alias)
    target_alias = normalize_token(target_alias)

    if config.dataset_label == "remote":
        source_tile = remote_tile_token(config.source)
        target_tile = remote_tile_token(config.target)
        source_patterns = [
            f"pseltae_{source_tile}",
            f"timematch_{source_tile}_to_",
        ]
        if not any(pattern in text for pattern in source_patterns):
            return None
        target_hit = f"_to_{target_tile}" in text or target_alias in text
    elif config.dataset_label == "HAR":
        source_patterns = [
            f"har_{source_alias}_to_",
            f"har_baseline_{source_alias}_to_",
        ]
        if not any(pattern in text for pattern in source_patterns) or "hhar" in text:
            return None
        target_hit = (
            f"har_{source_alias}_to_{target_alias}" in text
            or f"har_baseline_{source_alias}_to_{target_alias}" in text
        )
    elif config.dataset_label == "HHAR_SA":
        source_patterns = [
            f"hhar_sa_{source_alias}_to_",
            f"hhar_sa_baseline_{source_alias}_to_",
        ]
        if not any(pattern in text for pattern in source_patterns):
            return None
        target_hit = (
            f"hhar_sa_{source_alias}_to_{target_alias}" in text
            or f"hhar_sa_baseline_{source_alias}_to_{target_alias}" in text
        )
    else:
        return None

    score = 0.0
    if "source" in text:
        score += 10
    if "baseline" in text:
        score += 4
    if "sourcephasecompact" in text or "compact" in text or "v243" in text or "v244" in text:
        score += 2
    if target_hit:
        score += 8 if args.checkpoint_match_target else 1
    elif args.checkpoint_match_target:
        return None
    # Prefer shorter, more direct paths after semantic scoring.
    score -= min(len(str(path)), 500) / 100000.0
    return score


def find_task_checkpoint(args, config, source_alias, target_alias):
    if not args.checkpoint_root:
        return None
    root = Path(args.checkpoint_root)
    if not root.exists():
        print(f"[warn] checkpoint_root not found: {root}")
        return None

    candidates = []
    for path in root.rglob("model.pt"):
        score = checkpoint_name_score(path, config, source_alias, target_alias, args)
        if score is None:
            continue
        if not checkpoint_is_compatible(path, config):
            continue
        candidates.append((score, path))

    if not candidates:
        print(f"[warn] no checkpoint candidate found for {config.dataset_label} {config.task} under {root}")
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = candidates[0][1]
    print(f"Auto checkpoint for {config.dataset_label} {config.task}: {chosen}")
    return str(chosen)


def make_loader(config, dataset, shuffle=False):
    if config.dataset_type == "har":
        return data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return data.DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(dataset, config.batch_size),
        pin_memory=torch.cuda.is_available(),
    )


def extract_trajectories(model, loader, device, feature_mode, max_samples):
    feats = []
    labels = []
    positions_ref = None
    seen = 0
    model.eval()
    with torch.no_grad():
        for sample in loader:
            pixels, mask, positions, extra = to_cuda(sample, device)
            batch_labels = sample["label"].cpu().numpy().astype(np.int64)
            if feature_mode == "pse":
                values = model.spatial_encoder(pixels, mask, extra)
            else:
                valid = mask.float().unsqueeze(2)
                denom = valid.sum(dim=-1).clamp_min(1.0)
                values = (pixels.float() * valid).sum(dim=-1) / denom
            values = values.detach().cpu().numpy().astype(np.float32)
            pos = positions.detach().cpu().numpy().astype(np.float32)
            if positions_ref is None:
                positions_ref = pos[0]
            feats.append(values)
            labels.append(batch_labels)
            seen += values.shape[0]
            if max_samples and seen >= max_samples:
                break
    if not feats:
        raise RuntimeError("No features extracted")
    features = np.concatenate(feats, axis=0)[:max_samples]
    y = np.concatenate(labels, axis=0)[:max_samples]
    return features, y, positions_ref


def resample_trajectory(x, length):
    if x.shape[0] == length:
        return x
    old = np.linspace(0.0, 1.0, x.shape[0])
    new = np.linspace(0.0, 1.0, length)
    out = np.empty((length, x.shape[1]), dtype=np.float32)
    for dim in range(x.shape[1]):
        out[:, dim] = np.interp(new, old, x[:, dim])
    return out


def pooled(x, mode="mean"):
    if mode == "mean":
        return x.mean(axis=0)
    if mode == "meanmax":
        return np.concatenate([x.mean(axis=0), x.max(axis=0)])
    raise ValueError(mode)


def prototypes_by_class(features, labels, num_classes):
    protos = []
    counts = []
    for cls in range(num_classes):
        cls_feats = features[labels == cls]
        counts.append(int(cls_feats.shape[0]))
        if cls_feats.shape[0] == 0:
            protos.append(np.full(features.shape[1:], np.nan, dtype=np.float32))
        else:
            protos.append(cls_feats.mean(axis=0))
    return np.stack(protos, axis=0), np.asarray(counts)


def pairwise_dist(a, b):
    return float(np.linalg.norm(a - b))


def safe_mean(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def source_reliability_metrics(source_features, labels, num_classes):
    protos, counts = prototypes_by_class(source_features, labels, num_classes)
    compact = []
    for cls in range(num_classes):
        cls_feats = source_features[labels == cls]
        if cls_feats.shape[0] == 0 or not np.isfinite(protos[cls]).all():
            continue
        proto_pool = pooled(protos[cls], "mean")
        compact.extend(np.linalg.norm(np.asarray([pooled(x, "mean") for x in cls_feats]) - proto_pool, axis=1).tolist())

    sep = []
    valid = [idx for idx in range(num_classes) if counts[idx] > 0 and np.isfinite(protos[idx]).all()]
    for i, cls_i in enumerate(valid):
        for cls_j in valid[i + 1 :]:
            sep.append(pairwise_dist(pooled(protos[cls_i], "mean"), pooled(protos[cls_j], "mean")))

    compact_mean = safe_mean(compact)
    sep_mean = safe_mean(sep)
    reliability = sep_mean / (compact_mean + 1e-6) if np.isfinite(compact_mean) and np.isfinite(sep_mean) else float("nan")

    radius_by_t = []
    for t in range(source_features.shape[1]):
        vals = []
        for cls in valid:
            cls_feats = source_features[labels == cls, t, :]
            if cls_feats.shape[0] > 1:
                vals.append(float(np.linalg.norm(cls_feats - protos[cls, t], axis=1).mean()))
        radius_by_t.append(safe_mean(vals))
    radius_by_t = np.asarray(radius_by_t, dtype=np.float32)
    phase_cv = float(np.nanstd(radius_by_t) / (np.nanmean(radius_by_t) + 1e-6))
    late_early = float(np.nanmean(radius_by_t[len(radius_by_t) // 2 :]) / (np.nanmean(radius_by_t[: max(1, len(radius_by_t) // 2)]) + 1e-6))

    return {
        "source_compactness": compact_mean,
        "source_separability": sep_mean,
        "source_reliability": reliability,
        "source_radius_phase_cv": phase_cv,
        "source_late_early_radius_ratio": late_early,
    }, protos, counts


def target_explainability_metrics(target_features, source_protos):
    length = target_features.shape[1]
    protos = np.stack([resample_trajectory(proto, length) for proto in source_protos if np.isfinite(proto).all()], axis=0)
    if protos.shape[0] < 2:
        return {"target_nearest_distance": float("nan"), "target_margin": float("nan"), "target_margin_ratio": float("nan"), "target_assignment_entropy": float("nan")}
    proto_pooled = np.asarray([pooled(proto, "mean") for proto in protos])
    target_pooled = np.asarray([pooled(x, "mean") for x in target_features])
    distances = np.linalg.norm(target_pooled[:, None, :] - proto_pooled[None, :, :], axis=-1)
    sorted_dist = np.sort(distances, axis=1)
    nearest = sorted_dist[:, 0]
    second = sorted_dist[:, 1]
    margin = second - nearest
    margin_ratio = margin / (second + 1e-6)
    assignments = distances.argmin(axis=1)
    counts = np.bincount(assignments, minlength=protos.shape[0]).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)) / math.log(max(len(probs), 2)))
    return {
        "target_nearest_distance": float(np.mean(nearest)),
        "target_margin": float(np.mean(margin)),
        "target_margin_ratio": float(np.mean(margin_ratio)),
        "target_assignment_entropy": entropy,
    }


def mismatch_metrics(source_features, target_features):
    length = min(source_features.shape[1], target_features.shape[1])
    src_mean = resample_trajectory(source_features.mean(axis=0), length)
    tgt_mean = resample_trajectory(target_features.mean(axis=0), length)
    curve = np.linalg.norm(src_mean - tgt_mean, axis=1)
    mean = float(np.mean(curve))
    std = float(np.std(curve))
    cv = std / (mean + 1e-6)
    burst = float(np.max(curve) / (mean + 1e-6))
    topk = max(1, int(round(0.2 * len(curve))))
    top_ratio = float(np.sort(curve)[-topk:].sum() / (curve.sum() + 1e-6))
    compression = {}
    for k in (3, 5, 8):
        chunks = np.array_split(curve, k)
        within = sum(len(chunk) / len(curve) * float(np.var(chunk)) for chunk in chunks)
        global_var = float(np.var(curve))
        comp = 1.0 - within / (global_var + 1e-6)
        compression[f"uniform_k{k}_compression"] = float(comp)
        compression[f"uniform_k{k}_adjusted_compression"] = float(comp / math.sqrt(k))
    return {
        "temporal_mismatch_mean": mean,
        "temporal_mismatch_std": std,
        "temporal_mismatch_cv": float(cv),
        "temporal_mismatch_burstiness": burst,
        "temporal_mismatch_top20_ratio": top_ratio,
        **compression,
    }


def dynamics_metrics(source_features, target_features):
    length = min(source_features.shape[1], target_features.shape[1])
    src = resample_trajectory(source_features.mean(axis=0), length)
    tgt = resample_trajectory(target_features.mean(axis=0), length)
    src_delta = np.diff(src, axis=0)
    tgt_delta = np.diff(tgt, axis=0)
    denom = np.linalg.norm(src_delta, axis=1) * np.linalg.norm(tgt_delta, axis=1) + 1e-6
    cosine = np.sum(src_delta * tgt_delta, axis=1) / denom
    return {
        "dynamics_delta_cosine_mean": float(np.mean(cosine)),
        "dynamics_delta_cosine_std": float(np.std(cosine)),
        "source_delta_activity": float(np.linalg.norm(src_delta, axis=1).mean()),
        "target_delta_activity": float(np.linalg.norm(tgt_delta, axis=1).mean()),
    }


def view_scores(metrics):
    reliability = metrics.get("source_reliability", 0.0)
    margin = metrics.get("target_margin_ratio", 0.0)
    entropy_health = 1.0 - abs(metrics.get("target_assignment_entropy", 1.0) - 0.75)
    base = np.nan_to_num(reliability, nan=0.0) * max(np.nan_to_num(margin, nan=0.0), 0.0) * max(entropy_health, 0.0)
    global_score = base / (1.0 + max(metrics.get("temporal_mismatch_cv", 0.0), 0.0))
    seg_score = base * max(metrics.get("uniform_k5_adjusted_compression", 0.0), 0.0)
    dyn_score = base * max((metrics.get("dynamics_delta_cosine_mean", 0.0) + 1.0) / 2.0, 0.0)
    scores = {
        "view_global_score": float(global_score),
        "view_segmented_score": float(seg_score),
        "view_dynamics_score": float(dyn_score),
    }
    best = max(scores, key=scores.get)
    q = max(scores.values())
    scores["suggested_view"] = best.replace("view_", "").replace("_score", "")
    factor = float(np.clip(1.0 + 0.25 * (q - 0.5), 0.7, 1.2))
    scores["suggested_strength_factor"] = factor
    return scores


def audit_one(dataset_name, source, target, args):
    config = resolve_dataset_config(dataset_name, source, target, args)
    config.classes = get_classes_for_config(config)
    config.num_classes = len(config.classes)

    indices = {
        config.source: get_dataset_length(config, config.source, split="train"),
        config.target: get_dataset_length(config, config.target, split="train"),
    }
    splits = create_train_val_test_folds([config.source, config.target], 1, indices, config.val_ratio, config.test_ratio)[0]

    transform = make_eval_transform(config, sample_pixels=(config.dataset_type != "har"), sample_time=False)
    source_dataset = build_dataset(config, config.source, config.classes, transform, indices=splits[config.source]["train"], split="train")
    target_dataset = build_dataset(config, config.target, config.classes, transform, indices=splits[config.target]["train"], split="train")
    source_loader = make_loader(config, source_dataset)
    target_loader = make_loader(config, target_dataset)

    device = torch.device(args.device)
    model = build_model(config, config.num_classes, device)
    checkpoint_path = args.checkpoint or find_task_checkpoint(args, config, source, target)
    checkpoint_loaded = maybe_load_checkpoint(model, checkpoint_path, device)
    feature_mode = args.feature_mode
    if feature_mode == "auto":
        feature_mode = "pse" if checkpoint_loaded else "raw_input"
    if feature_mode == "pse" and not checkpoint_loaded:
        raise FileNotFoundError(
            f"feature_mode=pse requires a checkpoint for {config.dataset_label} {config.task}. "
            "Pass --checkpoint or --checkpoint_root, or use --feature_mode auto/raw_input."
        )

    source_features, labels, _ = extract_trajectories(model, source_loader, device, feature_mode, args.max_source_samples)
    target_features, _, _ = extract_trajectories(model, target_loader, device, feature_mode, args.max_target_samples)

    source_metrics, source_protos, counts = source_reliability_metrics(source_features, labels, config.num_classes)
    metrics = {
        "dataset": config.dataset_label,
        "task": config.task,
        "source": config.source,
        "target": config.target,
        "feature_mode": feature_mode,
        "checkpoint_path": checkpoint_path or "",
        "checkpoint_loaded": checkpoint_loaded,
        "num_classes": config.num_classes,
        "source_samples": int(source_features.shape[0]),
        "target_samples": int(target_features.shape[0]),
        "sequence_length_source": int(source_features.shape[1]),
        "sequence_length_target": int(target_features.shape[1]),
        "feature_dim": int(source_features.shape[2]),
        "min_source_class_count": int(counts.min()) if len(counts) else 0,
        **source_metrics,
        **target_explainability_metrics(target_features, source_protos),
        **mismatch_metrics(source_features, target_features),
        **dynamics_metrics(source_features, target_features),
    }
    metrics.update(view_scores(metrics))
    return metrics


def write_outputs(rows, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    write_markdown(md_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def write_markdown(path, rows):
    headers = ["dataset", "task", "feature", "src_rel", "tgt_margin", "mismatch_cv", "segK5_adj", "dyn_cos", "suggested", "factor"]
    lines = [
        "# v2.6.0 Structure Reliability Audit",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("task", "")),
                    str(row.get("feature_mode", "")),
                    fmt(row.get("source_reliability")),
                    fmt(row.get("target_margin_ratio")),
                    fmt(row.get("temporal_mismatch_cv")),
                    fmt(row.get("uniform_k5_adjusted_compression")),
                    fmt(row.get("dynamics_delta_cosine_mean")),
                    str(row.get("suggested_view", "")),
                    fmt(row.get("suggested_strength_factor")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def collect_outputs(args):
    paths = sorted(Path(args.collect_dir).glob("**/*_single.json"))
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows.extend(data)
        else:
            rows.append(data)
    write_outputs(rows, Path(args.output_dir), args.output_stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="remote_quick", help="remote_quick, remote12, har, hhar, har_hhar, full, comma-separated ok")
    parser.add_argument("--dataset_name", default="remote")
    parser.add_argument("--task", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--remote_data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--har_data_root", default="/data/user/dataset/UCIHAR/HAR")
    parser.add_argument("--hhar_data_root", default="/data/user/dataset/HHAR/HHAR_SA")
    parser.add_argument("--output_dir", default="result/_summary/v260_structure_reliability_audit")
    parser.add_argument("--output_stem", default="v260_structure_reliability_audit")
    parser.add_argument("--collect_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default=None, help="Recursively search this root for task source model.pt checkpoints")
    parser.add_argument("--checkpoint_match_target", action="store_true", help="Prefer checkpoints whose path also contains the target token")
    parser.add_argument("--feature_mode", default="auto", choices=["auto", "raw_input", "pse"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="pseltae", choices=["pseltae", "psetae", "psetcnn", "psegru"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--remote_num_pixels", type=int, default=64)
    parser.add_argument("--remote_seq_length", type=int, default=30)
    parser.add_argument("--har_seq_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--max_source_samples", type=int, default=2048)
    parser.add_argument("--max_target_samples", type=int, default=2048)
    args = parser.parse_args()

    if args.collect_dir:
        collect_outputs(args)
        return

    rows = []
    for dataset_name, source, target in expand_tasks(args):
        print(f"Auditing {dataset_name} {source}->{target}")
        row = audit_one(dataset_name, source, target, args)
        rows.append(row)
        single_dir = Path(args.output_dir) / "single"
        single_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{row['dataset']}_{row['task'].replace('->', '_to_')}".replace("/", "_")
        (single_dir / f"{tag}_single.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    write_outputs(rows, Path(args.output_dir), args.output_stem)


if __name__ == "__main__":
    main()
