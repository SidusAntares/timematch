#!/usr/bin/env python3
import argparse
import csv
import math
import os
import pickle as pkl
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


TASK_ORDER = [
    "FR1_to_FR2",
    "FR1_to_DK1",
    "FR1_to_AT1",
    "FR2_to_FR1",
    "FR2_to_DK1",
    "FR2_to_AT1",
    "DK1_to_FR1",
    "DK1_to_FR2",
    "DK1_to_AT1",
    "AT1_to_FR1",
    "AT1_to_FR2",
    "AT1_to_DK1",
]

TAG_TO_DATASET = {
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
    "DK1": "denmark/32VNH/2017",
    "AT1": "austria/33UVP/2017",
}

RESTRICTED_CONFIGS = ["plain", "raw1", "timepoint1", "smooth_k3", "elastic_r2"]

CONFIG_FAMILY = {
    "plain": "off_or_weak",
    "raw1": "global_compact",
    "timepoint1": "high_rigidity_shape",
    "smooth_k3": "default_smooth",
    "elastic_r2": "loose_elastic",
}

ACTION_RULES = [
    ("elastic_help", "elastic_r2"),
    ("raw_help", "raw1"),
    ("timepoint_help", "timepoint1"),
    ("weak_help", "plain"),
]

OUTCOME_FIELDS = [
    "smooth_gain_vs_plain",
    "raw_gain_vs_plain",
    "timepoint_gain_vs_plain",
    "elastic_r2_gain_vs_smooth",
    "restricted_oracle_gain_vs_smooth",
    "elastic_help",
    "raw_help",
    "timepoint_help",
    "structure_help",
    "weak_help",
    "weak_best",
    "global_best",
    "shape_best",
    "smooth_best",
    "elastic_best",
]

IDENTITY_FIELDS = [
    "source_AT",
    "source_DK",
    "source_FR",
    "target_AT",
    "target_DK",
    "target_FR",
    "same_country_flag",
    "france_pair_flag",
    "same_source_family_flag",
]

NA = None
EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract stable descriptors for v287 regime/state diagnostics without training."
    )
    parser.add_argument("--v285_dir", default="logs/v285_regime_diagnostic_20260627_201553")
    parser.add_argument("--v286_dir", default="logs/v286_regime_state_routing_diagnostic_20260627_225007")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--data_root", default="/data/user/DBL/timematch_data")
    parser.add_argument("--closed_set", default=True, type=bool_flag)
    parser.add_argument("--max_raw_samples", default=512, type=int)
    parser.add_argument("--max_mmd_samples", default=128, type=int)
    parser.add_argument("--max_shift", default=3, type=int)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument("--allow_identity_rules", default=False, type=bool_flag)
    return parser.parse_args()


def bool_flag(value):
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def safe_float(value):
    if value in (None, "", "NA"):
        return None
    try:
        if isinstance(value, bool):
            return float(value)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def fmt_md(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_task(task):
    source, target = task.split("_to_")
    return source, target


def domain_family(tag):
    if tag.startswith("FR"):
        return "FR"
    if tag.startswith("DK"):
        return "DK"
    if tag.startswith("AT"):
        return "AT"
    return tag


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_md_table(handle, rows, fields, headers=None, limit=None):
    rows = rows[:limit] if limit else rows
    headers = headers or fields
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("|" + "|".join("---" for _ in headers) + "|\n")
    for row in rows:
        handle.write("| " + " | ".join(fmt_md(row.get(field)) for field in fields) + " |\n")
    handle.write("\n")


def mean(values):
    vals = [safe_float(value) for value in values]
    vals = [value for value in vals if value is not None]
    return None if not vals else sum(vals) / len(vals)


def delta(left, right):
    left = safe_float(left)
    right = safe_float(right)
    if left is None or right is None:
        return None
    return left - right


def ranks(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def pearson(xs, ys):
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def spearman(xs, ys):
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    rx = ranks([x for x, _ in pairs])
    ry = ranks([y for _, y in pairs])
    return pearson(rx, ry)


def load_input_rows(v285_dir, v286_dir):
    files = {
        "v285_task_config_matrix": Path(v285_dir) / "task_config_matrix.tsv",
        "v285_task_oracle_summary": Path(v285_dir) / "task_oracle_summary.tsv",
        "v285_pair_regime_descriptor": Path(v285_dir) / "pair_regime_descriptor.tsv",
        "v286_state_candidate_matrix": Path(v286_dir) / "state_candidate_matrix.tsv",
        "v286_restricted_oracle_summary": Path(v286_dir) / "restricted_oracle_summary.tsv",
        "v286_source_state_router_sim": Path(v286_dir) / "source_state_router_sim.tsv",
        "v286_target_state_router_sim": Path(v286_dir) / "target_state_router_sim.tsv",
        "v286_descriptor_rule_router_sim": Path(v286_dir) / "descriptor_rule_router_sim.tsv",
        "v286_leave_one_source_eval": Path(v286_dir) / "leave_one_source_eval.tsv",
        "v286_leave_one_target_eval": Path(v286_dir) / "leave_one_target_eval.tsv",
    }
    loaded = {}
    coverage = []
    for name, path in files.items():
        rows = read_tsv(path)
        loaded[name] = rows
        coverage.append(
            {
                "file": str(path),
                "rows": len(rows),
                "status": "ok" if rows else "missing_or_empty",
            }
        )
    return loaded, coverage


def build_outcomes(loaded):
    rows = loaded.get("v286_restricted_oracle_summary") or loaded.get("v285_task_oracle_summary") or []
    by_task = {row.get("task"): row for row in rows}
    out = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        row = by_task.get(task, {})
        plain = safe_float(row.get("plain_da"))
        raw = safe_float(row.get("raw1_da"))
        timepoint = safe_float(row.get("timepoint1_da"))
        smooth = safe_float(row.get("smooth_k3_da"))
        elastic = safe_float(row.get("elastic_r2_da"))
        values = {
            "plain": plain,
            "raw1": raw,
            "timepoint1": timepoint,
            "smooth_k3": smooth,
            "elastic_r2": elastic,
        }
        valid = {key: value for key, value in values.items() if value is not None}
        best_config = row.get("best_restricted_config") or (max(valid, key=valid.get) if valid else None)
        best_da = safe_float(row.get("best_restricted_da"))
        if best_da is None and best_config is not None:
            best_da = valid.get(best_config)
        item = {
            "task": task,
            "source": source,
            "target": target,
            "plain_da": plain,
            "raw1_da": raw,
            "timepoint1_da": timepoint,
            "smooth_k3_da": smooth,
            "elastic_r2_da": elastic,
            "best_restricted_config": best_config,
            "best_restricted_da": best_da,
            "restricted_oracle_gain_vs_smooth": delta(best_da, smooth),
            "smooth_gain_vs_plain": delta(smooth, plain),
            "raw_gain_vs_plain": delta(raw, plain),
            "timepoint_gain_vs_plain": delta(timepoint, plain),
            "elastic_r2_gain_vs_smooth": delta(elastic, smooth),
            "elastic_help": bool_num(elastic is not None and smooth is not None and elastic > smooth),
            "raw_help": bool_num(raw is not None and smooth is not None and raw > smooth),
            "timepoint_help": bool_num(timepoint is not None and smooth is not None and timepoint > smooth),
            "structure_help": bool_num(valid and plain is not None and max(v for k, v in valid.items() if k != "plain") > plain),
            "weak_help": bool_num(plain is not None and smooth is not None and plain >= smooth),
            "weak_best": bool_num(best_config == "plain"),
            "global_best": bool_num(best_config == "raw1"),
            "shape_best": bool_num(best_config == "timepoint1"),
            "smooth_best": bool_num(best_config == "smooth_k3"),
            "elastic_best": bool_num(best_config == "elastic_r2"),
        }
        out.append(item)
    return out


def bool_num(value):
    return 1 if bool(value) else 0


def load_domain_metadata(data_root, tag):
    dataset_name = TAG_TO_DATASET[tag]
    meta_path = Path(data_root) / dataset_name / "meta" / "metadata.pkl"
    if not meta_path.exists():
        return None, f"metadata_missing:{meta_path}"
    try:
        with meta_path.open("rb") as handle:
            metadata = pkl.load(handle)
        dates = metadata.get("dates", [])
        start_date = metadata.get("start_date")
        positions = date_positions(start_date, dates)
        return {"positions": np.asarray(positions, dtype=np.float64), "metadata": metadata}, "ok"
    except Exception as exc:
        return None, f"metadata_error:{type(exc).__name__}:{exc}"


def date_positions(start_date, dates):
    import datetime as dt

    def parse(date):
        text = str(date)
        return int(text[:4]), int(text[4:6]), int(text[6:])

    if start_date is None:
        return list(range(len(dates)))
    start = dt.datetime(*parse(start_date))
    out = []
    for date in dates:
        out.append(abs((dt.datetime(*parse(date)) - start).days))
    return out


def wasserstein_1d(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    qs = np.linspace(0.0, 1.0, num=max(len(a), len(b)))
    aq = np.quantile(a, qs)
    bq = np.quantile(b, qs)
    return float(np.mean(np.abs(aq - bq)))


def metadata_descriptor_rows(data_root):
    domain_meta = {}
    availability = {}
    for tag in TAG_TO_DATASET:
        domain_meta[tag], availability[tag] = load_domain_metadata(data_root, tag)

    rows = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        src = domain_meta.get(source)
        tgt = domain_meta.get(target)
        row = {"task": task, "source": source, "target": target}
        fields = [
            "source_doy_mean",
            "target_doy_mean",
            "doy_mean_gap",
            "source_doy_std",
            "target_doy_std",
            "doy_std_gap",
            "source_doy_min",
            "source_doy_max",
            "target_doy_min",
            "target_doy_max",
            "doy_overlap_length",
            "doy_union_length",
            "doy_overlap_ratio",
            "doy_wasserstein",
            "doy_coverage_jaccard",
        ]
        for field in fields:
            row[field] = None
        if src is not None and tgt is not None:
            sp = src["positions"]
            tp = tgt["positions"]
            if len(sp) and len(tp):
                smin, smax = float(np.min(sp)), float(np.max(sp))
                tmin, tmax = float(np.min(tp)), float(np.max(tp))
                overlap = max(0.0, min(smax, tmax) - max(smin, tmin))
                union = max(smax, tmax) - min(smin, tmin)
                row.update(
                    {
                        "source_doy_mean": float(np.mean(sp)),
                        "target_doy_mean": float(np.mean(tp)),
                        "doy_mean_gap": float(abs(np.mean(sp) - np.mean(tp))),
                        "source_doy_std": float(np.std(sp)),
                        "target_doy_std": float(np.std(tp)),
                        "doy_std_gap": float(abs(np.std(sp) - np.std(tp))),
                        "source_doy_min": smin,
                        "source_doy_max": smax,
                        "target_doy_min": tmin,
                        "target_doy_max": tmax,
                        "doy_overlap_length": overlap,
                        "doy_union_length": union,
                        "doy_overlap_ratio": overlap / max(union, EPS),
                        "doy_wasserstein": wasserstein_1d(sp, tp),
                        "doy_coverage_jaccard": jaccard_bins(sp, tp),
                    }
                )
        row["metadata_status"] = f"{source}:{availability.get(source)};{target}:{availability.get(target)}"
        rows.append(row)
    return rows, availability


def jaccard_bins(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return None
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if hi <= lo:
        return 1.0
    bins = np.linspace(lo, hi, num=17)
    ah = set(np.where(np.histogram(a, bins=bins)[0] > 0)[0].tolist())
    bh = set(np.where(np.histogram(b, bins=bins)[0] > 0)[0].tolist())
    if not ah and not bh:
        return None
    return len(ah & bh) / max(len(ah | bh), 1)


def get_task_classes(data_root, source_tag, closed_set):
    try:
        from dataset import PixelSetData
        from utils import label_utils

        dataset_name = TAG_TO_DATASET[source_tag]
        country = dataset_name.split("/")[0]
        classes = label_utils.get_classes(country, combine_spring_and_winter=False)
        data = PixelSetData(data_root, dataset_name, classes, transform=None, closed_set=closed_set)
        labels, counts = np.unique(data.get_labels(), return_counts=True)
        classes = [classes[int(label)] for label, count in zip(labels, counts) if count >= 200]
        if closed_set:
            classes = [name for name in classes if name != "unknown"]
        return classes, "ok"
    except Exception as exc:
        return [], f"class_error:{type(exc).__name__}:{exc}"


def load_raw_domain(data_root, tag, classes, closed_set, max_samples, seed, use_labels):
    try:
        import zarr
        from dataset import PixelSetData

        dataset_name = TAG_TO_DATASET[tag]
        dataset = PixelSetData(data_root, dataset_name, classes, transform=None, closed_set=closed_set)
        indices = list(range(len(dataset.samples)))
        rng = random.Random(seed + sum(ord(ch) for ch in tag))
        rng.shuffle(indices)
        indices = indices[: min(max_samples, len(indices))]
        curves = []
        labels = []
        for idx in indices:
            path, _parcel_idx, label, _extra = dataset.samples[idx]
            pixels = zarr.load(path).astype(np.float32)
            if pixels.ndim != 3:
                continue
            curve = pixels.mean(axis=-1)
            curves.append(curve)
            labels.append(int(label))
        if not curves:
            return None, None, None, "raw_empty"
        min_t = min(curve.shape[0] for curve in curves)
        min_b = min(curve.shape[1] for curve in curves)
        array = np.stack([curve[:min_t, :min_b] for curve in curves], axis=0)
        labels_arr = np.asarray(labels, dtype=np.int64) if use_labels else None
        return array, labels_arr, np.asarray(dataset.date_positions[:min_t], dtype=np.float64), "ok"
    except Exception as exc:
        return None, None, None, f"raw_error:{type(exc).__name__}:{exc}"


def smooth3(arr):
    if arr.shape[1] < 3:
        return arr.copy()
    pad = np.pad(arr, [(0, 0), (1, 1), (0, 0)], mode="edge")
    return (pad[:, :-2, :] + pad[:, 1:-1, :] + pad[:, 2:, :]) / 3.0


def fisher_ratio(features, labels):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2 or features.shape[0] < 2:
        return None
    overall = features.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls in classes:
        cls_feats = features[labels == cls]
        if len(cls_feats) == 0:
            continue
        center = cls_feats.mean(axis=0)
        between += len(cls_feats) * float(np.mean((center - overall) ** 2))
        within += float(np.mean((cls_feats - center) ** 2))
    between /= max(len(features), 1)
    within /= max(len(classes), 1)
    return between / (within + EPS)


def source_raw_descriptors(curves, labels):
    fields = source_raw_descriptor_fields()
    out = {field: None for field in fields}
    if curves is None or labels is None or len(curves) == 0:
        return out
    curves = np.asarray(curves, dtype=np.float64)
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        return out
    class_means = []
    within_terms = []
    for cls in classes:
        cls_curves = curves[labels == cls]
        center = cls_curves.mean(axis=0)
        class_means.append(center)
        within_terms.append(float(np.mean((cls_curves - center) ** 2)))
    class_means = np.stack(class_means, axis=0)
    pooled = curves.mean(axis=1)
    smoothed = smooth3(curves)
    shape = curves - curves.mean(axis=1, keepdims=True)
    out.update(
        {
            "source_temporal_activity": float(np.mean(np.diff(class_means, axis=1) ** 2)),
            "source_highfreq_ratio": float(np.mean((class_means - smooth3(class_means)) ** 2) / (np.mean(class_means ** 2) + EPS)),
            "source_raw_pooled_fisher": fisher_ratio(pooled, labels),
            "source_raw_timepoint_fisher": mean([fisher_ratio(curves[:, t, :], labels) for t in range(curves.shape[1])]),
            "source_raw_smoothed_timepoint_fisher": mean([fisher_ratio(smoothed[:, t, :], labels) for t in range(curves.shape[1])]),
            "source_raw_temporal_shape_fisher": fisher_ratio(shape.reshape(shape.shape[0], -1), labels),
            "source_class_center_curve_variance": float(np.var(class_means)),
            "source_within_class_curve_variance": mean(within_terms),
        }
    )
    return out


def source_raw_descriptor_fields():
    return [
        "source_temporal_activity",
        "source_highfreq_ratio",
        "source_raw_pooled_fisher",
        "source_raw_timepoint_fisher",
        "source_raw_smoothed_timepoint_fisher",
        "source_raw_temporal_shape_fisher",
        "source_class_center_curve_variance",
        "source_within_class_curve_variance",
        "source_shape_dependence_index",
    ]


def pair_raw_descriptor_fields():
    return [
        "pair_raw_mean_curve_distance",
        "pair_raw_best_shift",
        "pair_raw_best_shift_abs",
        "pair_raw_shifted_distance",
        "pair_raw_shift_gain",
        "pair_raw_shift_gain_ratio",
        "pair_raw_xcorr_best_shift",
        "pair_raw_xcorr_peak",
        "pair_raw_xcorr_margin",
        "pair_raw_norm_gap",
        "pair_raw_cov_trace_gap",
        "pair_raw_mean_gap",
        "pair_raw_mmd_rbf",
        "pair_raw_coral",
    ]


def pair_raw_descriptors(source_curves, target_curves, max_shift, max_mmd_samples, seed):
    out = {field: None for field in pair_raw_descriptor_fields()}
    if source_curves is None or target_curves is None:
        return out
    s = np.asarray(source_curves, dtype=np.float64)
    t = np.asarray(target_curves, dtype=np.float64)
    if len(s) == 0 or len(t) == 0:
        return out
    min_t = min(s.shape[1], t.shape[1])
    min_b = min(s.shape[2], t.shape[2])
    s = s[:, :min_t, :min_b]
    t = t[:, :min_t, :min_b]
    sm = s.mean(axis=0)
    tm = t.mean(axis=0)
    d0 = float(np.mean((sm - tm) ** 2))
    shift_rows = []
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            ss = sm[-shift:]
            tt = tm[: min_t + shift]
        elif shift > 0:
            ss = sm[: min_t - shift]
            tt = tm[shift:]
        else:
            ss = sm
            tt = tm
        if len(ss) == 0 or len(tt) == 0:
            continue
        dist = float(np.mean((ss - tt) ** 2))
        shift_rows.append((shift, dist, normalized_corr(ss, tt)))
    best = min(shift_rows, key=lambda item: item[1]) if shift_rows else (0, d0, None)
    xcorr_rows = [row for row in shift_rows if row[2] is not None]
    xcorr_rows.sort(key=lambda item: item[2], reverse=True)
    xcorr_best = xcorr_rows[0] if xcorr_rows else (None, None, None)
    xcorr_margin = None
    if len(xcorr_rows) >= 2:
        xcorr_margin = xcorr_rows[0][2] - xcorr_rows[1][2]
    sf = s.reshape(s.shape[0], -1)
    tf = t.reshape(t.shape[0], -1)
    out.update(
        {
            "pair_raw_mean_curve_distance": d0,
            "pair_raw_best_shift": best[0],
            "pair_raw_best_shift_abs": abs(best[0]),
            "pair_raw_shifted_distance": best[1],
            "pair_raw_shift_gain": d0 - best[1],
            "pair_raw_shift_gain_ratio": (d0 - best[1]) / (d0 + EPS),
            "pair_raw_xcorr_best_shift": xcorr_best[0],
            "pair_raw_xcorr_peak": xcorr_best[2],
            "pair_raw_xcorr_margin": xcorr_margin,
            "pair_raw_norm_gap": abs(float(np.linalg.norm(sf, axis=1).mean()) - float(np.linalg.norm(tf, axis=1).mean())),
            "pair_raw_cov_trace_gap": abs(cov_trace(sf) - cov_trace(tf)),
            "pair_raw_mean_gap": float(np.mean((sf.mean(axis=0) - tf.mean(axis=0)) ** 2)),
            "pair_raw_coral": coral_distance(sf, tf),
            "pair_raw_mmd_rbf": mmd_rbf(sf, tf, max_mmd_samples=max_mmd_samples, seed=seed),
        }
    )
    return out


def normalized_corr(a, b):
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= EPS:
        return None
    return float(np.dot(av, bv) / denom)


def cov_trace(features):
    if features.shape[0] < 2:
        return 0.0
    return float(np.var(features, axis=0).sum())


def coral_distance(source, target):
    if source.shape[0] < 2 or target.shape[0] < 2:
        return None
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    cs = source_centered.T @ source_centered / max(source.shape[0] - 1, 1)
    ct = target_centered.T @ target_centered / max(target.shape[0] - 1, 1)
    return float(np.mean((cs - ct) ** 2))


def mmd_rbf(source, target, max_mmd_samples, seed):
    rng = np.random.default_rng(seed)
    if source.shape[0] > max_mmd_samples:
        source = source[rng.choice(source.shape[0], size=max_mmd_samples, replace=False)]
    if target.shape[0] > max_mmd_samples:
        target = target[rng.choice(target.shape[0], size=max_mmd_samples, replace=False)]
    if source.shape[0] < 2 or target.shape[0] < 2:
        return None
    both = np.concatenate([source, target], axis=0)
    sample = both[rng.choice(both.shape[0], size=min(128, both.shape[0]), replace=False)]
    dists = pairwise_sq_dists(sample, sample)
    bandwidth = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    bandwidth = max(bandwidth, EPS)
    kxx = np.exp(-pairwise_sq_dists(source, source) / bandwidth)
    kyy = np.exp(-pairwise_sq_dists(target, target) / bandwidth)
    kxy = np.exp(-pairwise_sq_dists(source, target) / bandwidth)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def pairwise_sq_dists(a, b):
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)


def build_raw_descriptor_rows(args):
    domain_cache = {}
    class_status = {}
    raw_status = {}
    source_descriptor_by_tag = {}
    for tag in TAG_TO_DATASET:
        classes, status = get_task_classes(args.data_root, tag, args.closed_set)
        class_status[tag] = status
        if not classes:
            source_descriptor_by_tag[tag] = {field: None for field in source_raw_descriptor_fields()}
            domain_cache[(tag, "source")] = (None, None)
            raw_status[tag] = status
            continue
        source_curves, labels, _positions, raw_load_status = load_raw_domain(
            args.data_root, tag, classes, args.closed_set, args.max_raw_samples, args.seed, use_labels=True
        )
        source_descriptor_by_tag[tag] = source_raw_descriptors(source_curves, labels)
        domain_cache[(tag, "source")] = (source_curves, labels)
        raw_status[f"{tag}:source"] = raw_load_status

        target_classes = sorted(set(classes + ["unknown"]))
        target_curves, _target_labels, _target_positions, target_status = load_raw_domain(
            args.data_root, tag, target_classes, False, args.max_raw_samples, args.seed + 17, use_labels=False
        )
        domain_cache[(tag, "target_unlabeled")] = (target_curves, None)
        raw_status[f"{tag}:target_unlabeled"] = target_status

    source_rows = []
    source_values = defaultdict(list)
    for task in TASK_ORDER:
        source, _target = parse_task(task)
        desc = dict(source_descriptor_by_tag.get(source, {}))
        source_rows.append((task, desc))
        for field in source_raw_descriptor_fields():
            if field != "source_shape_dependence_index":
                source_values[field].append(desc.get(field))
    shape_inputs = [
        "source_raw_temporal_shape_fisher",
        "source_raw_timepoint_fisher",
        "source_temporal_activity",
    ]
    zscores = {field: zscore([desc.get(field) for _, desc in source_rows]) for field in shape_inputs}

    rows = []
    for row_idx, task in enumerate(TASK_ORDER):
        source, target = parse_task(task)
        row = {"task": task, "source": source, "target": target}
        source_desc = dict(source_descriptor_by_tag.get(source, {}))
        shape_parts = [zscores[field][row_idx] for field in shape_inputs if zscores[field][row_idx] is not None]
        source_desc["source_shape_dependence_index"] = sum(shape_parts) if shape_parts else None
        row.update(source_desc)
        source_curves = domain_cache.get((source, "source"), (None, None))[0]
        target_curves = domain_cache.get((target, "target_unlabeled"), (None, None))[0]
        row.update(pair_raw_descriptors(source_curves, target_curves, args.max_shift, args.max_mmd_samples, args.seed))
        row["raw_status"] = f"source:{raw_status.get(source + ':source')};target:{raw_status.get(target + ':target_unlabeled')}"
        rows.append(row)
    return rows, raw_status


def zscore(values):
    nums = [safe_float(value) for value in values]
    valid = [value for value in nums if value is not None]
    if len(valid) < 2:
        return [None for _ in nums]
    mu = sum(valid) / len(valid)
    var = sum((value - mu) ** 2 for value in valid) / len(valid)
    sd = math.sqrt(var)
    if sd <= EPS:
        return [0.0 if value is not None else None for value in nums]
    return [(value - mu) / sd if value is not None else None for value in nums]


def feature_descriptor_rows(reason="feature_descriptor_disabled_no_canonical_checkpoint_inference"):
    fields = feature_descriptor_fields()
    rows = []
    for task in TASK_ORDER:
        source, target = parse_task(task)
        row = {"task": task, "source": source, "target": target}
        for field in fields:
            row[field] = None
        row["feature_status"] = reason
        rows.append(row)
    return rows


def feature_descriptor_fields():
    return [
        "source_feat_pooled_fisher",
        "source_feat_timepoint_fisher",
        "source_feat_smoothed_timepoint_fisher",
        "source_feat_temporal_shape_fisher",
        "source_feat_temporal_activity",
        "source_feat_highfreq_ratio",
        "source_feat_cov_trace",
        "source_feat_norm",
        "pair_feat_mean_curve_distance",
        "pair_feat_best_shift",
        "pair_feat_best_shift_abs",
        "pair_feat_shifted_distance",
        "pair_feat_shift_gain",
        "pair_feat_shift_gain_ratio",
        "pair_feat_norm_gap",
        "pair_feat_cov_trace_gap",
        "pair_feat_coral",
        "pair_feat_mmd_rbf",
    ]


def merge_rows(outcomes, metadata_rows, raw_rows, feature_rows):
    meta = {row["task"]: row for row in metadata_rows}
    raw = {row["task"]: row for row in raw_rows}
    feat = {row["task"]: row for row in feature_rows}
    rows = []
    for outcome in outcomes:
        task = outcome["task"]
        source = outcome["source"]
        target = outcome["target"]
        row = dict(outcome)
        for source_row in [meta.get(task, {}), raw.get(task, {}), feat.get(task, {})]:
            for key, value in source_row.items():
                if key not in {"task", "source", "target"}:
                    row[key] = value
        sf = domain_family(source)
        tf = domain_family(target)
        row.update(
            {
                "source_AT": bool_num(sf == "AT"),
                "source_DK": bool_num(sf == "DK"),
                "source_FR": bool_num(sf == "FR"),
                "target_AT": bool_num(tf == "AT"),
                "target_DK": bool_num(tf == "DK"),
                "target_FR": bool_num(tf == "FR"),
                "same_country_flag": bool_num((source, target) in {("FR1", "FR2"), ("FR2", "FR1")}),
                "france_pair_flag": bool_num(sf == "FR" and tf == "FR"),
                "same_source_family_flag": bool_num(sf == tf),
            }
        )
        rows.append(row)
    return rows


def numeric_descriptor_fields(rows, allow_identity_rules=False):
    if not rows:
        return []
    exclude = {
        "task",
        "source",
        "target",
        "best_restricted_config",
        "best_restricted_da",
        "plain_da",
        "raw1_da",
        "timepoint1_da",
        "smooth_k3_da",
        "elastic_r2_da",
        "metadata_status",
        "raw_status",
        "feature_status",
    }
    exclude.update(OUTCOME_FIELDS)
    if not allow_identity_rules:
        exclude.update(IDENTITY_FIELDS)
    fields = []
    for field in rows[0].keys():
        if field in exclude:
            continue
        values = [safe_float(row.get(field)) for row in rows]
        if sum(value is not None for value in values) >= 3:
            fields.append(field)
    return fields


def descriptor_outcome_analysis(rows, descriptor_fields):
    out = []
    for desc in descriptor_fields:
        xs = [row.get(desc) for row in rows]
        for outcome in OUTCOME_FIELDS:
            ys = [row.get(outcome) for row in rows]
            valid_pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
            valid_pairs = [(x, y) for x, y in valid_pairs if x is not None and y is not None]
            if len(valid_pairs) < 3:
                continue
            p = pearson([x for x, _ in valid_pairs], [y for _, y in valid_pairs])
            s = spearman([x for x, _ in valid_pairs], [y for _, y in valid_pairs])
            row = {
                "descriptor": desc,
                "outcome": outcome,
                "n_valid": len(valid_pairs),
                "pearson": p,
                "spearman": s,
                "abs_pearson": abs(p) if p is not None else None,
                "abs_spearman": abs(s) if s is not None else None,
                "analysis_note": "exploratory_only_n12",
            }
            if outcome.endswith("_help") or outcome.endswith("_best") or outcome == "structure_help":
                true_values = [x for x, y in valid_pairs if y >= 0.5]
                false_values = [x for x, y in valid_pairs if y < 0.5]
                row["group_mean_true"] = mean(true_values)
                row["group_mean_false"] = mean(false_values)
                row["mean_diff"] = delta(row["group_mean_true"], row["group_mean_false"])
            else:
                row["group_mean_true"] = None
                row["group_mean_false"] = None
                row["mean_diff"] = None
            out.append(row)
    out.sort(key=lambda row: (-(row["abs_pearson"] or 0), row["descriptor"], row["outcome"]))
    return out


def make_single_rule(desc, op, threshold, target_outcome):
    return {
        "kind": "single",
        "target_outcome": target_outcome,
        "parts": [(desc, op, threshold)],
        "rule_text": f"{desc} {op} {threshold:.6g}",
    }


def make_double_rule(a, opa, ta, b, opb, tb, target_outcome):
    return {
        "kind": "double",
        "target_outcome": target_outcome,
        "parts": [(a, opa, ta), (b, opb, tb)],
        "rule_text": f"{a} {opa} {ta:.6g} AND {b} {opb} {tb:.6g}",
    }


def rule_fires(rule, row):
    for desc, op, threshold in rule["parts"]:
        value = safe_float(row.get(desc))
        if value is None:
            return False
        if op == ">" and not value > threshold:
            return False
        if op == "<" and not value < threshold:
            return False
    return True


def evaluate_rule(rule, rows, target_outcome, target_delta_field=None):
    selected = [row for row in rows if rule_fires(rule, row)]
    positives = [row for row in rows if safe_float(row.get(target_outcome)) and safe_float(row.get(target_outcome)) >= 0.5]
    true_positive = [row for row in selected if safe_float(row.get(target_outcome)) and safe_float(row.get(target_outcome)) >= 0.5]
    fp = [row for row in selected if row not in true_positive]
    fn = [row for row in positives if row not in true_positive]
    precision = len(true_positive) / len(selected) if selected else 0.0
    recall = len(true_positive) / len(positives) if positives else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    avg_delta = mean([row.get(target_delta_field) for row in selected]) if target_delta_field else None
    return {
        "n_selected": len(selected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selected_tasks": ",".join(row["task"] for row in selected),
        "false_positive_tasks": ",".join(row["task"] for row in fp),
        "false_negative_tasks": ",".join(row["task"] for row in fn),
        "avg_delta_selected": avg_delta,
    }


def quantile_thresholds(values):
    vals = sorted(safe_float(value) for value in values if safe_float(value) is not None)
    if len(vals) < 3:
        return []
    return sorted(set(float(np.quantile(vals, q)) for q in [0.25, 0.5, 0.75]))


def search_rules(rows, descriptor_fields, target_outcome, target_delta_field=None, max_pair_descriptors=16):
    candidates = []
    singles = []
    for desc in descriptor_fields:
        thresholds = quantile_thresholds([row.get(desc) for row in rows])
        for threshold in thresholds:
            for op in [">", "<"]:
                rule = make_single_rule(desc, op, threshold, target_outcome)
                metrics = evaluate_rule(rule, rows, target_outcome, target_delta_field)
                item = {**rule, **metrics}
                singles.append(item)
                candidates.append(item)
    singles.sort(key=lambda row: (row["f1"], row["precision"], row["avg_delta_selected"] or -999), reverse=True)
    top_desc = []
    for item in singles:
        desc = item["parts"][0][0]
        if desc not in top_desc:
            top_desc.append(desc)
        if len(top_desc) >= max_pair_descriptors:
            break
    for i, a in enumerate(top_desc):
        for b in top_desc[i + 1 :]:
            ats = quantile_thresholds([row.get(a) for row in rows])
            bts = quantile_thresholds([row.get(b) for row in rows])
            for ta in ats:
                for tb in bts:
                    for opa, opb in [(">", "<"), ("<", ">")]:
                        rule = make_double_rule(a, opa, ta, b, opb, tb, target_outcome)
                        metrics = evaluate_rule(rule, rows, target_outcome, target_delta_field)
                        candidates.append({**rule, **metrics})
    candidates.sort(
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            row["avg_delta_selected"] if row["avg_delta_selected"] is not None else -999,
            -row["n_selected"],
        ),
        reverse=True,
    )
    return candidates


def threshold_rule_rows(rows, descriptor_fields):
    outcome_to_delta = {
        "elastic_help": "elastic_r2_gain_vs_smooth",
        "raw_help": "raw_gain_vs_plain",
        "timepoint_help": "timepoint_gain_vs_plain",
        "weak_help": "smooth_gain_vs_plain",
    }
    out = []
    rule_id = 0
    for outcome, _config in ACTION_RULES:
        rules = search_rules(rows, descriptor_fields, outcome, outcome_to_delta.get(outcome))
        for rule in rules[:30]:
            rule_id += 1
            item = {
                "rule_id": f"rule_{rule_id:04d}",
                "target_outcome": outcome,
                "rule_text": rule["rule_text"],
                "n_selected": rule["n_selected"],
                "precision": rule["precision"],
                "recall": rule["recall"],
                "f1": rule["f1"],
                "selected_tasks": rule["selected_tasks"],
                "false_positive_tasks": rule["false_positive_tasks"],
                "false_negative_tasks": rule["false_negative_tasks"],
                "avg_delta_selected": rule["avg_delta_selected"],
            }
            out.append(item)
    return out


def best_rules_for_training(train_rows, descriptor_fields):
    outcome_to_delta = {
        "elastic_help": "elastic_r2_gain_vs_smooth",
        "raw_help": "raw_gain_vs_plain",
        "timepoint_help": "timepoint_gain_vs_plain",
        "weak_help": "smooth_gain_vs_plain",
    }
    best = []
    for outcome, config in ACTION_RULES:
        rules = search_rules(train_rows, descriptor_fields, outcome, outcome_to_delta.get(outcome))
        if not rules:
            continue
        top = rules[0]
        best.append(
            {
                "target_outcome": outcome,
                "config": config,
                "rule": top,
                "precision": top["precision"],
                "avg_delta_selected": top["avg_delta_selected"],
                "f1": top["f1"],
            }
        )
    best.sort(
        key=lambda item: (
            item["precision"],
            item["avg_delta_selected"] if item["avg_delta_selected"] is not None else -999,
            item["f1"],
        ),
        reverse=True,
    )
    return best


def apply_routing_rules(row, rules):
    fired = []
    for item in rules:
        if rule_fires(item["rule"], row):
            fired.append(item)
    if not fired:
        return "smooth_k3", "", "default=smooth_k3"
    selected = fired[0]
    return selected["config"], ";".join(item["target_outcome"] for item in fired), selected["rule"]["rule_text"]


def descriptor_routing_cv(rows, descriptor_fields):
    out = []
    folds = []
    for task in TASK_ORDER:
        folds.append(("leave_one_task", task, lambda row, task=task: row["task"] == task))
    for source in ["FR1", "FR2", "DK1", "AT1"]:
        folds.append(("leave_one_source", source, lambda row, source=source: row["source"] == source))
    for target in ["FR1", "FR2", "DK1", "AT1"]:
        folds.append(("leave_one_target", target, lambda row, target=target: row["target"] == target))

    for cv_type, heldout, is_eval in folds:
        eval_rows = [row for row in rows if is_eval(row)]
        train_rows = [row for row in rows if not is_eval(row)]
        rules = best_rules_for_training(train_rows, descriptor_fields)
        for row in eval_rows:
            selected, fired, selected_rule = apply_routing_rules(row, rules)
            selected_da = safe_float(row.get(f"{selected}_da"))
            smooth_da = safe_float(row.get("smooth_k3_da"))
            best_da = safe_float(row.get("best_restricted_da"))
            out.append(
                {
                    "row_type": "task_result",
                    "cv_type": cv_type,
                    "heldout": heldout,
                    "rule_type": "threshold_rule_stack",
                    "selected_rule": selected_rule,
                    "task": row["task"],
                    "selected_config": selected,
                    "selected_da": selected_da,
                    "smooth_da": smooth_da,
                    "delta_vs_smooth": delta(selected_da, smooth_da),
                    "best_restricted_config": row.get("best_restricted_config"),
                    "best_restricted_da": best_da,
                    "gap_to_oracle": delta(best_da, selected_da),
                    "fired_rules": fired,
                }
            )
    out.extend(cv_summary_rows(out))
    return out


def cv_summary_rows(cv_rows):
    summaries = []
    groups = defaultdict(list)
    for row in cv_rows:
        if row.get("row_type") == "task_result":
            groups[(row["cv_type"], row["rule_type"])].append(row)
    for (cv_type, rule_type), rows in groups.items():
        counts = Counter(row["selected_config"] for row in rows)
        summaries.append(
            {
                "row_type": "summary",
                "cv_type": cv_type,
                "heldout": "ALL",
                "rule_type": rule_type,
                "selected_rule": "",
                "task": "",
                "selected_config": "",
                "selected_da": "",
                "smooth_da": "",
                "delta_vs_smooth": "",
                "best_restricted_config": "",
                "best_restricted_da": "",
                "gap_to_oracle": mean([row.get("gap_to_oracle") for row in rows]),
                "fired_rules": "",
                "avg_da": mean([row.get("selected_da") for row in rows]),
                "smooth_avg_da": mean([row.get("smooth_da") for row in rows]),
                "gain_vs_smooth": mean([row.get("delta_vs_smooth") for row in rows]),
                "positive_tasks": sum(1 for row in rows if safe_float(row.get("delta_vs_smooth")) is not None and safe_float(row.get("delta_vs_smooth")) > 0),
                "negative_tasks": sum(1 for row in rows if safe_float(row.get("delta_vs_smooth")) is not None and safe_float(row.get("delta_vs_smooth")) < 0),
                "selected_config_counts": ";".join(f"{key}:{counts[key]}" for key in sorted(counts)),
            }
        )
    return summaries


def failure_type(row):
    selected = row.get("selected_config")
    selected_da = safe_float(row.get("selected_da"))
    smooth_da = safe_float(row.get("smooth_da"))
    best_da = safe_float(row.get("best_restricted_da"))
    if selected_da is None or smooth_da is None:
        return "missing_da"
    if selected_da < smooth_da:
        if selected == "elastic_r2":
            return "elastic_overuse"
        if selected == "timepoint1":
            return "rigidity_overuse"
        if selected == "raw1":
            return "too_global"
        return "hurts_vs_smooth"
    if selected == "plain" and best_da is not None and best_da - selected_da > 0.02:
        return "under_structured"
    if best_da is not None and best_da - selected_da <= 0.005:
        return "near_oracle_good"
    return "improves_but_not_best"


def descriptor_failure_cases(cv_rows, merged_rows, descriptor_fields):
    merged = {row["task"]: row for row in merged_rows}
    out = []
    for row in cv_rows:
        if row.get("row_type") != "task_result":
            continue
        ftype = failure_type(row)
        if ftype == "near_oracle_good":
            continue
        source_row = merged.get(row["task"], {})
        top_values = []
        for field in descriptor_fields[:12]:
            value = source_row.get(field)
            if safe_float(value) is not None:
                top_values.append(f"{field}={fmt(value)}")
        out.append(
            {
                "cv_type": row.get("cv_type"),
                "heldout": row.get("heldout"),
                "task": row.get("task"),
                "selected_config": row.get("selected_config"),
                "selected_da": row.get("selected_da"),
                "smooth_da": row.get("smooth_da"),
                "best_restricted_config": row.get("best_restricted_config"),
                "best_restricted_da": row.get("best_restricted_da"),
                "error_vs_smooth": delta(row.get("smooth_da"), row.get("selected_da")),
                "error_vs_oracle": row.get("gap_to_oracle"),
                "fired_rules": row.get("fired_rules"),
                "top_descriptors_values": ";".join(top_values),
                "failure_type": ftype,
            }
        )
    return out


def field_union(rows, preferred=None):
    fields = []
    for field in preferred or []:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def write_summary(path, input_coverage, availability, merged_rows, outcome_analysis, threshold_rules, cv_rows, failures):
    summary_rows = [row for row in cv_rows if row.get("row_type") == "summary"]
    failure_counts = Counter(row["failure_type"] for row in failures)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# v287 Stable Descriptor Extraction Summary\n\n")
        handle.write("## 1. Input and Coverage\n\n")
        write_md_table(handle, input_coverage, ["file", "rows", "status"])

        handle.write("## 2. Descriptor Availability\n\n")
        availability_rows = [{"descriptor_group": key, "status": value} for key, value in availability.items()]
        write_md_table(handle, availability_rows, ["descriptor_group", "status"])

        handle.write("## 3. Merged Descriptor Table\n\n")
        compact_fields = [
            "task",
            "plain_da",
            "raw1_da",
            "timepoint1_da",
            "smooth_k3_da",
            "elastic_r2_da",
            "best_restricted_config",
            "source_raw_pooled_fisher",
            "source_raw_timepoint_fisher",
            "source_raw_temporal_shape_fisher",
            "pair_raw_shift_gain_ratio",
            "pair_raw_best_shift_abs",
        ]
        write_md_table(handle, merged_rows, compact_fields)

        handle.write("## 4. Descriptor-Outcome Correlation\n\n")
        for outcome in ["elastic_r2_gain_vs_smooth", "raw_gain_vs_plain", "timepoint_gain_vs_plain", "weak_best"]:
            top = [row for row in outcome_analysis if row["outcome"] == outcome][:8]
            handle.write(f"### {outcome}\n\n")
            write_md_table(handle, top, ["descriptor", "n_valid", "pearson", "spearman", "abs_pearson", "analysis_note"])

        handle.write("## 5. Threshold Rule Search\n\n")
        for outcome in ["elastic_help", "raw_help", "timepoint_help", "weak_help"]:
            top = [row for row in threshold_rules if row["target_outcome"] == outcome][:5]
            handle.write(f"### {outcome}\n\n")
            write_md_table(handle, top, ["rule_id", "rule_text", "n_selected", "precision", "recall", "f1", "avg_delta_selected"])

        handle.write("## 6. Descriptor Routing CV\n\n")
        write_md_table(
            handle,
            summary_rows,
            ["cv_type", "rule_type", "avg_da", "smooth_avg_da", "gain_vs_smooth", "positive_tasks", "negative_tasks", "gap_to_oracle", "selected_config_counts"],
        )

        handle.write("## 7. Failure Cases\n\n")
        failure_rows = [{"failure_type": key, "n": failure_counts[key]} for key in sorted(failure_counts)]
        write_md_table(handle, failure_rows, ["failure_type", "n"])

        handle.write("## 8. Minimal Observations\n\n")
        handle.write("- All correlation rows are exploratory only because n=12.\n")
        handle.write("- Feature descriptors are NA unless canonical checkpoint inference is explicitly added later.\n")
        raw_ok = availability.get("raw_descriptor")
        if raw_ok:
            handle.write(f"- raw_descriptor: {raw_ok}\n")
        if summary_rows:
            for row in summary_rows:
                handle.write(
                    f"- {row['cv_type']} / {row['rule_type']}: avg_da={fmt_md(row.get('avg_da'))}, "
                    f"gain_vs_smooth={fmt_md(row.get('gain_vs_smooth'))}.\n"
                )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path("logs") / f"v287_stable_descriptor_extraction_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded, input_coverage = load_input_rows(args.v285_dir, args.v286_dir)
    outcomes = build_outcomes(loaded)
    metadata_rows, metadata_status = metadata_descriptor_rows(args.data_root)
    raw_rows, raw_status = build_raw_descriptor_rows(args)
    feature_rows = feature_descriptor_rows()
    merged_rows = merge_rows(outcomes, metadata_rows, raw_rows, feature_rows)
    descriptor_fields = numeric_descriptor_fields(merged_rows, allow_identity_rules=args.allow_identity_rules)
    outcome_analysis = descriptor_outcome_analysis(merged_rows, descriptor_fields)
    threshold_rules = threshold_rule_rows(merged_rows, descriptor_fields)
    cv_rows = descriptor_routing_cv(merged_rows, descriptor_fields)
    failures = descriptor_failure_cases(cv_rows, merged_rows, descriptor_fields)

    metadata_fields = field_union(metadata_rows, ["task", "source", "target"])
    raw_fields = field_union(raw_rows, ["task", "source", "target"])
    feature_fields = field_union(feature_rows, ["task", "source", "target"])
    merged_fields = field_union(merged_rows, ["task", "source", "target"])

    write_tsv(output_dir / "pair_metadata_descriptor.tsv", metadata_rows, metadata_fields)
    write_tsv(output_dir / "pair_raw_timeseries_descriptor.tsv", raw_rows, raw_fields)
    write_tsv(output_dir / "pair_feature_descriptor.tsv", feature_rows, feature_fields)
    write_tsv(output_dir / "pair_descriptor_merged.tsv", merged_rows, merged_fields)
    write_tsv(
        output_dir / "descriptor_outcome_analysis.tsv",
        outcome_analysis,
        [
            "descriptor",
            "outcome",
            "n_valid",
            "pearson",
            "spearman",
            "abs_pearson",
            "abs_spearman",
            "group_mean_true",
            "group_mean_false",
            "mean_diff",
            "analysis_note",
        ],
    )
    write_tsv(
        output_dir / "descriptor_threshold_rules.tsv",
        threshold_rules,
        [
            "rule_id",
            "target_outcome",
            "rule_text",
            "n_selected",
            "precision",
            "recall",
            "f1",
            "selected_tasks",
            "false_positive_tasks",
            "false_negative_tasks",
            "avg_delta_selected",
        ],
    )
    write_tsv(
        output_dir / "descriptor_routing_cv.tsv",
        cv_rows,
        [
            "row_type",
            "cv_type",
            "heldout",
            "rule_type",
            "selected_rule",
            "task",
            "selected_config",
            "selected_da",
            "smooth_da",
            "delta_vs_smooth",
            "best_restricted_config",
            "best_restricted_da",
            "gap_to_oracle",
            "fired_rules",
            "avg_da",
            "smooth_avg_da",
            "gain_vs_smooth",
            "positive_tasks",
            "negative_tasks",
            "selected_config_counts",
        ],
    )
    write_tsv(
        output_dir / "descriptor_failure_cases.tsv",
        failures,
        [
            "cv_type",
            "heldout",
            "task",
            "selected_config",
            "selected_da",
            "smooth_da",
            "best_restricted_config",
            "best_restricted_da",
            "error_vs_smooth",
            "error_vs_oracle",
            "fired_rules",
            "top_descriptors_values",
            "failure_type",
        ],
    )

    availability = {
        "metadata_descriptor": summarize_status(metadata_status),
        "raw_descriptor": summarize_status(raw_status),
        "feature_descriptor": "NA: feature_descriptor_disabled_no_canonical_checkpoint_inference",
        "identity_descriptor": "available",
        "threshold_rule_identity_fields": "excluded" if not args.allow_identity_rules else "included",
    }
    write_summary(
        output_dir / "v287_stable_descriptor_summary.md",
        input_coverage,
        availability,
        merged_rows,
        outcome_analysis,
        threshold_rules,
        cv_rows,
        failures,
    )

    print(f"OUTPUT_DIR={output_dir}")
    for name in [
        "pair_metadata_descriptor.tsv",
        "pair_raw_timeseries_descriptor.tsv",
        "pair_feature_descriptor.tsv",
        "pair_descriptor_merged.tsv",
        "descriptor_outcome_analysis.tsv",
        "descriptor_threshold_rules.tsv",
        "descriptor_routing_cv.tsv",
        "descriptor_failure_cases.tsv",
        "v287_stable_descriptor_summary.md",
    ]:
        print(f"OUTPUT|{name}|{output_dir / name}")


def summarize_status(status_map):
    counts = Counter(str(value).split(":")[0] for value in status_map.values())
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def timestamp():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
