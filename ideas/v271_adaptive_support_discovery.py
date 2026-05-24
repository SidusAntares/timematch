import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ideas.source_phase_compactness import _segment_masks_from_spec
from utils.train_utils import to_cuda


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


def parse_pair_key(key):
    parts = str(key).replace(",", ":").replace(";", ":").split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid class-pair key: {key}")
    a, b = int(parts[0]), int(parts[1])
    _, a, b = pair_key(a, b)
    return a, b


def candidate_shifts(target_to_source_shift, shift_jitter):
    shift = int(target_to_source_shift)
    jitter = int(shift_jitter)
    if jitter <= 0:
        return [shift]
    shifts = [shift, shift - jitter, shift + jitter]
    seen = set()
    ordered = []
    for item in shifts:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def bounded_ambiguity_weight(margin, top2_mass, max_margin, min_top2_mass, soft_evidence=True):
    margin = max(float(margin), 0.0)
    top2_mass = max(float(top2_mass), 0.0)
    max_margin = max(float(max_margin), 1e-6)
    min_top2_mass = max(float(min_top2_mass), 1e-6)
    if not bool(soft_evidence):
        if margin <= max_margin and top2_mass >= min_top2_mass:
            return 1.0
        return 0.0
    gap_weight = math.exp(-margin / max_margin)
    mass_weight = min(1.0, top2_mass / min_top2_mass)
    return max(0.0, min(1.0, gap_weight * mass_weight))


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


def _accumulate_score(
    store,
    score,
    source_separability,
    explainability,
    ambiguity,
    shift_stability=1.0,
    actual_raw_score=0.0,
    baseline_raw_score=0.0,
    baseline_separability=0.0,
    baseline_explainability=0.0,
):
    store["score_sum"] += score
    store["source_separability_sum"] += source_separability
    store["target_explainability_sum"] += explainability
    store["ambiguity_sum"] += ambiguity
    store["shift_stability_sum"] += shift_stability
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
                "shift_stability": stats.get("shift_stability_sum", 0.0) / count,
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


def _stats_to_pair_segment_rows(pair_segment_stats, phase_partition_spec, segment_count):
    intervals = phase_partition_spec.get("intervals") or [(idx + 1, idx + 1) for idx in range(segment_count)]
    rows = []
    for (pair, seg_idx), stats in pair_segment_stats.items():
        count = max(stats.get("count", 0.0), 1.0)
        start, end = intervals[seg_idx] if seg_idx < len(intervals) else (seg_idx + 1, seg_idx + 1)
        actual_mean = stats.get("actual_raw_score_sum", 0.0) / count
        baseline_mean = stats.get("baseline_raw_score_sum", 0.0) / count
        relative_mean = actual_mean - baseline_mean
        ratio_mean = actual_mean / max(baseline_mean, 1e-6)
        rows.append(
            {
                "pair": pair,
                "segment": seg_idx + 1,
                "start": int(start),
                "end": int(end),
                "score": max(relative_mean, 0.0),
                "support_count": int(stats.get("count", 0.0)),
                "source_separability": stats.get("source_separability_sum", 0.0) / count,
                "target_explainability": stats.get("target_explainability_sum", 0.0) / count,
                "ambiguity": stats.get("ambiguity_sum", 0.0) / count,
                "shift_stability": stats.get("shift_stability_sum", 0.0) / count,
                "actual_raw_score": actual_mean,
                "baseline_raw_score": baseline_mean,
                "relative_score": relative_mean,
                "ratio_score": ratio_mean,
                "baseline_separability": stats.get("baseline_separability_sum", 0.0) / count,
                "baseline_explainability": stats.get("baseline_explainability_sum", 0.0) / count,
            }
        )
    rows.sort(key=lambda row: (row["pair"], row["segment"]))
    return rows


def _stats_to_pair_rows(pair_stats):
    rows = []
    for key, stats in pair_stats.items():
        count = max(stats.get("count", 0.0), 1.0)
        rows.append(
            {
                "pair": key,
                "score": stats.get("score_sum", 0.0) / count,
                "support_count": int(stats.get("count", 0.0)),
                "source_separability": stats.get("source_separability_sum", 0.0) / count,
                "target_explainability": stats.get("target_explainability_sum", 0.0) / count,
                "ambiguity": stats.get("ambiguity_sum", 0.0) / count,
                "shift_stability": stats.get("shift_stability_sum", 0.0) / count,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


@torch.no_grad()
def discover_target_pair_segments(
    model,
    reshaper,
    loader,
    prototypes,
    variances,
    counts,
    phase_partition_spec,
    device,
    target_to_source_shift,
    max_batches,
    max_margin,
    min_top2_mass,
    prototype_temperature,
    shift_jitter,
    soft_evidence=True,
    shuffle_pair_baseline=True,
    baseline_pairs_per_sample=4,
    baseline_mode="mean",
):
    num_classes, segment_count, _ = prototypes.shape
    segment_stats = defaultdict(lambda: defaultdict(float))
    pair_stats = defaultdict(lambda: defaultdict(float))
    pair_segment_stats = defaultdict(lambda: defaultdict(float))
    baseline_segment_stats = defaultdict(lambda: defaultdict(float))
    accepted_weight = 0.0
    accepted_hard = 0
    seen = 0

    shifts = candidate_shifts(target_to_source_shift, shift_jitter)

    for batch_idx, sample in enumerate(tqdm(loader, desc="target ambiguity")):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        pixels, mask, positions, extra = to_cuda(sample, device)
        feats_for_segments = model.spatial_encoder(pixels, mask, extra)

        top2_probs = []
        top2_indices = []
        pair_counts_per_sample = [defaultdict(int) for _ in range(feats_for_segments.shape[0])]
        for shift in shifts:
            logits = model.decoder(model.temporal_encoder(feats_for_segments, positions + int(shift)))
            probs = F.softmax(logits, dim=1)
            shift_top2_prob, shift_top2_idx = torch.topk(probs, k=min(2, num_classes), dim=1)
            if shift_top2_prob.shape[1] < 2:
                continue
            top2_probs.append(shift_top2_prob)
            top2_indices.append(shift_top2_idx)
            for sample_idx in range(shift_top2_idx.shape[0]):
                key = pair_key(
                    int(shift_top2_idx[sample_idx, 0].item()),
                    int(shift_top2_idx[sample_idx, 1].item()),
                )[0]
                pair_counts_per_sample[sample_idx][key] += 1

        if not top2_probs:
            continue

        seen += int(feats_for_segments.shape[0])
        for shift_idx, shift in enumerate(shifts[: len(top2_probs)]):
            shifted_positions = positions + int(shift)
            masks = _segment_masks_from_spec(shifted_positions, phase_partition_spec)
            probs = top2_probs[shift_idx]
            indices = top2_indices[shift_idx]
            margins = probs[:, 0] - probs[:, 1]
            top2_mass = probs.sum(dim=1)
            sample_evidence = {}
            for sample_idx in range(indices.shape[0]):
                class_a = int(indices[sample_idx, 0].item())
                class_b = int(indices[sample_idx, 1].item())
                key, class_a, class_b = pair_key(class_a, class_b)
                ambiguity = bounded_ambiguity_weight(
                    margins[sample_idx].item(),
                    top2_mass[sample_idx].item(),
                    max_margin=max_margin,
                    min_top2_mass=min_top2_mass,
                    soft_evidence=soft_evidence,
                )
                if ambiguity <= 0.0:
                    continue
                shift_stability = pair_counts_per_sample[sample_idx][key] / max(len(top2_probs), 1)
                evidence_weight = ambiguity * shift_stability
                if evidence_weight <= 0.0:
                    continue
                sample_evidence[sample_idx] = {
                    "key": key,
                    "class_a": class_a,
                    "class_b": class_b,
                    "ambiguity": ambiguity,
                    "shift_stability": shift_stability,
                    "evidence_weight": evidence_weight,
                }
                accepted_hard += 1
                accepted_weight += evidence_weight
            if not sample_evidence:
                continue

            for seg_idx, seg_mask in enumerate(masks):
                valid_segment = seg_mask.sum(dim=1) > 0
                if not bool(valid_segment.any().item()):
                    continue
                weights = seg_mask.to(dtype=feats_for_segments.dtype).unsqueeze(-1)
                pooled = (feats_for_segments * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

                for sample_idx in torch.nonzero(valid_segment, as_tuple=False).flatten():
                    evidence = sample_evidence.get(int(sample_idx.item()))
                    if evidence is None:
                        continue
                    key = evidence["key"]
                    class_a = evidence["class_a"]
                    class_b = evidence["class_b"]
                    ambiguity = evidence["ambiguity"]
                    shift_stability = evidence["shift_stability"]
                    evidence_weight = evidence["evidence_weight"]
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
                    source_separability = actual_components["source_separability"]
                    explainability = actual_components["target_explainability"]
                    actual_raw_score = evidence_weight * source_separability * explainability

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
                                evidence_weight
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

                    for store in (
                        segment_stats[seg_idx],
                        pair_stats[key],
                        pair_segment_stats[(key, seg_idx)],
                    ):
                        _accumulate_score(
                            store,
                            actual_raw_score,
                            source_separability,
                            explainability,
                            ambiguity,
                            shift_stability=shift_stability,
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
                        shift_stability=shift_stability,
                        actual_raw_score=actual_raw_score,
                        baseline_raw_score=baseline_raw_score,
                        baseline_separability=baseline_sep,
                        baseline_explainability=baseline_expl,
                    )

    segment_rows = _stats_to_segment_rows(segment_stats, phase_partition_spec, segment_count)
    baseline_rows = _stats_to_segment_rows(baseline_segment_stats, phase_partition_spec, segment_count)
    pair_segment_rows = _stats_to_pair_segment_rows(pair_segment_stats, phase_partition_spec, segment_count)
    candidate_count = max(seen * max(len(shifts), 1), 1)
    return {
        "seen_target_samples": seen,
        "accepted_ambiguous_samples": accepted_hard,
        "accepted_ambiguity_events": accepted_hard,
        "accepted_fraction": accepted_hard / candidate_count,
        "accepted_evidence_weight": accepted_weight,
        "shifts": shifts,
        "segment_rows": sorted(segment_rows, key=lambda row: row["score"], reverse=True),
        "segment_rows_by_index": segment_rows,
        "baseline_segment_rows": sorted(baseline_rows, key=lambda row: row["score"], reverse=True),
        "pair_rows": _stats_to_pair_rows(pair_stats),
        "pair_segment_rows": pair_segment_rows,
    }


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
                "shift_stability_sum": row.get("shift_stability", 1.0),
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
            current["shift_stability_sum"] += row.get("shift_stability", 1.0)
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
                "shift_stability": segment["shift_stability_sum"] / n,
                "actual_raw_score": segment["actual_raw_score_sum"] / n,
                "baseline_raw_score": segment["baseline_raw_score_sum"] / n,
                "relative_score": segment["relative_score_sum"] / n,
                "ratio_score": segment["ratio_score_sum"] / n,
                "dominant_pair": best_pair,
            }
        )
    merged.sort(key=lambda row: row["score"], reverse=True)
    return threshold, merged


def _finalize_pair_support(group, pair):
    n = max(len(group["atomic_segments"]), 1)
    class_a, class_b = parse_pair_key(pair)
    return {
        "classes": [class_a, class_b],
        "class_pair": [class_a, class_b],
        "start": int(group["start"]),
        "end": int(group["end"]),
        "atomic_segments": group["atomic_segments"],
        "score": group["score_sum"] / n,
        "gate": 1.0,
        "support_count": int(group["support_count"]),
        "source_separability": group["source_separability_sum"] / n,
        "target_explainability": group["target_explainability_sum"] / n,
        "ambiguity": group["ambiguity_sum"] / n,
        "shift_stability": group["shift_stability_sum"] / n,
        "actual_raw_score": group["actual_raw_score_sum"] / n,
        "baseline_raw_score": group["baseline_raw_score_sum"] / n,
        "relative_score": group["relative_score_sum"] / n,
        "ratio_score": group["ratio_score_sum"] / n,
    }


def _start_pair_group(row):
    return {
        "start": row["start"],
        "end": row["end"],
        "last_segment": row["segment"],
        "atomic_segments": [row["segment"]],
        "score_sum": row["score"],
        "support_count": row["support_count"],
        "source_separability_sum": row["source_separability"],
        "target_explainability_sum": row["target_explainability"],
        "ambiguity_sum": row["ambiguity"],
        "shift_stability_sum": row.get("shift_stability", 1.0),
        "actual_raw_score_sum": row["actual_raw_score"],
        "baseline_raw_score_sum": row["baseline_raw_score"],
        "relative_score_sum": row["relative_score"],
        "ratio_score_sum": row["ratio_score"],
    }


def _candidate_group_after_append(group, row):
    candidate = dict(group)
    candidate["end"] = row["end"]
    candidate["last_segment"] = row["segment"]
    candidate["atomic_segments"] = list(group["atomic_segments"]) + [row["segment"]]
    candidate["score_sum"] = group["score_sum"] + row["score"]
    candidate["support_count"] = group["support_count"] + row["support_count"]
    candidate["source_separability_sum"] = group["source_separability_sum"] + row["source_separability"]
    candidate["target_explainability_sum"] = group["target_explainability_sum"] + row["target_explainability"]
    candidate["ambiguity_sum"] = group["ambiguity_sum"] + row["ambiguity"]
    candidate["shift_stability_sum"] = group["shift_stability_sum"] + row.get("shift_stability", 1.0)
    candidate["actual_raw_score_sum"] = group["actual_raw_score_sum"] + row["actual_raw_score"]
    candidate["baseline_raw_score_sum"] = group["baseline_raw_score_sum"] + row["baseline_raw_score"]
    candidate["relative_score_sum"] = group["relative_score_sum"] + row["relative_score"]
    candidate["ratio_score_sum"] = group["ratio_score_sum"] + row["ratio_score"]
    return candidate


def _support_span(start, end):
    return int(end) - int(start) + 1


def _row_passes_pair_filter(row, threshold, min_ratio, min_support_count, min_shift_stability):
    return (
        row["score"] >= threshold
        and row.get("ratio_score", 0.0) >= float(min_ratio)
        and row.get("support_count", 0) >= int(min_support_count)
        and row.get("shift_stability", 0.0) >= float(min_shift_stability)
    )


def _group_passes_shape_constraints(group, max_support_atoms, max_interval_span):
    if int(max_support_atoms) > 0 and len(group["atomic_segments"]) > int(max_support_atoms):
        return False
    if int(max_interval_span) > 0 and _support_span(group["start"], group["end"]) > int(max_interval_span):
        return False
    return True


def _support_passes_reliability(support, min_support_count, min_shift_stability, max_support_atoms, max_interval_span):
    return (
        support.get("support_count", 0) >= int(min_support_count)
        and support.get("shift_stability", 0.0) >= float(min_shift_stability)
        and _group_passes_shape_constraints(support, max_support_atoms, max_interval_span)
    )


def construct_pair_adaptive_supports(
    pair_segment_rows,
    score_quantile,
    min_score,
    min_ratio,
    top_m_per_pair=2,
    max_supports=12,
    min_support_count=1,
    min_shift_stability=0.0,
    max_support_atoms=0,
    max_interval_span=0,
    gate_score_high=0.0,
):
    positive_scores = [
        row["score"]
        for row in pair_segment_rows
        if row["score"] > 0.0
        and row.get("ratio_score", 0.0) >= float(min_ratio)
        and row.get("support_count", 0) >= int(min_support_count)
        and row.get("shift_stability", 0.0) >= float(min_shift_stability)
    ]
    if positive_scores:
        threshold = float(np.quantile(np.asarray(positive_scores), float(score_quantile)))
        threshold = max(threshold, float(min_score))
    else:
        threshold = float(min_score)

    rows_by_pair = defaultdict(list)
    for row in pair_segment_rows:
        rows_by_pair[str(row["pair"])].append(row)

    supports = []
    for pair, rows in rows_by_pair.items():
        rows = sorted(rows, key=lambda row: int(row["segment"]))
        pair_supports = []
        current = None
        for row in rows:
            keep = _row_passes_pair_filter(
                row,
                threshold,
                min_ratio,
                min_support_count,
                min_shift_stability,
            )
            if not keep:
                if current is not None:
                    pair_supports.append(_finalize_pair_support(current, pair))
                    current = None
                continue
            if current is None:
                current = _start_pair_group(row)
                continue
            if int(row["segment"]) == int(current["last_segment"]) + 1:
                candidate = _candidate_group_after_append(current, row)
                if _group_passes_shape_constraints(candidate, max_support_atoms, max_interval_span):
                    current = candidate
                else:
                    pair_supports.append(_finalize_pair_support(current, pair))
                    current = _start_pair_group(row)
            else:
                pair_supports.append(_finalize_pair_support(current, pair))
                current = _start_pair_group(row)
        if current is not None:
            pair_supports.append(_finalize_pair_support(current, pair))

        pair_supports = [
            support
            for support in pair_supports
            if _support_passes_reliability(
                support,
                min_support_count,
                min_shift_stability,
                max_support_atoms,
                max_interval_span,
            )
        ]
        pair_supports.sort(key=lambda item: item["score"], reverse=True)
        supports.extend(pair_supports[: max(1, int(top_m_per_pair))])

    supports.sort(key=lambda item: item["score"], reverse=True)
    supports = supports[: max(0, int(max_supports))]
    max_score = max([support["score"] for support in supports], default=0.0)
    for support in supports:
        if float(gate_score_high) > 0.0:
            support["gate"] = max(0.0, min(1.0, support["score"] / float(gate_score_high)))
        else:
            support["gate"] = 0.0 if max_score <= 0.0 else max(0.0, min(1.0, support["score"] / max_score))
    return threshold, supports
