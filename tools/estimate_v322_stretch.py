"""Offline domain-level stretch estimation for v3.2.2."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from distutils.util import strtobool
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import PixelSetData, count_pixelset_samples  # noqa: E402
from methods.temporal_alignment.affine_transform import transform_positions  # noqa: E402
from methods.temporal_alignment.range_gate import compute_range_statistics  # noqa: E402
from methods.temporal_alignment.stretch_estimator import (  # noqa: E402
    STRETCH_ESTIMATE_SCHEMA,
    build_formal_identity_audit,
    build_identity_debug_report,
    build_candidate_result,
    build_estimate_result,
    candidate_json_records,
    candidate_tsv_records,
    create_target_train_indices,
    file_sha256,
    finalize_identity_debug_report,
    fixed_stretch_candidates,
    require_identity_audit_passed,
    repeated_score_diagnostics,
    tensor_sha256,
    validate_checkpoint_sha256,
    validate_temporal_audit,
)
from methods.timematch_base.train_loop import score_shift_softmaxes  # noqa: E402
from models.stclassifier import PseLTae  # noqa: E402
from transforms import Normalize, RandomSamplePixels, ToTensor  # noqa: E402
from utils import label_utils  # noqa: E402


TSV_FIELDS = [
    "task",
    "global_shift",
    "anchor",
    "stretch",
    "score",
    "valid",
    "embedding_range_valid",
    "transformed_min",
    "transformed_max",
    "source_overlap_span",
    "source_overlap_ratio",
    "in_source_point_ratio",
    "duplicate_count",
    "duplicate_ratio",
    "rank",
    "selected",
    "at_boundary",
]


def bool_flag(value: Any) -> bool:
    return bool(strtobool(str(value)))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _move_sample(sample: Mapping[str, torch.Tensor], device: torch.device):
    """Move only model inputs; deliberately never inspect target labels."""

    extra = sample.get("extra")
    return (
        sample["pixels"].to(device=device, non_blocking=True),
        sample["valid_pixels"].to(device=device, non_blocking=True),
        sample["positions"].to(device=device, non_blocking=True),
        None if extra is None else extra.to(device=device, non_blocking=True),
    )


def _repository_identity(branch: str, commit: str) -> Dict[str, str]:
    branch = str(branch).strip()
    commit = str(commit).strip()
    if not branch:
        raise ValueError("repository branch must be nonempty")
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit) is None:
        raise ValueError("repository commit must be a 7 to 40 hexadecimal Git hash")
    return {"branch": branch, "commit": commit}


def _infer_source_classes(args: argparse.Namespace) -> Tuple[List[str], int]:
    classes = label_utils.get_classes(
        args.source.split("/")[0],
        combine_spring_and_winter=args.combine_spring_and_winter,
    )
    if args.closed_set:
        classes = [name for name in classes if name != "unknown"]
    source_data = PixelSetData(
        args.data_root,
        args.source,
        classes,
        closed_set=args.closed_set,
    )
    source_labels = source_data.get_labels()
    labels, counts = np.unique(source_labels, return_counts=True)
    kept_classes = [classes[index] for index in labels[counts >= args.source_min_count]]
    return kept_classes, len(source_data)


def _build_target_loader(
    args: argparse.Namespace,
    classes: Sequence[str],
    source_index_count: int,
) -> data.DataLoader:
    target_count = count_pixelset_samples(
        args.data_root, args.target, classes, closed_set=args.closed_set
    )
    target_train_indices = create_target_train_indices(
        source_index_count,
        target_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    weak_transform = transforms.Compose(
        [RandomSamplePixels(args.num_pixels), Normalize(), ToTensor()]
    )
    target_dataset = PixelSetData(
        args.data_root,
        args.target,
        classes,
        transform=weak_transform,
        indices=target_train_indices,
        with_extra=args.with_extra,
        closed_set=args.closed_set,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return data.DataLoader(
        target_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _load_model(args: argparse.Namespace, num_classes: int, device: torch.device) -> PseLTae:
    model = PseLTae(
        input_dim=args.input_dim,
        num_classes=num_classes,
        with_extra=args.with_extra,
        max_temporal_shift=args.max_temporal_shift,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _score_unlabeled_softmaxes(
    candidate_values: Sequence[float],
    softmaxes: np.ndarray,
    *,
    num_classes: int,
    epsilon: float,
) -> Mapping[str, Any]:
    """Reuse TimeMatch scoring while ensuring no target labels are consumed.

    Only ``is_scores`` are used. Dummy labels and a uniform class distribution
    satisfy legacy diagnostic fields without changing the Inception Score.
    """

    dummy_labels = np.zeros(softmaxes.shape[0], dtype=np.int64)
    uniform_distribution = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    return score_shift_softmaxes(
        list(candidate_values),
        softmaxes,
        dummy_labels,
        num_classes,
        class_distribution=uniform_distribution,
        shift_score_epsilon=epsilon,
    )


def _single_identity_score(
    softmaxes: torch.Tensor,
    *,
    num_classes: int,
    epsilon: float,
) -> float:
    scores = _score_unlabeled_softmaxes(
        [1.0],
        softmaxes.detach().cpu().numpy()[:, None, :],
        num_classes=num_classes,
        epsilon=epsilon,
    )
    return float(scores["is_scores"][0])


def _output_comparison(
    logits_left: torch.Tensor,
    logits_right: torch.Tensor,
    probabilities_left: torch.Tensor,
    probabilities_right: torch.Tensor,
    score_left: float,
    score_right: float,
    *,
    positions_equal: Optional[bool] = None,
) -> Dict[str, Any]:
    comparison = {
        "logits_max_abs_diff": float(
            (logits_left - logits_right).abs().max().item()
        ),
        "softmax_max_abs_diff": float(
            (probabilities_left - probabilities_right).abs().max().item()
        ),
        "argmax_equal": bool(
            torch.equal(logits_left.argmax(dim=1), logits_right.argmax(dim=1))
        ),
        "score_abs_diff": abs(float(score_left) - float(score_right)),
    }
    if positions_equal is not None:
        comparison["positions_equal"] = bool(positions_equal)
    return comparison


@torch.no_grad()
def _run_identity_debug(
    model: PseLTae,
    loader: data.DataLoader,
    device: torch.device,
    *,
    global_shift: int,
    global_shift_score: float,
    anchor: float,
    max_batches: int,
    num_classes: int,
    epsilon: float,
    identity_atol: float,
) -> Dict[str, Any]:
    """Run four independent, shape-equivalent a=1 forwards for diagnosis."""

    if max_batches not in (1, 2):
        raise ValueError("identity debug batches must be 1 or 2")
    model.eval()
    logits_by_run = {
        "baseline1": [],
        "baseline2": [],
        "affine1": [],
        "affine2": [],
    }
    probabilities_by_run = {name: [] for name in logits_by_run}
    baseline_positions = []
    affine_positions = []
    spatial_features_all = []
    batch_shapes = []
    per_batch_score_differences = []
    for batch_index, sample in enumerate(loader):
        if batch_index >= max_batches:
            break
        pixels, valid_pixels, positions, extra = _move_sample(sample, device)
        spatial_features = model.spatial_encoder(pixels, valid_pixels, extra)
        baseline_position = positions + int(global_shift)
        affine_position = transform_positions(
            positions,
            shift=global_shift,
            stretch=1.0,
            anchor=anchor,
        ).discrete_positions
        run_logits = {
            "baseline1": model.forward_from_temporal_features(
                spatial_features, baseline_position
            ),
            "baseline2": model.forward_from_temporal_features(
                spatial_features, baseline_position
            ),
            "affine1": model.forward_from_temporal_features(
                spatial_features, affine_position
            ),
            "affine2": model.forward_from_temporal_features(
                spatial_features, affine_position
            ),
        }
        run_probabilities = {
            name: F.softmax(logits, dim=1) for name, logits in run_logits.items()
        }
        baseline_batch_score = _single_identity_score(
            run_probabilities["baseline1"],
            num_classes=num_classes,
            epsilon=epsilon,
        )
        affine_batch_score = _single_identity_score(
            run_probabilities["affine1"],
            num_classes=num_classes,
            epsilon=epsilon,
        )
        per_batch_score_differences.append(
            abs(baseline_batch_score - affine_batch_score)
        )
        for name in logits_by_run:
            logits_by_run[name].append(run_logits[name].cpu())
            probabilities_by_run[name].append(run_probabilities[name].cpu())
        baseline_positions.append(baseline_position.cpu())
        affine_positions.append(affine_position.cpu())
        spatial_features_all.append(spatial_features.cpu())
        batch_shapes.append(list(run_logits["baseline1"].shape))
    if not baseline_positions:
        raise ValueError("target loader produced no batches for identity debug")

    logits = {
        name: torch.cat(values, dim=0) for name, values in logits_by_run.items()
    }
    probabilities = {
        name: torch.cat(values, dim=0)
        for name, values in probabilities_by_run.items()
    }
    shifted_positions = torch.cat(baseline_positions, dim=0)
    transformed_positions = torch.cat(affine_positions, dim=0)
    spatial_features = torch.cat(spatial_features_all, dim=0)
    scores = {
        name: _single_identity_score(
            value, num_classes=num_classes, epsilon=epsilon
        )
        for name, value in probabilities.items()
    }
    repeated = repeated_score_diagnostics(
        probabilities["baseline1"],
        lambda value: _single_identity_score(
            value, num_classes=num_classes, epsilon=epsilon
        ),
    )
    positions_equal = torch.equal(shifted_positions, transformed_positions)
    report = build_identity_debug_report(
        shift={
            "value": int(global_shift),
            "dtype": type(global_shift).__name__,
            "shape": [],
        },
        anchor=anchor,
        positions={
            "dtype": str(shifted_positions.dtype),
            "device": str(device),
            "shape": list(shifted_positions.shape),
            "baseline_affine_equal": bool(positions_equal),
            "max_abs_diff": float(
                (shifted_positions.to(torch.float64) - transformed_positions.to(torch.float64))
                .abs()
                .max()
                .item()
            ),
            "baseline_hash": tensor_sha256(shifted_positions),
            "affine_hash": tensor_sha256(transformed_positions),
        },
        execution={
            "model_training": bool(model.training),
            "grad_enabled": bool(torch.is_grad_enabled()),
            "spatial_features_reused": True,
            "spatial_feature_hash": tensor_sha256(spatial_features),
            "baseline_batch_shape": batch_shapes[0],
            "affine_batch_shape": batch_shapes[0],
            "batch_shapes": batch_shapes,
            "batch_count": len(batch_shapes),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "a1_batched_with_other_candidates": False,
            "temporal_forward_entry": "model.forward_from_temporal_features",
            "classifier_entry": "model.forward_from_temporal_features",
            "temporal_forward_count": 4 * len(batch_shapes),
            "other_stretch_candidate_count": 0,
        },
        comparisons={
            "baseline1_vs_baseline2": _output_comparison(
                logits["baseline1"],
                logits["baseline2"],
                probabilities["baseline1"],
                probabilities["baseline2"],
                scores["baseline1"],
                scores["baseline2"],
            ),
            "affine1_vs_affine2": _output_comparison(
                logits["affine1"],
                logits["affine2"],
                probabilities["affine1"],
                probabilities["affine2"],
                scores["affine1"],
                scores["affine2"],
            ),
            "baseline1_vs_affine1": _output_comparison(
                logits["baseline1"],
                logits["affine1"],
                probabilities["baseline1"],
                probabilities["affine1"],
                scores["baseline1"],
                scores["affine1"],
                positions_equal=positions_equal,
            ),
        },
        same_softmax_score_repeat_abs_diff=repeated[
            "same_softmax_score_repeat_abs_diff"
        ],
        baseline_affine_per_batch_score_abs_diff=max(
            per_batch_score_differences
        ),
        global_selected_score_abs_diff=abs(
            float(global_shift_score) - float(scores["baseline1"])
        ),
        identity_atol=identity_atol,
    )
    report["scores"] = scores
    return report


@torch.no_grad()
def _collect_global_shift_softmaxes(
    model: PseLTae,
    loader: data.DataLoader,
    device: torch.device,
    shifts: Sequence[int],
    max_batches: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    batches = []
    sample_count = 0
    spatial_forward_count = 0
    temporal_forward_count = 0
    for batch_index, sample in enumerate(loader):
        if batch_index >= max_batches:
            break
        pixels, valid_pixels, positions, extra = _move_sample(sample, device)
        spatial_features = model.spatial_encoder(pixels, valid_pixels, extra)
        spatial_forward_count += 1
        logits = []
        for shift in shifts:
            logits.append(
                model.forward_from_temporal_features(spatial_features, positions + int(shift))
            )
            temporal_forward_count += 1
        batches.append(F.softmax(torch.stack(logits, dim=1), dim=2).cpu())
        sample_count += int(positions.shape[0])
    if not batches:
        raise ValueError("target loader produced no batches for global shift estimation")
    return torch.cat(batches, dim=0).numpy(), {
        "target_sample_count": sample_count,
        "batch_count": len(batches),
        "spatial_forward_count": spatial_forward_count,
        "temporal_forward_count": temporal_forward_count,
    }


@torch.no_grad()
def _collect_stretch_softmaxes(
    model: PseLTae,
    loader: data.DataLoader,
    device: torch.device,
    *,
    stretches: Sequence[float],
    global_shift: int,
    anchor: float,
    source_range: Tuple[float, float],
    embedding_range: Tuple[int, int],
    max_batches: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[float],
    List[Any],
    Dict[str, Any],
    Dict[str, int],
]:
    candidate_batches = []
    baseline_batches = []
    independent_a1_batches = []
    candidate_diagnostics = None
    valid_stretches = None
    logits_max_abs_diff = 0.0
    softmax_max_abs_diff = 0.0
    positions_equal = True
    argmax_equal = True
    sample_count = 0
    spatial_forward_count = 0
    temporal_forward_count = 0
    for batch_index, sample in enumerate(loader):
        if batch_index >= max_batches:
            break
        pixels, valid_pixels, positions, extra = _move_sample(sample, device)
        spatial_features = model.spatial_encoder(pixels, valid_pixels, extra)
        spatial_forward_count += 1
        baseline_logits = model.forward_from_temporal_features(
            spatial_features, positions + int(global_shift)
        )
        temporal_forward_count += 1
        baseline_probs = F.softmax(baseline_logits, dim=1)
        stretch_logits = []
        per_batch_diagnostics = []
        for stretch in stretches:
            transformed = transform_positions(
                positions,
                shift=global_shift,
                stretch=stretch,
                anchor=anchor,
            )
            first_positions = positions[:1]
            first_transformed = transform_positions(
                first_positions,
                shift=global_shift,
                stretch=stretch,
                anchor=anchor,
            )
            stats = compute_range_statistics(
                first_positions,
                first_transformed.discrete_positions,
                source_min=source_range[0],
                source_max=source_range[1],
                embedding_index_min=embedding_range[0],
                embedding_index_max=embedding_range[1],
            )
            per_batch_diagnostics.append((first_transformed, stats))
            embedding_valid = bool(stats.embedding_range_valid.all().item())
            if not embedding_valid:
                continue
            logits = model.forward_from_temporal_features(
                spatial_features, transformed.discrete_positions
            )
            temporal_forward_count += 1
            stretch_logits.append((stretch, logits))
            if stretch == 1.0:
                positions_equal = positions_equal and torch.equal(
                    positions + int(global_shift), transformed.discrete_positions
                )
                logits_max_abs_diff = max(
                    logits_max_abs_diff,
                    float((baseline_logits - logits).abs().max().item()),
                )
                probs = F.softmax(logits, dim=1)
                independent_a1_batches.append(probs.cpu())
                softmax_max_abs_diff = max(
                    softmax_max_abs_diff,
                    float((baseline_probs - probs).abs().max().item()),
                )
                argmax_equal = argmax_equal and torch.equal(
                    baseline_logits.argmax(dim=1), logits.argmax(dim=1)
                )
        batch_valid_stretches = [item[0] for item in stretch_logits]
        if valid_stretches is None:
            valid_stretches = batch_valid_stretches
        elif batch_valid_stretches != valid_stretches:
            raise ValueError("candidate embedding validity changed across target batches")
        candidate_diagnostics = candidate_diagnostics or per_batch_diagnostics
        if not stretch_logits:
            raise ValueError("all stretch candidates are outside the embedding range")
        stacked_logits = torch.stack([item[1] for item in stretch_logits], dim=1)
        candidate_batches.append(F.softmax(stacked_logits, dim=2).cpu())
        baseline_batches.append(baseline_probs.cpu())
        sample_count += int(positions.shape[0])
    if not candidate_batches or candidate_diagnostics is None or valid_stretches is None:
        raise ValueError("target loader produced no batches for stretch estimation")
    if len(independent_a1_batches) != len(candidate_batches):
        raise ValueError("stretch=1 is outside the embedding range; identity audit cannot run")
    return (
        torch.cat(candidate_batches, dim=0).numpy(),
        torch.cat(baseline_batches, dim=0).numpy(),
        torch.cat(independent_a1_batches, dim=0).numpy(),
        valid_stretches,
        candidate_diagnostics,
        {
            "positions_equal": positions_equal,
            "logits_max_abs_diff": logits_max_abs_diff,
            "softmax_max_abs_diff": softmax_max_abs_diff,
            "argmax_equal": argmax_equal,
        },
        {
            "target_sample_count": sample_count,
            "batch_count": len(candidate_batches),
            "spatial_forward_count": spatial_forward_count,
            "temporal_forward_count": temporal_forward_count,
        },
    )


def _write_outputs(document: Mapping[str, Any], tsv_rows: Iterable[Mapping[str, Any]], args):
    output_json = Path(args.output_json)
    output_tsv = Path(args.output_tsv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in tsv_rows:
            writer.writerow({field: row.get(field) for field in TSV_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-audit-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    parser.add_argument("--repository-branch", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-pixels", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--min-shift", type=int, default=-60)
    parser.add_argument("--max-shift", type=int, default=60)
    parser.add_argument("--shift-score-epsilon", type=float, default=1e-5)
    parser.add_argument("--closed-set", type=bool_flag, default=True)
    parser.add_argument("--combine-spring-and-winter", type=bool_flag, default=False)
    parser.add_argument("--source-min-count", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--with-extra", type=bool_flag, default=False)
    parser.add_argument("--max-temporal-shift", type=int, default=100)
    parser.add_argument("--identity-atol", type=float, default=1e-7)
    parser.add_argument("--identity-debug-only", action="store_true")
    parser.add_argument("--identity-debug-batches", type=int, default=1)
    parser.add_argument("--identity-debug-output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.identity_atol != 1e-7:
        raise ValueError("v3.2.2 identity_atol is fixed at 1e-7")
    if args.identity_debug_only:
        if args.identity_debug_batches not in (1, 2):
            raise ValueError("identity debug batches must be 1 or 2")
        if not args.identity_debug_output:
            raise ValueError("--identity-debug-output is required in debug-only mode")
    elif not args.output_json or not args.output_tsv:
        raise ValueError("--output-json and --output-tsv are required in formal mode")
    total_start = time.perf_counter()
    _seed_all(args.seed)
    repository = _repository_identity(args.repository_branch, args.repository_commit)
    audit_path = Path(args.temporal_audit_json).expanduser().resolve()
    audit_document = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_inputs = validate_temporal_audit(
        audit_document, source=args.source, target=args.target
    )
    checkpoint_sha256 = validate_checkpoint_sha256(
        Path(args.checkpoint), audit_inputs["checkpoint_sha256"]
    )
    audit_sha256 = file_sha256(audit_path)

    classes, source_index_count = _infer_source_classes(args)
    shift_loader = _build_target_loader(args, classes, source_index_count)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model(args, len(classes), device)

    shifts = list(range(args.min_shift, args.max_shift + 1))
    shift_start = time.perf_counter()
    shift_batch_limit = (
        args.identity_debug_batches if args.identity_debug_only else args.sample_size
    )
    shift_softmaxes, shift_counts = _collect_global_shift_softmaxes(
        model, shift_loader, device, shifts, shift_batch_limit
    )
    shift_scores = _score_unlabeled_softmaxes(
        shifts,
        shift_softmaxes,
        num_classes=len(classes),
        epsilon=args.shift_score_epsilon,
    )
    global_shift = int(shift_scores["best_is_shift"])
    global_shift_index = shifts.index(global_shift)
    global_shift_score = float(shift_scores["is_scores"][global_shift_index])
    shift_runtime = time.perf_counter() - shift_start

    if args.identity_debug_only:
        _seed_all(args.seed)
        identity_loader = _build_target_loader(args, classes, source_index_count)
        identity_report = _run_identity_debug(
            model,
            identity_loader,
            device,
            global_shift=global_shift,
            global_shift_score=global_shift_score,
            anchor=audit_inputs["anchor"],
            max_batches=args.identity_debug_batches,
            num_classes=len(classes),
            epsilon=args.shift_score_epsilon,
            identity_atol=args.identity_atol,
        )
        identity_report.update(
            {
                "schema_version": "v322-identity-debug-v1",
                "repository": repository,
                "task": {
                    "name": args.task,
                    "source": args.source,
                    "target": args.target,
                    "seed": args.seed,
                },
                "checkpoint": {
                    "path": str(Path(args.checkpoint).expanduser().resolve()),
                    "sha256": checkpoint_sha256,
                },
                "temporal_audit": {
                    "path": str(audit_path),
                    "sha256": audit_sha256,
                },
                "global_shift_estimation": {
                    "batch_count": shift_counts["batch_count"],
                    "sample_count": shift_counts["target_sample_count"],
                    "runtime_s": shift_runtime,
                },
                "formal_stretch_output_generated": False,
            }
        )
        print(
            "V322_IDENTITY_DEBUG|"
            f"task={args.task}|shift={global_shift}|"
            f"passed={identity_report['passed']}|"
            f"failed_checks={','.join(identity_report['failed_checks']) or 'none'}|"
            f"json={Path(args.identity_debug_output).expanduser().resolve()}",
            flush=True,
        )
        finalize_identity_debug_report(
            identity_report, Path(args.identity_debug_output)
        )
        return 0

    stretches = fixed_stretch_candidates()
    _seed_all(args.seed)
    stretch_loader = _build_target_loader(args, classes, source_index_count)
    stretch_start = time.perf_counter()
    (
        stretch_softmaxes,
        baseline_softmaxes,
        independent_a1_softmaxes,
        valid_stretches,
        diagnostic_pairs,
        identity_partial,
        stretch_counts,
    ) = _collect_stretch_softmaxes(
        model,
        stretch_loader,
        device,
        stretches=stretches,
        global_shift=global_shift,
        anchor=audit_inputs["anchor"],
        source_range=audit_inputs["source_range"],
        embedding_range=audit_inputs["embedding_legal_range"],
        max_batches=args.sample_size,
    )
    stretch_scores = _score_unlabeled_softmaxes(
        valid_stretches,
        stretch_softmaxes,
        num_classes=len(classes),
        epsilon=args.shift_score_epsilon,
    )
    baseline_scores = _score_unlabeled_softmaxes(
        [1.0],
        baseline_softmaxes[:, None, :],
        num_classes=len(classes),
        epsilon=args.shift_score_epsilon,
    )
    independent_a1_scores = _score_unlabeled_softmaxes(
        [1.0],
        independent_a1_softmaxes[:, None, :],
        num_classes=len(classes),
        epsilon=args.shift_score_epsilon,
    )
    if 1.0 not in valid_stretches:
        raise ValueError("stretch=1 is outside the embedding range; identity audit cannot run")
    one_index = valid_stretches.index(1.0)
    score_abs_diff = abs(
        float(baseline_scores["is_scores"][0])
        - float(independent_a1_scores["is_scores"][0])
    )
    packed_a1_softmax_max_abs_diff = float(
        np.max(
            np.abs(
                independent_a1_softmaxes
                - stretch_softmaxes[:, one_index, :]
            )
        )
    )
    packed_a1_score_abs_diff = abs(
        float(independent_a1_scores["is_scores"][0])
        - float(stretch_scores["is_scores"][one_index])
    )
    global_score_abs_diff = abs(
        global_shift_score - float(baseline_scores["is_scores"][0])
    )
    identity_audit = build_formal_identity_audit(
        **identity_partial,
        score_abs_diff=score_abs_diff,
        packed_a1_softmax_max_abs_diff=packed_a1_softmax_max_abs_diff,
        packed_a1_score_abs_diff=packed_a1_score_abs_diff,
        global_selected_score_abs_diff=global_score_abs_diff,
        identity_atol=args.identity_atol,
    )
    require_identity_audit_passed(identity_audit)

    candidates = []
    score_by_stretch = {
        stretch: float(stretch_scores["is_scores"][index])
        for index, stretch in enumerate(valid_stretches)
    }
    for index, stretch in enumerate(stretches):
        transformed, statistics = diagnostic_pairs[index]
        candidates.append(
            build_candidate_result(
                stretch,
                score_by_stretch.get(stretch),
                transformed,
                statistics,
            )
        )
    estimate = build_estimate_result(
        candidates, global_shift=global_shift, anchor=audit_inputs["anchor"]
    )
    stretch_runtime = time.perf_counter() - stretch_start
    total_runtime = time.perf_counter() - total_start

    document = {
        "schema_version": STRETCH_ESTIMATE_SCHEMA,
        "repository": repository,
        "temporal_audit": {
            "path": str(audit_path),
            "sha256": audit_sha256,
            "schema_version": audit_inputs["audit_schema_version"],
            "repository_branch": audit_inputs["audit_repository_branch"],
            "repository_commit": audit_inputs["audit_repository_commit"],
        },
        "task": {
            "name": args.task,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).expanduser().resolve()),
            "sha256": checkpoint_sha256,
        },
        "position": {
            "source_range": list(audit_inputs["source_range"]),
            "target_range": list(audit_inputs["target_range"]),
            "embedding_legal_range": list(audit_inputs["embedding_legal_range"]),
            "anchor": audit_inputs["anchor"],
        },
        "global_shift": {
            "selected": global_shift,
            "epsilon": args.shift_score_epsilon,
            "candidate_scores": [
                {"shift": shift, "score": float(score)}
                for shift, score in zip(shifts, shift_scores["is_scores"])
            ],
        },
        "identity_audit": identity_audit,
        "stretch": {
            "candidates": candidate_json_records(estimate),
            "selected": estimate.selected_stretch,
            "selected_score": estimate.selected_score,
            "second_best": estimate.second_best_stretch,
            "second_best_score": estimate.second_best_score,
            "margin": estimate.top1_top2_margin,
            "at_boundary": estimate.selected_at_boundary,
        },
        "execution": {
            "target_sample_count": stretch_counts["target_sample_count"],
            "batch_count": stretch_counts["batch_count"],
            "spatial_forward_count": (
                shift_counts["spatial_forward_count"]
                + stretch_counts["spatial_forward_count"]
            ),
            "temporal_forward_count": (
                shift_counts["temporal_forward_count"]
                + stretch_counts["temporal_forward_count"]
            ),
        },
        "runtime": {
            "global_shift_estimation_s": shift_runtime,
            "stretch_evaluation_s": stretch_runtime,
            "total_s": total_runtime,
        },
    }
    _write_outputs(
        document, candidate_tsv_records(estimate, task=args.task), args
    )
    print(
        "V322_STRETCH_ESTIMATE|"
        f"task={args.task}|global_shift={global_shift}|"
        f"stretch={estimate.selected_stretch:.2f}|score={estimate.selected_score:.8f}|"
        f"identity_passed={identity_audit['passed']}|runtime_s={total_runtime:.3f}|"
        f"json={Path(args.output_json).resolve()}|tsv={Path(args.output_tsv).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
