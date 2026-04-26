from torch.utils.data.sampler import WeightedRandomSampler
import sklearn.metrics
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    RandomTemporalShift,
    Identity,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, to_cuda, cycle, should_disable_tqdm


def compute_class_prototypes(features, labels, ignore_class_id=None, min_samples=1):
    prototypes = {}
    counts = {}
    for class_id in torch.unique(labels):
        if ignore_class_id is not None and int(class_id.item()) == int(ignore_class_id):
            continue
        class_mask = labels == class_id
        class_count = int(torch.count_nonzero(class_mask).item())
        if class_count < min_samples:
            continue
        class_id_int = int(class_id.item())
        prototypes[class_id_int] = features[class_mask].mean(dim=0)
        counts[class_id_int] = class_count
    return prototypes, counts


def compute_temporal_instability(sequence_feats):
    if sequence_feats.dim() != 3:
        raise ValueError("sequence_feats must have shape (B, T, C)")

    if sequence_feats.shape[1] <= 1:
        return sequence_feats.new_zeros(sequence_feats.shape[:2])

    deltas = torch.norm(sequence_feats[:, 1:, :] - sequence_feats[:, :-1, :], dim=2)
    first_delta = deltas[:, :1]
    return torch.cat([first_delta, deltas], dim=1)


def compute_stable_temporal_mask(sequence_feats, keep_ratio):
    keep_ratio = float(min(max(keep_ratio, 0.0), 1.0))
    instability = compute_temporal_instability(sequence_feats)
    batch_size, seq_len = instability.shape

    if keep_ratio >= 1.0 or seq_len <= 1:
        mask = torch.ones_like(instability)
        return mask, instability

    keep_steps = max(1, int(round(seq_len * keep_ratio)))
    keep_steps = min(keep_steps, seq_len)

    sorted_idx = torch.argsort(instability, dim=1, descending=False)
    selected_idx = sorted_idx[:, :keep_steps]
    mask = torch.zeros_like(instability)
    mask.scatter_(1, selected_idx, 1.0)
    return mask, instability


def pool_stable_temporal_features(sequence_feats, keep_ratio):
    stable_mask, instability = compute_stable_temporal_mask(sequence_feats, keep_ratio)
    weights = stable_mask.unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    pooled = (sequence_feats * weights).sum(dim=1) / denom
    diagnostics = {
        "avg_instability": float(instability.mean().item()),
        "avg_stable_ratio": float(stable_mask.mean().item()),
    }
    return pooled, diagnostics


def compute_relation_matrix(prototypes, class_ids):
    proto_stack = torch.stack([prototypes[class_id] for class_id in class_ids], dim=0)
    proto_stack = F.normalize(proto_stack, dim=1)
    relation_matrix = proto_stack @ proto_stack.t()
    relation_matrix.fill_diagonal_(0.0)

    norm = relation_matrix.norm(p="fro")
    if norm > 0:
        relation_matrix = relation_matrix / norm

    return relation_matrix


def normalize_prototype_geometry(prototypes, class_ids, eps=1e-6):
    proto_stack = torch.stack([prototypes[class_id] for class_id in class_ids], dim=0)
    centered = proto_stack - proto_stack.mean(dim=0, keepdim=True)

    if centered.shape[0] >= 2:
        pairwise_dist = torch.pdist(centered, p=2)
        scale = pairwise_dist.mean()
    else:
        scale = centered.norm(dim=1).mean()

    if torch.isnan(scale) or torch.isinf(scale) or scale <= eps:
        scale = centered.new_tensor(1.0)

    normalized = centered / scale
    return {class_id: normalized[idx] for idx, class_id in enumerate(class_ids)}


def init_prototype_bank(num_classes, feat_dim, device):
    bank = torch.zeros(num_classes, feat_dim, device=device)
    valid_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    return bank, valid_mask


@torch.no_grad()
def update_prototype_bank(bank, valid_mask, prototypes, momentum):
    updated_classes = []
    for class_id, prototype in prototypes.items():
        if valid_mask[class_id]:
            bank[class_id] = momentum * bank[class_id] + (1.0 - momentum) * prototype.detach()
        else:
            bank[class_id] = prototype.detach()
            valid_mask[class_id] = True
        updated_classes.append(class_id)
    return updated_classes


def merge_prototypes_with_bank(current_prototypes, bank, valid_mask, ignore_class_id=None):
    merged = {}
    for class_id in range(valid_mask.shape[0]):
        if ignore_class_id is not None and class_id == int(ignore_class_id):
            continue
        if class_id in current_prototypes:
            merged[class_id] = current_prototypes[class_id]
        elif valid_mask[class_id]:
            merged[class_id] = bank[class_id]
    return merged


def prototype_relation_alignment_loss(source_prototypes, target_prototypes, normalize_geometry=False):
    shared_classes = sorted(set(source_prototypes.keys()) & set(target_prototypes.keys()))
    diagnostics = {
        'shared_classes': len(shared_classes),
        'active_pairs': 0,
        'enabled': int(len(shared_classes) >= 2),
    }
    if len(shared_classes) < 2:
        reference = next(iter(source_prototypes.values()), None)
        if reference is None:
            reference = next(iter(target_prototypes.values()))
        return reference.new_tensor(0.0), diagnostics

    if normalize_geometry:
        source_prototypes = normalize_prototype_geometry(source_prototypes, shared_classes)
        target_prototypes = normalize_prototype_geometry(target_prototypes, shared_classes)
    source_rel = compute_relation_matrix(source_prototypes, shared_classes)
    target_rel = compute_relation_matrix(target_prototypes, shared_classes)
    diagnostics['active_pairs'] = int(len(shared_classes) * (len(shared_classes) - 1) / 2)
    return F.mse_loss(source_rel, target_rel), diagnostics


def prototype_point_alignment_loss(source_prototypes, target_prototypes, normalize_geometry=False):
    shared_classes = sorted(set(source_prototypes.keys()) & set(target_prototypes.keys()))
    diagnostics = {
        'shared_classes': len(shared_classes),
        'active_points': len(shared_classes),
        'enabled': int(len(shared_classes) >= 1),
    }
    if len(shared_classes) == 0:
        reference = next(iter(source_prototypes.values()), None)
        if reference is None:
            reference = next(iter(target_prototypes.values()))
        return reference.new_tensor(0.0), diagnostics

    if normalize_geometry:
        source_prototypes = normalize_prototype_geometry(source_prototypes, shared_classes)
        target_prototypes = normalize_prototype_geometry(target_prototypes, shared_classes)
    source_stack = torch.stack([source_prototypes[class_id] for class_id in shared_classes], dim=0)
    target_stack = torch.stack([target_prototypes[class_id] for class_id in shared_classes], dim=0)
    source_stack = F.normalize(source_stack, dim=1)
    target_stack = F.normalize(target_stack, dim=1)
    point_loss = 1.0 - F.cosine_similarity(source_stack, target_stack, dim=1).mean()
    return point_loss, diagnostics


def extract_refined_prototype_features(base_feats, sequence_feats, config):
    if getattr(config, "pra_use_refined_prototypes", False) and sequence_feats is not None:
        refined_feats, refine_diag = pool_stable_temporal_features(
            sequence_feats, getattr(config, "pra_refinement_keep_ratio", 0.7)
        )
        return refined_feats, refine_diag

    return base_feats, {
        "avg_instability": 0.0,
        "avg_stable_ratio": 1.0,
    }


def get_pseudo_label_mask(teacher_probs, config):
    pseudo_conf, pseudo_targets = torch.max(teacher_probs, dim=1)
    pseudo_mask = pseudo_conf > config.pseudo_threshold

    top2_probs = torch.topk(teacher_probs, k=min(2, teacher_probs.shape[1]), dim=1).values
    if top2_probs.shape[1] >= 2:
        pseudo_margin = top2_probs[:, 0] - top2_probs[:, 1]
    else:
        pseudo_margin = torch.ones_like(pseudo_conf)

    pseudo_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=1)
    pseudo_entropy = pseudo_entropy / np.log(teacher_probs.shape[1])

    selection = getattr(config, "pseudo_selection", "confidence")
    if selection in {"margin", "confidence_margin", "hybrid"}:
        pseudo_mask = pseudo_mask & (pseudo_margin > config.pseudo_margin_threshold)
    if selection in {"entropy", "confidence_entropy", "hybrid"}:
        pseudo_mask = pseudo_mask & (pseudo_entropy < config.pseudo_entropy_threshold)

    diagnostics = {
        "confidence": pseudo_conf,
        "margin": pseudo_margin,
        "entropy": pseudo_entropy,
    }
    return pseudo_targets, pseudo_mask, diagnostics



def train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(splits, config, config.balance_source)

    # Setup model
    pretrained_path = f"{config.weights}/fold_{fold_num}"
    pretrained_weights = torch.load(f"{pretrained_path}/model.pt")["state_dict"]
    student.load_state_dict(pretrained_weights)
    teacher = deepcopy(student)
    student.to(device)
    teacher.to(device)

    # Training setup
    global_step, best_f1 = 0, 0
    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_loss_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    steps_per_epoch = config.steps_per_epoch

    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)

    source_iter = iter(cycle(source_loader))
    target_iter = iter(cycle(target_loader))
    min_shift, max_shift = -config.max_temporal_shift, config.max_temporal_shift
    target_to_source_shift = 0
    source_bank, source_bank_valid = None, None
    target_bank, target_bank_valid = None, None

    # To evaluate how well we estimate class distribution
    target_labels = target_loader_no_aug.dataset.get_labels()
    actual_class_distr = estimate_class_distribution(target_labels, config.num_classes)

    # estimate an initial guess for shift using Inception Score
    if config.estimate_shift:
        shift_estimator = 'IS' if config.shift_estimator == 'AM' else config.shift_estimator
        target_to_source_shift = estimate_temporal_shift(teacher, target_loader_no_aug, device, min_shift=min_shift, max_shift=max_shift, sample_size=config.sample_size, shift_estimator=shift_estimator)
        if target_to_source_shift >= 0:
            min_shift = 0
        else:
            max_shift = 0

        # Use estimated shift to get initial pseudo labels
        pseudo_softmaxes = get_pseudo_labels(teacher, target_loader_no_aug, device, target_to_source_shift, n=None)
        all_pseudo_labels = torch.max(pseudo_softmaxes, dim=1)[1]

    source_to_target_shift = 0
    for epoch in range(config.epochs):
        progress_bar = tqdm(
            range(steps_per_epoch),
            desc=f"TimeMatch Epoch {epoch + 1}/{config.epochs}",
            disable=should_disable_tqdm(config),
        )
        loss_meter = AverageMeter()
        point_loss_meter = AverageMeter()
        relation_loss_meter = AverageMeter()
        pra_batches_enabled = 0
        pra_batches_with_pairs = 0
        pra_batches_with_points = 0
        pra_active_points_total = 0
        pra_shared_classes_total = 0
        pra_active_pairs_total = 0
        pra_source_samples_total = 0
        pra_target_samples_total = 0
        pra_source_classes_updated_total = 0
        pra_target_classes_updated_total = 0
        pra_source_bank_classes_total = 0
        pra_target_bank_classes_total = 0

        if config.estimate_shift:
            confident_pseudo_labels = np.array([])
            if 'all_pseudo_labels' in locals():
                pseudo_labels_np = np.array(all_pseudo_labels)
                pseudo_mask_np = np.array(all_pseudo_mask, dtype=bool) if 'all_pseudo_mask' in locals() else None
                if pseudo_mask_np is not None and len(pseudo_mask_np) == len(pseudo_labels_np):
                    confident_pseudo_labels = pseudo_labels_np[pseudo_mask_np]
                elif len(pseudo_labels_np) > 0:
                    confident_pseudo_labels = pseudo_labels_np

            if len(confident_pseudo_labels) > 0:
                estimated_class_distr = estimate_class_distribution(confident_pseudo_labels, config.num_classes)
            else:
                estimated_class_distr = estimate_class_distribution(all_pseudo_labels, config.num_classes)

            writer.add_scalar("train/kl_d", kl_divergence(actual_class_distr, estimated_class_distr), epoch)
            target_to_source_shift = estimate_temporal_shift(teacher,
                    target_loader_no_aug, device, estimated_class_distr,
                    min_shift=min_shift, max_shift=max_shift, sample_size=config.sample_size,
                    shift_estimator=config.shift_estimator)
            if config.shift_source:
                source_to_target_shift = -target_to_source_shift
            else:
                source_to_target_shift = 0
            if epoch == 0:
                min_shift, max_shift = min(target_to_source_shift, 0), max(0, target_to_source_shift)
            writer.add_scalar("train/temporal_shift", target_to_source_shift, epoch)

        student.train()
        teacher.eval()  # don't update BN or use dropout for teacher

        all_labels, all_pseudo_labels, all_pseudo_mask = [], [], []
        for step in progress_bar:
            sample_source, (sample_target_weak, sample_target_strong) = next(source_iter), next(target_iter)

            # Get pseudo labels from teacher
            pixels_t_weak, mask_t_weak, position_t_weak, extra_t_weak = to_cuda(sample_target_weak, device)
            with torch.no_grad():
                teacher_preds = F.softmax(
                    teacher.forward(
                        pixels_t_weak,
                        mask_t_weak,
                        position_t_weak + target_to_source_shift,
                        extra_t_weak,
                    ),
                    dim=1,
                )
            pseudo_targets, pseudo_mask, pseudo_diag = get_pseudo_label_mask(teacher_preds, config)

            # Update student on shifted source data and pseudo-labeled target data
            pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
            source_labels = sample_source['label'].cuda(device, non_blocking=True)
            pixels_t, mask_t, position_t, extra_t = to_cuda(sample_target_strong, device)
            logits_target = None
            target_feats = None
            source_feats = None
            loss_target = 0.0
            loss_point = pixels_s.new_tensor(0.0)
            loss_relation = pixels_s.new_tensor(0.0)
            relation_stats = {
                'shared_classes': 0,
                'active_points': 0,
                'active_pairs': 0,
                'source_samples_used': 0,
                'target_samples_used': 0,
                'enabled': 0,
                'source_classes_updated': 0,
                'target_classes_updated': 0,
                'source_bank_classes': 0,
                'target_bank_classes': 0,
                'source_avg_instability': 0.0,
                'target_avg_instability': 0.0,
                'source_avg_stable_ratio': 1.0,
                'target_avg_stable_ratio': 1.0,
            }
            if config.domain_specific_bn:
                return_sequence_feats = getattr(config, "pra_use_refined_prototypes", False)
                source_outputs = student.forward(
                    pixels_s,
                    mask_s,
                    position_s + source_to_target_shift,
                    extra_s,
                    return_feats=True,
                    return_sequence_feats=return_sequence_feats,
                )
                if return_sequence_feats:
                    logits_source, source_feats, source_sequence_feats = source_outputs
                else:
                    logits_source, source_feats = source_outputs
                    source_sequence_feats = None
                if len(torch.nonzero(pseudo_mask)) >= 2:  # at least 2 examples required for BN
                    target_outputs = student.forward(
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        position_t[pseudo_mask],
                        extra_t[pseudo_mask],
                        return_feats=True,
                        return_sequence_feats=return_sequence_feats,
                    )
                    if return_sequence_feats:
                        logits_target, target_feats, target_sequence_feats = target_outputs
                    else:
                        logits_target, target_feats = target_outputs
                        target_sequence_feats = None
            else:
                pixels = torch.cat([pixels_s, pixels_t[pseudo_mask]])
                mask = torch.cat([mask_s, mask_t[pseudo_mask]])
                position = torch.cat([position_s + source_to_target_shift, position_t[pseudo_mask]])
                extra = torch.cat([extra_s, extra_t[pseudo_mask]])
                return_sequence_feats = getattr(config, "pra_use_refined_prototypes", False)
                outputs = student.forward(
                    pixels,
                    mask,
                    position,
                    extra,
                    return_feats=True,
                    return_sequence_feats=return_sequence_feats,
                )
                if return_sequence_feats:
                    logits, feats, sequence_feats = outputs
                else:
                    logits, feats = outputs
                    sequence_feats = None
                logits_source, logits_target = logits[:config.batch_size], logits[config.batch_size:]
                source_feats, target_feats = feats[:config.batch_size], feats[config.batch_size:]
                if sequence_feats is not None:
                    source_sequence_feats = sequence_feats[:config.batch_size]
                    target_sequence_feats = sequence_feats[config.batch_size:]
                else:
                    source_sequence_feats = None
                    target_sequence_feats = None

            loss_source = criterion(logits_source, source_labels)
            if logits_target is not None:
                loss_target = criterion(logits_target, pseudo_targets[pseudo_mask])
                use_pra = (
                    config.use_prototype_relation_alignment
                    and epoch >= config.pra_warmup_epochs
                )
                if use_pra:
                    if source_bank is None:
                        source_bank, source_bank_valid = init_prototype_bank(
                            config.num_classes, source_feats.shape[1], source_feats.device
                        )
                        target_bank, target_bank_valid = init_prototype_bank(
                            config.num_classes, target_feats.shape[1], target_feats.device
                        )

                    source_proto_feats, source_refine_diag = extract_refined_prototype_features(
                        source_feats, source_sequence_feats, config
                    )
                    target_proto_feats, target_refine_diag = extract_refined_prototype_features(
                        target_feats, target_sequence_feats, config
                    )

                    source_batch_prototypes, source_batch_counts = compute_class_prototypes(
                        source_proto_feats,
                        source_labels,
                        ignore_class_id=getattr(config, "unknown_class_idx", None),
                        min_samples=config.pra_min_samples_per_class,
                    )
                    target_batch_prototypes, target_batch_counts = compute_class_prototypes(
                        target_proto_feats,
                        pseudo_targets[pseudo_mask],
                        ignore_class_id=getattr(config, "unknown_class_idx", None),
                        min_samples=config.pra_min_samples_per_class,
                    )
                    updated_source_classes = update_prototype_bank(
                        source_bank,
                        source_bank_valid,
                        source_batch_prototypes,
                        momentum=config.pra_bank_momentum,
                    )
                    updated_target_classes = update_prototype_bank(
                        target_bank,
                        target_bank_valid,
                        target_batch_prototypes,
                        momentum=config.pra_bank_momentum,
                    )

                    merged_source_prototypes = merge_prototypes_with_bank(
                        source_batch_prototypes,
                        source_bank,
                        source_bank_valid,
                        ignore_class_id=getattr(config, "unknown_class_idx", None),
                    )
                    merged_target_prototypes = merge_prototypes_with_bank(
                        target_batch_prototypes,
                        target_bank,
                        target_bank_valid,
                        ignore_class_id=getattr(config, "unknown_class_idx", None),
                    )

                    loss_relation, relation_diag = prototype_relation_alignment_loss(
                        merged_source_prototypes,
                        merged_target_prototypes,
                        normalize_geometry=getattr(config, "pra_normalize_geometry", True),
                    )
                    loss_point, point_diag = prototype_point_alignment_loss(
                        merged_source_prototypes,
                        merged_target_prototypes,
                        normalize_geometry=getattr(config, "pra_normalize_geometry", True),
                    )
                    relation_stats = {
                        'shared_classes': max(relation_diag['shared_classes'], point_diag['shared_classes']),
                        'active_points': point_diag['active_points'],
                        'active_pairs': relation_diag['active_pairs'],
                        'source_samples_used': int(sum(source_batch_counts.values())),
                        'target_samples_used': int(sum(target_batch_counts.values())),
                        'enabled': int(relation_diag['enabled'] or point_diag['enabled']),
                        'source_classes_updated': len(updated_source_classes),
                        'target_classes_updated': len(updated_target_classes),
                        'source_bank_classes': int(source_bank_valid.sum().item()),
                        'target_bank_classes': int(target_bank_valid.sum().item()),
                        'source_avg_instability': source_refine_diag['avg_instability'],
                        'target_avg_instability': target_refine_diag['avg_instability'],
                        'source_avg_stable_ratio': source_refine_diag['avg_stable_ratio'],
                        'target_avg_stable_ratio': target_refine_diag['avg_stable_ratio'],
                    }
            loss = (
                loss_source
                + config.trade_off * loss_target
                + config.pra_point_trade_off * loss_point
                + config.pra_trade_off * loss_relation
            )

            # compute loss and backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)

            # Metrics
            loss_meter.update(loss.item())
            if relation_stats['enabled']:
                point_loss_meter.update(float(loss_point.item()))
                relation_loss_meter.update(float(loss_relation.item()))
                pra_batches_enabled += 1
            if relation_stats['active_points'] > 0:
                pra_batches_with_points += 1
                pra_active_points_total += relation_stats['active_points']
            if relation_stats['active_pairs'] > 0:
                pra_batches_with_pairs += 1
                pra_shared_classes_total += relation_stats['shared_classes']
                pra_active_pairs_total += relation_stats['active_pairs']
                pra_source_samples_total += relation_stats['source_samples_used']
                pra_target_samples_total += relation_stats['target_samples_used']
            pra_source_classes_updated_total += relation_stats['source_classes_updated']
            pra_target_classes_updated_total += relation_stats['target_classes_updated']
            pra_source_bank_classes_total += relation_stats['source_bank_classes']
            pra_target_bank_classes_total += relation_stats['target_bank_classes']

            if should_disable_tqdm(config):
                if step % config.log_step == 0:
                    print(
                        f"Epoch {epoch + 1}/{config.epochs} Step {step}/{steps_per_epoch}: "
                        f"loss={loss_meter.avg:.3f}, source={float(loss_source.item()):.3f}, "
                        f"target={float(loss_target) if isinstance(loss_target, float) else float(loss_target.item()):.3f}, "
                        f"point={float(loss_point.item()):.5f}, edge={float(loss_relation.item()):.5f}"
                    )
            else:
                progress_bar.set_postfix(
                    loss=f"{loss_meter.avg:.3f}",
                    point=f"{point_loss_meter.avg:.5f}" if point_loss_meter.count > 0 else "0.00000",
                    rel=f"{relation_loss_meter.avg:.5f}" if relation_loss_meter.count > 0 else "0.00000",
                )
            all_labels.extend(sample_target_weak['label'].tolist())
            all_pseudo_labels.extend(pseudo_targets.tolist())
            all_pseudo_mask.extend(pseudo_mask.tolist())

            if step % config.log_step == 0:
                writer.add_scalar("train/loss", loss_meter.val, global_step)
                writer.add_scalar("train/loss_source", float(loss_source.item()), global_step)
                writer.add_scalar("train/loss_target", float(loss_target) if isinstance(loss_target, float) else float(loss_target.item()), global_step)
                writer.add_scalar("train/loss_point", float(loss_point.item()), global_step)
                writer.add_scalar("train/loss_relation", float(loss_relation.item()), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/target_updates", len(torch.nonzero(pseudo_mask)), global_step)
                writer.add_scalar("train/pseudo_confidence", float(pseudo_diag["confidence"].mean().item()), global_step)
                writer.add_scalar("train/pseudo_margin", float(pseudo_diag["margin"].mean().item()), global_step)
                writer.add_scalar("train/pseudo_entropy", float(pseudo_diag["entropy"].mean().item()), global_step)
                writer.add_scalar("train/source_avg_instability", relation_stats["source_avg_instability"], global_step)
                writer.add_scalar("train/target_avg_instability", relation_stats["target_avg_instability"], global_step)
                writer.add_scalar("train/source_avg_stable_ratio", relation_stats["source_avg_stable_ratio"], global_step)
                writer.add_scalar("train/target_avg_stable_ratio", relation_stats["target_avg_stable_ratio"], global_step)

            global_step += 1

        progress_bar.close()

        # Evaluate pseudo labels
        all_labels, all_pseudo_labels, all_pseudo_mask = np.array(all_labels), np.array(all_pseudo_labels), np.array(all_pseudo_mask)
        pseudo_count = all_pseudo_mask.sum()
        conf_pseudo_f1 = sklearn.metrics.f1_score(all_labels[all_pseudo_mask], all_pseudo_labels[all_pseudo_mask], average='macro', zero_division=0)
        print(f"Teacher pseudo label F1 {conf_pseudo_f1:.3f} (n={pseudo_count})")
        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        pra_active_batches_denom = max(1, pra_batches_with_pairs)
        pra_point_batches_denom = max(1, pra_batches_with_points)
        avg_point_loss = point_loss_meter.avg if point_loss_meter.count > 0 else 0.0
        avg_relation_loss = relation_loss_meter.avg if relation_loss_meter.count > 0 else 0.0
        avg_active_points = pra_active_points_total / pra_point_batches_denom
        avg_shared_classes = pra_shared_classes_total / pra_active_batches_denom
        avg_active_pairs = pra_active_pairs_total / pra_active_batches_denom
        avg_source_samples = pra_source_samples_total / pra_active_batches_denom
        avg_target_samples = pra_target_samples_total / pra_active_batches_denom
        avg_source_classes_updated = pra_source_classes_updated_total / max(1, steps_per_epoch)
        avg_target_classes_updated = pra_target_classes_updated_total / max(1, steps_per_epoch)
        avg_source_bank_classes = pra_source_bank_classes_total / max(1, steps_per_epoch)
        avg_target_bank_classes = pra_target_bank_classes_total / max(1, steps_per_epoch)
        writer.add_scalar("train/pra_avg_point_loss", avg_point_loss, epoch)
        writer.add_scalar("train/pra_avg_relation_loss", avg_relation_loss, epoch)
        writer.add_scalar("train/pra_batches_enabled", pra_batches_enabled, epoch)
        writer.add_scalar("train/pra_batches_with_points", pra_batches_with_points, epoch)
        writer.add_scalar("train/pra_batches_with_pairs", pra_batches_with_pairs, epoch)
        writer.add_scalar("train/pra_avg_active_points", avg_active_points, epoch)
        writer.add_scalar("train/pra_avg_shared_classes", avg_shared_classes, epoch)
        writer.add_scalar("train/pra_avg_active_pairs", avg_active_pairs, epoch)
        writer.add_scalar("train/pra_avg_source_samples", avg_source_samples, epoch)
        writer.add_scalar("train/pra_avg_target_samples", avg_target_samples, epoch)
        writer.add_scalar("train/pra_avg_source_classes_updated", avg_source_classes_updated, epoch)
        writer.add_scalar("train/pra_avg_target_classes_updated", avg_target_classes_updated, epoch)
        writer.add_scalar("train/pra_avg_source_bank_classes", avg_source_bank_classes, epoch)
        writer.add_scalar("train/pra_avg_target_bank_classes", avg_target_bank_classes, epoch)
        print(
            "PRA diagnostics: "
            f"enabled_batches={pra_batches_enabled}/{steps_per_epoch}, "
            f"active_point_batches={pra_batches_with_points}/{steps_per_epoch}, "
            f"active_pair_batches={pra_batches_with_pairs}/{steps_per_epoch}, "
            f"avg_point_loss={avg_point_loss:.6f}, "
            f"avg_relation_loss={avg_relation_loss:.6f}, "
            f"avg_active_points={avg_active_points:.2f}, "
            f"avg_shared_classes={avg_shared_classes:.2f}, "
            f"avg_active_pairs={avg_active_pairs:.2f}, "
            f"avg_source_samples={avg_source_samples:.2f}, "
            f"avg_target_samples={avg_target_samples:.2f}, "
            f"avg_source_classes_updated={avg_source_classes_updated:.2f}, "
            f"avg_target_classes_updated={avg_target_classes_updated:.2f}, "
            f"avg_source_bank_classes={avg_source_bank_classes:.2f}, "
            f"avg_target_bank_classes={avg_target_bank_classes:.2f}"
        )

        if config.run_validation:
            if config.output_student:
                student.eval()
                best_f1 = validation(best_f1, best_model_path, config, criterion, device, epoch, student, val_loader, writer)
            else:
                teacher.eval()
                best_f1 = validation(best_f1, best_model_path, config, criterion, device, epoch, teacher, val_loader, writer)

    # If validation is disabled, fall back to the final epoch weights.
    if not config.run_validation:
        if config.output_student:
            torch.save({'state_dict': student.state_dict()}, best_model_path)
        else:
            torch.save({'state_dict': teacher.state_dict()}, best_model_path)

def estimate_class_distribution(labels, num_classes):
    return np.bincount(labels, minlength=num_classes) / len(labels)

def kl_divergence(actual, estimated):
    return np.sum(actual * (np.log(actual + 1e-5) - np.log(estimated + 1e-5)))

def sync_teacher_temporal_encoder(student, teacher):
    student_encoder = getattr(student, "temporal_encoder", None)
    teacher_encoder = getattr(teacher, "temporal_encoder", None)

    if student_encoder is None or teacher_encoder is None:
        return

    student_positional_enc = getattr(student_encoder, "positional_enc", None)
    teacher_positional_enc = getattr(teacher_encoder, "positional_enc", None)

    if student_positional_enc is None or teacher_positional_enc is None:
        return

    if teacher_positional_enc.weight.shape == student_positional_enc.weight.shape:
        teacher_encoder.max_temporal_shift = student_encoder.max_temporal_shift
        return

    teacher_encoder.max_temporal_shift = student_encoder.max_temporal_shift
    teacher_encoder.positional_enc = nn.Embedding.from_pretrained(
        student_positional_enc.weight.detach().clone(),
        freeze=True,
    ).to(student_positional_enc.weight.device)

@torch.no_grad()
def update_ema_variables(model, ema, decay=0.99):
    sync_teacher_temporal_encoder(model, ema)
    for ema_v, model_v in zip(ema.state_dict().values(), model.state_dict().values()):
        ema_v.copy_(decay * ema_v + (1. - decay) * model_v)


def get_data_loaders(splits, config, balance_source=True):
    weak_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        Normalize(),
        ToTensor(),
    ])

    source_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        Normalize(),
        ToTensor(),
    ])

    target_strong_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
        Normalize(),
        ToTensor(),
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, source_aug,
            indices=splits[config.source]['train'],)

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    target_dataset = PixelSetData(config.data_root, config.target,
            config.classes, None,
            indices=splits[config.target]['train'])

    strong_dataset = deepcopy(target_dataset)
    strong_dataset.transform = target_strong_aug
    weak_dataset = deepcopy(target_dataset)
    weak_dataset.transform = weak_aug
    target_dataset_weak_strong = TupleDataset(weak_dataset, strong_dataset)

    no_aug_dataset = deepcopy(target_dataset)
    no_aug_dataset.transform = weak_aug
    # For shift estimation
    target_loader_no_aug = data.DataLoader(
        no_aug_dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # For mean teacher training
    target_loader_weak_strong = data.DataLoader(
        target_dataset_weak_strong,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')
    print(f'size of target dataset: {len(target_dataset)} ({len(target_loader_weak_strong)} batches)')

    return source_loader, target_loader_no_aug, target_loader_weak_strong


class TupleDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.weak = dataset1
        self.strong = dataset2
        assert len(dataset1) == len(dataset2)
        self.len = len(dataset1)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.weak[index], self.strong[index])


@torch.no_grad()
def estimate_temporal_shift(model, target_loader, device, class_distribution=None, min_shift=-60, max_shift=60, sample_size=100, shift_estimator='IS'):
    shifts = list(range(min_shift, max_shift + 1))
    model.eval()
    if sample_size is None:
        sample_size = len(target_loader)

    target_iter = iter(target_loader)
    shift_softmaxes, labels = [], []
    for _ in tqdm(
        range(sample_size),
        desc=f'Estimating shift between [{min_shift}, {max_shift}]',
        disable=should_disable_tqdm(),
    ):
        sample = next(target_iter)
        labels.extend(sample['label'].tolist())
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        spatial_feats = model.spatial_encoder.forward(pixels, valid_pixels, extra)
        shift_logits = torch.stack([model.decoder(model.temporal_encoder(spatial_feats, positions + shift)) for shift in shifts], dim=1)
        shift_probs = F.softmax(shift_logits, dim=2)
        shift_softmaxes.append(shift_probs)
    shift_softmaxes = torch.cat(shift_softmaxes).cpu().numpy()  # (N, n_shifts, n_classes)
    labels = np.array(labels)
    shift_predictions = np.argmax(shift_softmaxes, axis=2)  # (N, n_shifts)

    # shift_f1_scores = [f1_score(labels, shift_predictions, num_classes) for shift_predictions in all_shift_predictions]
    shift_acc_scores = [(labels == predictions).mean() for predictions in np.moveaxis(shift_predictions, 0, 1)]
    best_acc_idx = int(np.argmax(shift_acc_scores))
    best_acc_shift = shifts[best_acc_idx]
    best_acc_score = float(np.max(shift_acc_scores))
    print(f"Most accurate shift {best_acc_shift} with {best_acc_score:.3f}")

    p_yx = shift_softmaxes # (N, n_shifts, n_classes)
    p_y = shift_softmaxes.mean(axis=0)  # (n_shifts, n_classes)


    if shift_estimator == 'IS':
        inception_score = np.mean(np.sum(p_yx * (np.log(p_yx + 1e-5) - np.log(p_y[np.newaxis] + 1e-5)), axis=2), axis=0)  # (n_shifts)

        shift_indices_ranked = np.argsort(inception_score)[::-1]  # max is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best Inception Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")
        return best_shift

    elif shift_estimator == 'ENT':
        entropy_score = -np.mean(np.sum(p_yx * np.log(p_yx + 1e-5), axis=2), axis=0)  # (n_shifts)
        shift_indices_ranked = np.argsort(entropy_score)  # min is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best Entropy Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")
        return best_shift

    elif shift_estimator == 'AM':
        assert class_distribution is not None, 'Target class distribution required to compute AM score'

        # estimate class distribution
        one_hot_p_y = np.zeros_like(p_y)
        for i in range(len(shifts)):
            one_hot = np.zeros((shift_softmaxes.shape[0], shift_softmaxes.shape[-1]))  # (n, classes)
            one_hot[np.arange(one_hot.shape[0]), shift_predictions[:, i]] = 1
            one_hot_p_y[i] = one_hot.mean(axis=0)

        c_train = class_distribution
        # kl_d = np.sum(c_train * (np.log(c_train + 1e-5) - np.log(p_y + 1e-5)), axis=1) # soft class distr
        kl_d = np.sum(c_train * (np.log(c_train + 1e-5) - np.log(one_hot_p_y + 1e-5)), axis=1)
        entropy = np.mean(np.sum(-p_yx * np.log(p_yx + 1e-5), axis=2), axis=0)
        am = kl_d + entropy
        acc_scores = np.array(shift_acc_scores)
        # Keep AM from jumping to a far-away shift when several candidates have similar
        # classification quality. We only search among shifts close to the best accuracy.
        acc_margin = 0.02
        candidate_indices = np.where(acc_scores >= best_acc_score - acc_margin)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.array([best_acc_idx])

        local_radius = 6
        candidate_indices = np.array([
            idx for idx in candidate_indices
            if abs(shifts[idx] - best_acc_shift) <= local_radius
        ])
        if len(candidate_indices) == 0:
            candidate_indices = np.array([best_acc_idx])

        best_local = candidate_indices[np.argmin(am[candidate_indices])]
        best_shift_idx = int(best_local)
        best_shift = shifts[best_shift_idx]
        candidate_shifts = [shifts[idx] for idx in candidate_indices.tolist()]
        print(
            f"Best AM Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f} "
            f"(candidates near accuracy peak: {candidate_shifts})"
        )

        return best_shift
    elif shift_estimator == 'ACC':  # for upperbound comparison
        shift_indices_ranked = np.argsort(shift_acc_scores)[::-1]  # max is best
        return shifts[np.argmax(shift_acc_scores)]
    else:
        raise NotImplementedError




@torch.no_grad()
def get_pseudo_labels(model, data_loader, device, best_shift, n=500):
    model.eval()
    pseudo_softmaxes = []
    indices = []
    for i, sample in enumerate(tqdm(data_loader, "computing pseudo labels", disable=should_disable_tqdm())):
        if n is not None and i == n:
            break
        indices.extend(sample["index"].tolist())

        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        logits = model.forward(pixels, valid_pixels, positions + best_shift, extra)
        probs = F.softmax(logits, dim=1).cpu()
        pseudo_softmaxes.extend(probs.tolist())

    indices = torch.as_tensor(indices)
    pseudo_softmaxes = torch.as_tensor(pseudo_softmaxes)
    pseudo_softmaxes = pseudo_softmaxes[torch.argsort(indices)]

    return pseudo_softmaxes
