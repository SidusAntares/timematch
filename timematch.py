from torch.utils.data.sampler import WeightedRandomSampler
import sklearn.metrics
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from ideas.source_phase_compactness import (
    SourcePhaseWeightTracker,
    build_source_phase_partition_spec,
    compute_source_structure_loss,
    describe_source_phase_partition_spec,
)
from ideas.source_feature_reshaper import (
    build_source_feature_reshaper,
    compute_dual_path_relation_regularization,
    compute_source_feature_reshaper_regularization,
)
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    RandomTemporalShift,
    Identity,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, to_cuda, cycle


def _check_temporal_index_range(model, positions, applied_shift, tag):
    temporal_encoder = model.temporal_encoder
    min_pos = int(positions.min().item())
    max_pos = int(positions.max().item())
    min_idx = min_pos + applied_shift + temporal_encoder.max_temporal_shift
    max_idx = max_pos + applied_shift + temporal_encoder.max_temporal_shift
    table_size = temporal_encoder.positional_enc.num_embeddings

    if min_idx < 0 or max_idx >= table_size:
        raise ValueError(
            f"{tag} temporal indices out of range: "
            f"positions=[{min_pos}, {max_pos}], shift={applied_shift}, "
            f"embedding_indices=[{min_idx}, {max_idx}], table_size={table_size}. "
            "This usually means an extra temporal shift was applied on top of TimeMatch "
            "alignment or the positional encoding range is inconsistent with the dataset dates."
        )


def train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    assert not getattr(config, "with_shift_aug", False), (
        "TimeMatch with source phase compactness / v2.3 phase-aware partition must not enable "
        "RandomTemporalShift-style source-side augmentation, because compactness is defined on the original source time axis."
    )
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(splits, config, config.balance_source)

    # Setup model
    pretrained_path = f"{config.weights}/fold_{fold_num}"
    pretrained_checkpoint = torch.load(f"{pretrained_path}/model.pt", weights_only=False)
    pretrained_weights = pretrained_checkpoint["state_dict"]
    student.load_state_dict(pretrained_weights)
    teacher = deepcopy(student)
    student.to(device)
    teacher.to(device)

    source_feature_reshaper = build_source_feature_reshaper(
        kind=getattr(config, "source_feature_reshaper", "none"),
        feature_dim=student.spatial_encoder.output_dim,
        strength=getattr(config, "source_feature_reshaper_strength", 0.10),
        kernel_size=getattr(config, "source_feature_reshaper_kernel_size", 3),
    )
    if source_feature_reshaper is not None:
        source_feature_reshaper.to(device)
        if "source_feature_reshaper_state_dict" in pretrained_checkpoint:
            source_feature_reshaper.load_state_dict(pretrained_checkpoint["source_feature_reshaper_state_dict"])

    # Training setup
    global_step, best_f1 = 0, 0
    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_loss_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    steps_per_epoch = config.steps_per_epoch

    params = list(student.parameters())
    if source_feature_reshaper is not None:
        params += list(source_feature_reshaper.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)
    source_phase_partition_spec = build_source_phase_partition_spec(
        source_loader.dataset.date_positions,
        mode=getattr(config, "source_phase_partition_mode", "uniform"),
        phase_count=getattr(config, "source_phase_count", 5),
        gap_threshold=getattr(config, "source_phase_gap_threshold", 45),
        min_points=getattr(config, "source_phase_min_points", 3),
        max_points=getattr(config, "source_phase_max_points", 8),
        max_span=getattr(config, "source_phase_max_span", 120),
    )
    print("source phase partition:", describe_source_phase_partition_spec(source_phase_partition_spec))
    phase_weight_tracker = SourcePhaseWeightTracker(
        phase_count=source_phase_partition_spec["phase_count"],
        phase_partition_spec=source_phase_partition_spec,
        min_sample_points_per_phase=getattr(config, "source_phase_min_sample_points", 2),
    )

    source_iter = iter(cycle(source_loader))
    target_iter = iter(cycle(target_loader))
    min_shift, max_shift = -config.max_temporal_shift, config.max_temporal_shift
    target_to_source_shift = 0
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
        progress_bar = tqdm(range(steps_per_epoch), desc=f"TimeMatch Epoch {epoch + 1}/{config.epochs}")
        loss_meter = AverageMeter()

        if config.estimate_shift:
            estimated_class_distr = estimate_class_distribution(all_pseudo_labels, config.num_classes)
            writer.add_scalar("train/kl_d", kl_divergence(actual_class_distr, estimated_class_distr), epoch)
            target_to_source_shift = estimate_temporal_shift(teacher,
                    target_loader_no_aug, device, estimated_class_distr,
                    min_shift=min_shift, max_shift=max_shift, sample_size=config.sample_size,
                    shift_estimator=config.shift_estimator)
            if epoch == 0:
                if config.shift_source:
                    source_to_target_shift = -target_to_source_shift
                else:
                    source_to_target_shift = 0
                min_shift, max_shift = min(target_to_source_shift, 0), max(0, target_to_source_shift)
            writer.add_scalar("train/temporal_shift", target_to_source_shift, epoch)

        student.train()
        teacher.eval()  # don't update BN or use dropout for teacher
        if source_feature_reshaper is not None:
            source_feature_reshaper.train()

        all_labels, all_pseudo_labels, all_pseudo_mask = [], [], []
        for step in progress_bar:
            sample_source, (sample_target_weak, sample_target_strong) = next(source_iter), next(target_iter)

            # Get pseudo labels from teacher
            pixels_t_weak, mask_t_weak, position_t_weak, extra_t_weak = to_cuda(sample_target_weak, device)
            with torch.no_grad():
                teacher_preds = F.softmax(teacher.forward(pixels_t_weak, mask_t_weak, position_t_weak + target_to_source_shift, extra_t_weak), dim=1)
            pseudo_conf, pseudo_targets = torch.max(teacher_preds, dim=1)
            pseudo_mask = pseudo_conf > config.pseudo_threshold
            target_update_count = int(pseudo_mask.sum().item())

            # Update student on shifted source data and pseudo-labeled target data
            pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
            source_labels = sample_source['label'].cuda(device, non_blocking=True)
            pixels_t, mask_t, position_t, extra_t = to_cuda(sample_target_strong, device)
            logits_target = None
            loss_target = 0.0
            reshaper_loss = pixels_s.sum() * 0.0
            reshaper_logs = {}
            compact_loss = pixels_s.sum() * 0.0
            compact_logs = {}
            dual_relation_loss = pixels_s.sum() * 0.0
            dual_relation_logs = {}
            loss_source_reshaped = pixels_s.sum() * 0.0
            if config.domain_specific_bn:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                spatial_feats_source_raw = student.spatial_encoder(pixels_s, mask_s, extra_s)
                temporal_feats_source_raw = student.temporal_encoder(
                    spatial_feats_source_raw,
                    position_s + source_to_target_shift,
                )
                logits_source_raw = student.decoder(temporal_feats_source_raw)
                logits_source = logits_source_raw
                if source_feature_reshaper is not None:
                    spatial_feats_source = source_feature_reshaper(
                        spatial_feats_source_raw.detach(),
                        positions=position_s,
                        labels=source_labels,
                    )
                    reshaper_loss, reshaper_logs = compute_source_feature_reshaper_regularization(
                        spatial_feats_source_raw.detach(),
                        spatial_feats_source,
                    )
                    compact_loss, compact_logs = compute_source_structure_loss(
                        spatial_feats_source,
                        position_s,
                        source_labels,
                        weight_tracker=phase_weight_tracker,
                        version=getattr(config, "source_structure_loss_version", "compactness"),
                        intra_trade_off=getattr(config, "source_structure_intra_trade_off", 1.0),
                        amplitude_trade_off=getattr(config, "source_structure_amplitude_trade_off", 0.25),
                        interphase_trade_off=getattr(config, "source_structure_interphase_trade_off", 0.25),
                    )
                    temporal_feats_source = student.temporal_encoder(
                        spatial_feats_source.detach(),
                        position_s + source_to_target_shift,
                    )
                    logits_source = student.decoder(temporal_feats_source)
                    loss_source_reshaped = criterion(logits_source, source_labels)
                    if getattr(config, "source_feature_dual_path", False):
                        dual_relation_loss, dual_relation_logs = compute_dual_path_relation_regularization(
                            logits_source_raw,
                            logits_source,
                            raw_temporal_feats=temporal_feats_source_raw,
                            reshaped_temporal_feats=temporal_feats_source,
                        )
                if target_update_count >= 2:  # at least 2 examples required for BN
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                    logits_target = student.forward(pixels_t[pseudo_mask], mask_t[pseudo_mask], position_t[pseudo_mask], extra_t[pseudo_mask])
            else:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                spatial_feats_source_raw = student.spatial_encoder(pixels_s, mask_s, extra_s)
                temporal_feats_source_raw = student.temporal_encoder(
                    spatial_feats_source_raw,
                    position_s + source_to_target_shift,
                )
                logits_source_raw = student.decoder(temporal_feats_source_raw)
                spatial_feats_source = spatial_feats_source_raw
                temporal_feats_source = temporal_feats_source_raw
                logits_source = logits_source_raw
                if source_feature_reshaper is not None:
                    spatial_feats_source = source_feature_reshaper(
                        spatial_feats_source_raw.detach(),
                        positions=position_s,
                        labels=source_labels,
                    )
                    reshaper_loss, reshaper_logs = compute_source_feature_reshaper_regularization(
                        spatial_feats_source_raw.detach(),
                        spatial_feats_source,
                    )
                    compact_loss, compact_logs = compute_source_structure_loss(
                        spatial_feats_source,
                        position_s,
                        source_labels,
                        weight_tracker=phase_weight_tracker,
                        version=getattr(config, "source_structure_loss_version", "compactness"),
                        intra_trade_off=getattr(config, "source_structure_intra_trade_off", 1.0),
                        amplitude_trade_off=getattr(config, "source_structure_amplitude_trade_off", 0.25),
                        interphase_trade_off=getattr(config, "source_structure_interphase_trade_off", 0.25),
                    )
                    temporal_feats_source = student.temporal_encoder(
                        spatial_feats_source.detach(),
                        position_s + source_to_target_shift,
                    )
                    logits_source = student.decoder(temporal_feats_source)
                    loss_source_reshaped = criterion(logits_source, source_labels)
                    if getattr(config, "source_feature_dual_path", False):
                        dual_relation_loss, dual_relation_logs = compute_dual_path_relation_regularization(
                            logits_source_raw,
                            logits_source,
                            raw_temporal_feats=temporal_feats_source_raw,
                            reshaped_temporal_feats=temporal_feats_source,
                        )

                if target_update_count > 0:
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                    logits_target = student.forward(
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        position_t[pseudo_mask],
                        extra_t[pseudo_mask],
                    )

            loss_source_raw = criterion(logits_source_raw, source_labels)
            if source_feature_reshaper is not None and getattr(config, "source_feature_dual_path", False):
                loss_source = (
                    loss_source_raw
                    + getattr(config, "source_feature_dual_cls_trade_off", 1.0) * loss_source_reshaped
                    + getattr(config, "source_feature_dual_relation_trade_off", 0.05) * dual_relation_loss
                )
            else:
                loss_source = criterion(logits_source, source_labels)
            if logits_target is not None:
                loss_target = criterion(logits_target, pseudo_targets[pseudo_mask])
            loss = loss_source + config.trade_off * loss_target
            if source_feature_reshaper is not None:
                loss = loss + compact_loss + getattr(config, "source_feature_reshaper_reg_trade_off", 0.0) * reshaper_loss

            # compute loss and backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)

            # Metrics
            loss_meter.update(loss.item())
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.3f}")
            all_labels.extend(sample_target_weak['label'].tolist())
            all_pseudo_labels.extend(pseudo_targets.tolist())
            all_pseudo_mask.extend(pseudo_mask.tolist())

            if step % config.log_step == 0:
                writer.add_scalar("train/loss", loss_meter.val, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/target_updates", target_update_count, global_step)
                if source_feature_reshaper is not None:
                    writer.add_scalar(
                        "train/source_feature_reshaper_reg_loss",
                        float(reshaper_logs.get("source_reshaper_reg_loss", 0.0)),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_phase_compactness_loss",
                        float(compact_logs.get("compactness_loss", 0.0)),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_structure_loss",
                        float(compact_logs.get("structure_loss", compact_logs.get("compactness_loss", 0.0))),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_cls_loss_raw",
                        float(loss_source_raw.detach().item()),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/source_cls_loss_reshaped",
                        float(loss_source_reshaped.detach().item()),
                        global_step,
                    )
                    if getattr(config, "source_feature_dual_path", False):
                        writer.add_scalar(
                            "train/source_dual_relation_loss",
                            float(dual_relation_logs.get("source_dual_relation_loss", 0.0)),
                            global_step,
                        )
                for name, value in reshaper_logs.items():
                    writer.add_scalar(f"train/{name}", value, global_step)
                for name, value in compact_logs.items():
                    if name != "compactness_loss":
                        writer.add_scalar(f"train/{name}", value, global_step)
                for name, value in dual_relation_logs.items():
                    writer.add_scalar(f"train/{name}", value, global_step)

            global_step += 1

        progress_bar.close()

        # Evaluate pseudo labels
        all_labels, all_pseudo_labels, all_pseudo_mask = np.array(all_labels), np.array(all_pseudo_labels), np.array(all_pseudo_mask)
        pseudo_count = all_pseudo_mask.sum()
        conf_pseudo_f1 = sklearn.metrics.f1_score(all_labels[all_pseudo_mask], all_pseudo_labels[all_pseudo_mask], average='macro', zero_division=0)
        print(f"Teacher pseudo label F1 {conf_pseudo_f1:.3f} (n={pseudo_count})")
        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        if config.run_validation:
            if config.output_student:
                student.eval()
                if source_feature_reshaper is not None:
                    source_feature_reshaper.eval()
                best_f1 = validation(
                    best_f1,
                    None,
                    config,
                    criterion,
                    device,
                    epoch,
                    student,
                    val_loader,
                    writer,
                    source_feature_reshaper=source_feature_reshaper,
                    apply_source_feature_reshaper=False,
                )
            else:
                teacher.eval()
                best_f1 = validation(
                    best_f1,
                    None,
                    config,
                    criterion,
                    device,
                    epoch,
                    teacher,
                    val_loader,
                    writer,
                    source_feature_reshaper=None,
                    apply_source_feature_reshaper=False,
                )

    # Save model final model 
    if config.output_student:
        checkpoint = {'state_dict': student.state_dict()}
        if source_feature_reshaper is not None:
            checkpoint['source_feature_reshaper_state_dict'] = source_feature_reshaper.state_dict()
        torch.save(checkpoint, best_model_path)
    else:
        checkpoint = {'state_dict': teacher.state_dict()}
        if source_feature_reshaper is not None:
            checkpoint['source_feature_reshaper_state_dict'] = source_feature_reshaper.state_dict()
        torch.save(checkpoint, best_model_path)

def estimate_class_distribution(labels, num_classes):
    return np.bincount(labels, minlength=num_classes) / len(labels)

def kl_divergence(actual, estimated):
    return np.sum(actual * (np.log(actual + 1e-5) - np.log(estimated + 1e-5)))

@torch.no_grad()
def update_ema_variables(model, ema, decay=0.99):
    for ema_v, model_v in zip(ema.state_dict().values(), model.state_dict().values()):
        ema_v.copy_(decay * ema_v + (1. - decay) * model_v)


def get_data_loaders(splits, config, balance_source=True):
    weak_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        Normalize(),
        ToTensor(),
    ])

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, strong_aug,
            indices=splits[config.source]['train'],
            closed_set=getattr(config, 'closed_set', False),)

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
            indices=splits[config.target]['train'],
            closed_set=getattr(config, 'closed_set', False))

    strong_dataset = deepcopy(target_dataset)
    strong_dataset.transform = strong_aug
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
    for _ in tqdm(range(sample_size), desc=f'Estimating shift between [{min_shift}, {max_shift}]'):
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
    print(f"Most accurate shift {shifts[np.argmax(shift_acc_scores)]} with {np.max(shift_acc_scores):.3f}")

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
        shift_indices_ranked = np.argsort(am)  # min is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best AM Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")

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
    for i, sample in enumerate(tqdm(data_loader, "computing pseudo labels")):
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
