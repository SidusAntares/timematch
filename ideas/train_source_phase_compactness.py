import torch
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData, create_train_loader
from evaluation import validation
from ideas.source_phase_compactness import (
    SourcePhaseWeightTracker,
    compute_source_phase_compactness_loss,
)
from ideas.source_feature_reshaper import (
    build_source_feature_reshaper,
    compute_dual_path_relation_regularization,
    compute_source_feature_reshaper_regularization,
)
from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    RandomTemporalShift,
    ToTensor,
    build_source_structure_transform,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, to_cuda


def train_supervised_source_phase_compactness(model, config, writer, splits, val_loader, device, best_model_path):
    """
    Source-only training with source-domain phase compactness regularization.

    Design choice:
    - compactness is computed on top of PSE / spatial encoder outputs
    - gradients flow normally through the upstream path of those features
    - LTAE / decoder are still trained by the supervised classification loss,
      but the compactness regularizer itself is only attached to PSE features
    """
    model.to(device)

    train_transform = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
        build_source_structure_transform(
            kind=getattr(config, "source_structure_transform", "none"),
            strength=getattr(config, "source_structure_strength", 0.0),
            phase_count=getattr(config, "source_structure_phase_count", 5),
        ),
        Normalize(),
        ToTensor(),
    ])
    dataset_name = config.source
    if config.train_on_target:
        dataset_name = config.target

    dataset = PixelSetData(
        config.data_root,
        dataset_name,
        config.classes,
        train_transform,
        splits[dataset_name]['train'],
        closed_set=config.closed_set,
    )
    data_loader = create_train_loader(dataset, config.batch_size, config.num_workers)
    print(f'training dataset: {dataset_name}, n={len(dataset)}, batches={len(data_loader)}')

    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    steps_per_epoch = len(data_loader)
    source_feature_reshaper = build_source_feature_reshaper(
        kind=getattr(config, "source_feature_reshaper", "none"),
        feature_dim=model.spatial_encoder.output_dim,
        strength=getattr(config, "source_feature_reshaper_strength", 0.10),
        kernel_size=getattr(config, "source_feature_reshaper_kernel_size", 3),
        phase_count=getattr(config, "source_structure_phase_count", 5),
        component_alpha_temperature=getattr(config, "source_component_alpha_temperature", 0.75),
        component_alpha_floor=getattr(config, "source_component_alpha_floor", 0.10),
        component_phase_scale=getattr(config, "source_component_phase_scale", 0.85),
    )
    params = list(model.parameters())
    if source_feature_reshaper is not None:
        source_feature_reshaper.to(device)
        params += list(source_feature_reshaper.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)
    phase_weight_tracker = SourcePhaseWeightTracker(phase_count=5)

    best_f1 = 0
    for epoch in range(config.epochs):
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        compact_loss_meter = AverageMeter()
        reshaper_loss_meter = AverageMeter()
        dual_cls_loss_meter = AverageMeter()
        dual_relation_loss_meter = AverageMeter()

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch + 1}/{config.epochs}')
        global_step = epoch * len(data_loader)
        for step, sample in progress_bar:
            targets = sample['label'].cuda(device=device, non_blocking=True)
            pixels, mask, positions, extra = to_cuda(sample, device)

            spatial_feats_raw = model.spatial_encoder(pixels, mask, extra)
            temporal_feats_raw = model.temporal_encoder(spatial_feats_raw, positions)
            outputs_raw = model.decoder(temporal_feats_raw)

            spatial_feats = spatial_feats_raw
            reshaper_loss = spatial_feats_raw.sum() * 0.0
            reshaper_logs = {}
            if source_feature_reshaper is not None:
                spatial_feats = source_feature_reshaper(spatial_feats_raw, positions=positions, labels=targets)
                reshaper_loss, reshaper_logs = compute_source_feature_reshaper_regularization(
                    spatial_feats_raw,
                    spatial_feats,
                )
                reshaper_logs.update(getattr(source_feature_reshaper, "last_logs", {}))

            temporal_feats = model.temporal_encoder(spatial_feats, positions)
            outputs = model.decoder(temporal_feats)

            cls_loss_raw = criterion(outputs_raw, targets)
            cls_loss = criterion(outputs, targets)
            compact_loss, compact_logs = compute_source_phase_compactness_loss(
                spatial_feats,
                positions,
                targets,
                weight_tracker=phase_weight_tracker,
                domain_adaptive_phase_weights=getattr(config, "source_domain_adaptive_phase_weights", False),
                phase_blend_alpha=getattr(config, "source_domain_phase_blend_alpha", 0.0),
            )
            dual_relation_loss = spatial_feats_raw.sum() * 0.0
            dual_relation_logs = {}
            if source_feature_reshaper is not None and getattr(config, "source_feature_dual_path", False):
                dual_relation_loss, dual_relation_logs = compute_dual_path_relation_regularization(
                    outputs_raw,
                    outputs,
                    raw_temporal_feats=temporal_feats_raw,
                    reshaped_temporal_feats=temporal_feats,
                )

            if source_feature_reshaper is not None and getattr(config, "source_feature_dual_path", False):
                loss = (
                    cls_loss_raw
                    + getattr(config, "source_feature_dual_cls_trade_off", 1.0) * cls_loss
                    + getattr(config, "source_feature_dual_relation_trade_off", 0.05) * dual_relation_loss
                )
            else:
                loss = cls_loss
            loss = loss + compact_loss + getattr(config, "source_feature_reshaper_reg_trade_off", 0.0) * reshaper_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)
            cls_loss_meter.update(cls_loss_raw.item(), n=config.batch_size)
            compact_loss_meter.update(compact_logs["compactness_loss"], n=config.batch_size)
            if source_feature_reshaper is not None:
                reshaper_loss_meter.update(reshaper_logs["source_reshaper_reg_loss"], n=config.batch_size)
                dual_cls_loss_meter.update(cls_loss.item(), n=config.batch_size)
                if getattr(config, "source_feature_dual_path", False):
                    dual_relation_loss_meter.update(dual_relation_logs["source_dual_relation_loss"], n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    lr=f'{lr:.1E}',
                    loss=f"{loss_meter.avg:.3f}",
                    cls=f"{cls_loss_meter.avg:.3f}",
                    compact=f"{compact_loss_meter.avg:.3f}",
                    reshaper=f"{reshaper_loss_meter.avg:.3f}",
                    dualcls=f"{dual_cls_loss_meter.avg:.3f}",
                    dualrel=f"{dual_relation_loss_meter.avg:.3f}",
                )
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)
                writer.add_scalar("train/source_cls_loss_raw", cls_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_phase_compactness_loss", compact_loss_meter.val, global_step + step)
                if source_feature_reshaper is not None:
                    writer.add_scalar("train/source_feature_reshaper_reg_loss", reshaper_loss_meter.val, global_step + step)
                    writer.add_scalar("train/source_cls_loss_reshaped", dual_cls_loss_meter.val, global_step + step)
                    if getattr(config, "source_feature_dual_path", False):
                        writer.add_scalar("train/source_dual_relation_loss", dual_relation_loss_meter.val, global_step + step)
                for key, value in compact_logs.items():
                    if key != "compactness_loss":
                        writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in reshaper_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)
                for key, value in dual_relation_logs.items():
                    writer.add_scalar(f"train/{key}", value, global_step + step)

        progress_bar.close()

        model.eval()
        best_f1 = validation(
            best_f1,
            best_model_path,
            config,
            criterion,
            device,
            epoch,
            model,
            val_loader,
            writer,
            source_feature_reshaper=source_feature_reshaper,
            apply_source_feature_reshaper=(config.target == config.source),
        )
