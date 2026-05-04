import torch
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData, create_train_loader
from evaluation import validation
from ideas.source_phase_compactness import (
    SourcePhaseWeightTracker,
    compute_source_phase_compactness_loss,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)
    phase_weight_tracker = SourcePhaseWeightTracker(phase_count=5)

    best_f1 = 0
    for epoch in range(config.epochs):
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        compact_loss_meter = AverageMeter()

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch + 1}/{config.epochs}')
        global_step = epoch * len(data_loader)
        for step, sample in progress_bar:
            targets = sample['label'].cuda(device=device, non_blocking=True)
            pixels, mask, positions, extra = to_cuda(sample, device)

            spatial_feats = model.spatial_encoder(pixels, mask, extra)
            temporal_feats = model.temporal_encoder(spatial_feats, positions)
            outputs = model.decoder(temporal_feats)

            cls_loss = criterion(outputs, targets)
            compact_loss, compact_logs = compute_source_phase_compactness_loss(
                spatial_feats,
                positions,
                targets,
                weight_tracker=phase_weight_tracker,
            )
            loss = cls_loss + compact_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)
            cls_loss_meter.update(cls_loss.item(), n=config.batch_size)
            compact_loss_meter.update(compact_logs["compactness_loss"], n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    lr=f'{lr:.1E}',
                    loss=f"{loss_meter.avg:.3f}",
                    cls=f"{cls_loss_meter.avg:.3f}",
                    compact=f"{compact_loss_meter.avg:.3f}",
                )
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)
                writer.add_scalar("train/source_cls_loss", cls_loss_meter.val, global_step + step)
                writer.add_scalar("train/source_phase_compactness_loss", compact_loss_meter.val, global_step + step)
                for key, value in compact_logs.items():
                    if key != "compactness_loss":
                        writer.add_scalar(f"train/{key}", value, global_step + step)

        progress_bar.close()

        model.eval()
        best_f1 = validation(best_f1, best_model_path, config, criterion, device, epoch, model, val_loader, writer)
