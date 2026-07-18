"""Source training with the optional raw-global compactness baseline."""

import time

import torch
from torchvision import transforms

from dataset import PixelSetData, create_train_loader
from evaluation import validation
from methods.source_structure.raw_global import compute_raw_global_compactness
from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    RandomTemporalShift,
    ToTensor,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, to_cuda


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _format_elapsed_seconds(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def train_source_structure(model, config, writer, splits, val_loader, device, best_model_path):
    """Train source classification with ``off`` or ``raw_global`` structure."""

    mode = str(getattr(config, "source_structure_mode", "off"))
    weight = float(getattr(config, "source_structure_weight", 0.0))
    if mode not in {"off", "raw_global"}:
        raise ValueError(f"unsupported source_structure_mode: {mode}")
    if weight < 0.0:
        raise ValueError("source_structure_weight must be non-negative")
    if getattr(config, "with_shift_aug", False):
        raise ValueError("source structure training requires with_shift_aug=False")

    model.to(device)
    dataset_name = config.target if config.train_on_target else config.source
    train_transform = transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p)
            if config.with_shift_aug
            else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )
    dataset = PixelSetData(
        config.data_root,
        dataset_name,
        config.classes,
        train_transform,
        splits[dataset_name]["train"],
        closed_set=config.closed_set,
    )
    data_loader = create_train_loader(
        dataset,
        config.batch_size,
        config.num_workers,
        timeout=getattr(config, "data_loader_timeout", 0),
    )
    print(
        "SOURCE_TRAIN_START|"
        f"method=source_structure|mode={mode}|weight={weight:.8f}|"
        f"dataset={dataset_name}|samples={len(dataset)}|batches={len(data_loader)}|"
        f"batch_size={config.batch_size}|num_workers={config.num_workers}|"
        f"epochs={config.epochs}",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    steps_per_epoch = len(data_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs * steps_per_epoch,
        eta_min=0,
    )

    best_f1 = 0
    train_start_time = time.time()
    for epoch in range(config.epochs):
        model.train()
        total_meter = AverageMeter()
        classification_meter = AverageMeter()
        raw_global_meter = AverageMeter()
        contribution_meter = AverageMeter()
        valid_classes_meter = AverageMeter()
        epoch_start_time = time.time()
        print(
            f"---------epoch {epoch + 1}/{config.epochs} | "
            f"timestamp={_timestamp()} | "
            f"elapsed={_format_elapsed_seconds(epoch_start_time - train_start_time)} ---------",
            flush=True,
        )
        speed_probe_steps = {
            step for step in (10, 50, 100, 200, 500) if step <= len(data_loader)
        }
        speed_probe_steps.add(len(data_loader))

        global_step = epoch * len(data_loader)
        for step, sample in enumerate(data_loader):
            labels = sample["label"].cuda(device=device, non_blocking=True)
            pixels, mask, positions, extra = to_cuda(sample, device)

            temporal_features = model.spatial_encoder(pixels, mask, extra)
            # Future decomposition insertion point:
            # H_components = decompose(temporal_features, positions, temporal_mask=None)
            encoded_features = model.temporal_encoder(temporal_features, positions)
            logits = model.decoder(encoded_features)
            classification_loss = criterion(logits, labels)

            if mode == "raw_global":
                raw_result = compute_raw_global_compactness(temporal_features, labels)
                raw_global_loss = raw_result.loss
                valid_class_count = raw_result.valid_class_count
            else:
                raw_global_loss = temporal_features.sum() * 0.0
                valid_class_count = 0
            raw_global_contribution = weight * raw_global_loss
            total_loss = classification_loss + raw_global_contribution

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = int(labels.shape[0])
            total_meter.update(total_loss.item(), n=batch_size)
            classification_meter.update(classification_loss.item(), n=batch_size)
            raw_global_meter.update(raw_global_loss.item(), n=batch_size)
            contribution_meter.update(raw_global_contribution.item(), n=batch_size)
            valid_classes_meter.update(valid_class_count, n=1)

            if step % config.log_step == 0:
                writer.add_scalar("train/source_total_loss", total_meter.val, global_step + step)
                writer.add_scalar(
                    "train/source_cls_loss", classification_meter.val, global_step + step
                )
                writer.add_scalar(
                    "train/source_raw_global_loss", raw_global_meter.val, global_step + step
                )
                writer.add_scalar(
                    "train/source_raw_global_contribution",
                    contribution_meter.val,
                    global_step + step,
                )
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step + step)

            if epoch == 0 and (step + 1) in speed_probe_steps:
                elapsed = time.time() - epoch_start_time
                print(
                    "SOURCE_SPEED_PROBE|"
                    f"timestamp={_timestamp()}|mode={mode}|epoch={epoch + 1}|"
                    f"batches={step + 1}|elapsed={_format_elapsed_seconds(elapsed)}|"
                    f"batches_per_sec={(step + 1) / max(elapsed, 1e-9):.3f}|"
                    f"source_total_loss={total_meter.avg:.6f}",
                    flush=True,
                )

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        print(
            "SOURCE_EPOCH_SUMMARY|"
            f"timestamp={_timestamp()}|epoch={epoch + 1}|elapsed={_format_elapsed_seconds(total_elapsed)}|"
            f"epoch_elapsed={_format_elapsed_seconds(epoch_elapsed)}|"
            f"source_cls_loss={classification_meter.avg:.6f}|"
            f"source_raw_global_loss={raw_global_meter.avg:.6f}|"
            f"source_raw_global_weight={weight:.8f}|"
            f"source_raw_global_contribution={contribution_meter.avg:.6f}|"
            f"source_total_loss={total_meter.avg:.6f}|"
            f"source_raw_global_valid_classes={valid_classes_meter.avg:.3f}",
            flush=True,
        )

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
        )
