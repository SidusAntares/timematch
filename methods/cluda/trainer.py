import math
import os

import torch
import torch.nn.functional as F

from utils.train_utils import cycle

from .augmentations import CLUDAAugmenter, make_four_views
from .config import CLUDAConfig
from .losses import compose_losses
from .model import CLUDA


class _Compose:
    def __init__(self, operations):
        self.operations = tuple(operations)

    def __call__(self, sample):
        for operation in self.operations:
            sample = operation(sample)
        return sample


def grl_alpha(step, total_steps, schedule="official"):
    if schedule == "official":
        progress = float(step) / 1000.0
    elif schedule == "training_progress":
        progress = float(step) / max(int(total_steps) - 1, 1)
    elif schedule == "constant":
        return 1.0
    else:
        raise ValueError(f"unknown GRL schedule: {schedule}")
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def _batch_to_raw(sample, device):
    return (
        sample["pixels"].to(device, non_blocking=True),
        sample["valid_pixels"].to(device, non_blocking=True),
        sample["positions"].to(device, non_blocking=True),
    )


def cluda_training_step(model, optimizer, source_sample, target_sample, config,
                        global_step, total_steps, augmenter=None):
    """One faithful CLUDA update. Target labels are deliberately not accepted/read."""
    device = next(model.parameters()).device
    augmenter = augmenter or CLUDAAugmenter.from_config(config)
    source_pixels, source_valid, source_positions = _batch_to_raw(source_sample, device)
    target_pixels, target_valid, target_positions = _batch_to_raw(target_sample, device)
    source_q, source_k, target_q, target_k = make_four_views(
        source_pixels, source_valid, source_positions,
        target_pixels, target_valid, target_positions,
        augmenter,
    )
    alpha = grl_alpha(global_step, total_steps, config.grl_schedule)
    output = model(source_q, source_k, target_q, target_k, alpha)

    source_labels = source_sample["label"].to(device, non_blocking=True)
    losses = {
        "source_contrastive": F.cross_entropy(output.logits_source, output.labels_source),
        "target_contrastive": F.cross_entropy(output.logits_target, output.labels_target),
        "cross_domain_nn": F.cross_entropy(output.logits_nn, output.labels_nn),
        "domain": F.binary_cross_entropy(output.domain_prediction, output.domain_labels),
        "prediction": F.cross_entropy(output.source_prediction, source_labels),
    }
    total = compose_losses(losses, config.loss_weights)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return {
        "loss_source_contrastive": losses["source_contrastive"].detach().item(),
        "loss_target_contrastive": losses["target_contrastive"].detach().item(),
        "loss_cross_domain_nn": losses["cross_domain_nn"].detach().item(),
        "loss_domain": losses["domain"].detach().item(),
        "loss_prediction": losses["prediction"].detach().item(),
        "loss_total": total.detach().item(),
        "grl_alpha": alpha,
        "queue_pointer": int(model.queue_ptr.item()),
        "source_positive_similarity": output.source_positive_similarity.mean().detach().item(),
        "target_positive_similarity": output.target_positive_similarity.mean().detach().item(),
    }


def load_source_initialization(full_model, classifier, weights, device, expected_classes=None):
    path = os.path.join(weights, "fold_0", "model.pt") if os.path.isdir(weights) else weights
    run_dir = weights if os.path.isdir(weights) else os.path.dirname(os.path.dirname(path))
    if expected_classes is not None:
        import json
        with open(os.path.join(run_dir, "train_config.json"), encoding="utf-8") as file:
            source_config = json.load(file)
        if source_config.get("classes") != list(expected_classes):
            raise ValueError(
                "source checkpoint ordered class mapping does not match full CLUDA: "
                f"source={source_config.get('classes')}, expected={list(expected_classes)}"
            )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["state_dict"], strict=True)
    full_model.encoder_q.load_state_dict(classifier.encoder.state_dict(), strict=True)
    full_model.predictor.load_state_dict(classifier.predictor.state_dict(), strict=True)
    full_model.encoder_k.load_state_dict(full_model.encoder_q.state_dict(), strict=True)
    full_model._freeze_pse_batch_norm_stats()
    print(f"CLUDA_SOURCE_CHECKPOINT_OK|path={path}|strict=true", flush=True)


def _export_classifier(full_model, classifier):
    classifier.encoder.load_state_dict(full_model.encoder_q.state_dict())
    classifier.predictor.load_state_dict(full_model.predictor.state_dict())


def train_cluda_full(classifier, config, writer, _val_loader, device, best_model_path, _fold_num, splits):
    """Joint CLUDA training with chronological checkpoints and no target validation."""
    from dataset import PixelSetData, create_train_loader
    from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor

    method_config = CLUDAConfig.from_namespace(config)
    full_model = CLUDA(
        config.input_dim, config.num_classes, method_config.channels,
        method_config.hidden_dim, True, method_config.num_neighbors,
        method_config.kernel_size, method_config.stride, method_config.dilation_factor,
        method_config.dropout, method_config.queue_size, method_config.momentum,
        method_config.temperature, method_config.max_temporal_shift,
    ).to(device)
    if config.cluda_init == "source_weights":
        if not config.weights:
            raise ValueError("--weights is required for --cluda_init source_weights")
        load_source_initialization(full_model, classifier, config.weights, device, config.classes)

    transform = _Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        Normalize(),
        ToTensor(),
    ])
    source_dataset = PixelSetData(
        config.data_root, config.source, config.classes, transform,
        splits[config.source]["train"], closed_set=config.closed_set,
    )
    target_dataset = PixelSetData(
        config.data_root, config.target, config.classes, transform,
        splits[config.target]["train"], closed_set=False, ignore_labels=True,
    )
    source_loader = create_train_loader(source_dataset, config.batch_size, config.num_workers,
                                        timeout=getattr(config, "data_loader_timeout", 0))
    target_loader = create_train_loader(target_dataset, config.batch_size, config.num_workers,
                                        timeout=getattr(config, "data_loader_timeout", 0))
    optimizer = torch.optim.Adam(full_model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay, betas=(0.5, 0.99))
    augmenter = CLUDAAugmenter.from_config(method_config)
    total_steps = config.epochs * config.steps_per_epoch
    source_iterator, target_iterator = cycle(source_loader), cycle(target_loader)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    global_step = 0
    for epoch in range(config.epochs):
        full_model.train()
        for step in range(config.steps_per_epoch):
            metrics = cluda_training_step(
                full_model, optimizer, next(source_iterator), next(target_iterator),
                method_config, global_step, total_steps, augmenter,
            )
            for name, value in metrics.items():
                writer.add_scalar(f"cluda/{name}", value, global_step)
            if global_step == 0:
                print("CLUDA_FIRST_STEP|" + "|".join(f"{k}={v:.9g}" for k, v in metrics.items()), flush=True)
            if step % config.log_step == 0:
                print("CLUDA_STEP|" + "|".join(f"{k}={v:.6g}" for k, v in metrics.items()), flush=True)
            global_step += 1
        _export_classifier(full_model, classifier)
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": classifier.state_dict(),
            "cluda_state_dict": full_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "selection": "chronological_no_target_validation",
        }, best_model_path)

    _export_classifier(full_model, classifier)
