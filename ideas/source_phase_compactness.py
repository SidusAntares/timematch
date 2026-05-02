import torch


UNIFORM_PHASE_COUNT = 5
SOURCE_PHASE_COMPACTNESS_LAMBDA = 0.05
# Source-level dynamic phase-weight construction:
# - keep the phase partition fixed (uniform 5 phases)
# - accumulate source-domain phase structure statistics across batches
# - derive the applied weights from a source-level EMA summary instead of
#   re-deciding them from each batch independently
PHASE_WEIGHT_COMPACTNESS_COEFF = 1.0
PHASE_WEIGHT_MARGIN_COEFF = 1.0
PHASE_WEIGHT_TEMPERATURE = 1.0
PHASE_WEIGHT_EMA_MOMENTUM = 0.9


def _uniform_phase_slices(sequence_length, phase_count):
    phase_indices = torch.arange(sequence_length, dtype=torch.long)
    return torch.tensor_split(phase_indices, phase_count)


def _sorted_sequence_features(spatial_feats, positions):
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1])
    return torch.gather(spatial_feats, dim=1, index=expanded)


def _zscore_or_zero(values, eps):
    if values.numel() <= 1:
        return values.new_zeros(values.shape)
    centered = values - values.mean()
    std = centered.std(unbiased=False)
    if float(std.item()) < eps:
        return values.new_zeros(values.shape)
    return centered / (std + eps)


def _compute_phase_weights(phase_structures, reference_tensor, eps):
    compactness_scores = []
    margin_scores = []
    valid_mask = []

    for stats in phase_structures:
        valid = stats["valid_class_count"] > 0
        valid_mask.append(valid)
        if valid:
            compactness_scores.append(stats["compactness_score"])
            margin_scores.append(stats["margin_score"])
        else:
            compactness_scores.append(reference_tensor.new_tensor(0.0))
            margin_scores.append(reference_tensor.new_tensor(0.0))

    compactness_scores = torch.stack(compactness_scores)
    margin_scores = torch.stack(margin_scores)
    valid_mask_tensor = torch.tensor(valid_mask, device=reference_tensor.device, dtype=torch.bool)

    if not bool(valid_mask_tensor.any()):
        return reference_tensor.new_full((len(phase_structures),), 1.0 / max(len(phase_structures), 1))

    compactness_z = _zscore_or_zero(compactness_scores[valid_mask_tensor], eps)
    margin_z = _zscore_or_zero(margin_scores[valid_mask_tensor], eps)
    combined = (
        PHASE_WEIGHT_COMPACTNESS_COEFF * compactness_z
        + PHASE_WEIGHT_MARGIN_COEFF * margin_z
    )
    valid_weights = torch.softmax(combined / PHASE_WEIGHT_TEMPERATURE, dim=0)

    weights = reference_tensor.new_zeros(len(phase_structures))
    weights[valid_mask_tensor] = valid_weights
    return weights


class SourcePhaseWeightTracker:
    def __init__(self, phase_count, ema_momentum=PHASE_WEIGHT_EMA_MOMENTUM):
        self.phase_count = phase_count
        self.ema_momentum = ema_momentum
        self.running_compactness = None
        self.running_margin = None
        self.valid_counts = None

    def update(self, phase_structures):
        compactness_scores = []
        margin_scores = []
        valid_mask = []

        for stats in phase_structures:
            valid = stats["valid_class_count"] > 0
            valid_mask.append(valid)
            compactness_scores.append(stats["compactness_score"].detach().cpu())
            margin_scores.append(stats["margin_score"].detach().cpu())

        compactness_scores = torch.stack(compactness_scores)
        margin_scores = torch.stack(margin_scores)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        if self.running_compactness is None:
            self.running_compactness = compactness_scores.clone()
            self.running_margin = margin_scores.clone()
            self.valid_counts = valid_mask_tensor.to(torch.long)
            return

        for phase_idx in range(self.phase_count):
            if not bool(valid_mask_tensor[phase_idx].item()):
                continue

            if int(self.valid_counts[phase_idx].item()) == 0:
                self.running_compactness[phase_idx] = compactness_scores[phase_idx]
                self.running_margin[phase_idx] = margin_scores[phase_idx]
            else:
                momentum = self.ema_momentum
                self.running_compactness[phase_idx] = (
                    momentum * self.running_compactness[phase_idx]
                    + (1.0 - momentum) * compactness_scores[phase_idx]
                )
                self.running_margin[phase_idx] = (
                    momentum * self.running_margin[phase_idx]
                    + (1.0 - momentum) * margin_scores[phase_idx]
                )

            self.valid_counts[phase_idx] += 1

    def get_weights(self, reference_tensor, eps):
        if self.running_compactness is None or self.valid_counts is None:
            return reference_tensor.new_full((self.phase_count,), 1.0 / max(self.phase_count, 1))

        valid_mask = self.valid_counts > 0
        if not bool(valid_mask.any().item()):
            return reference_tensor.new_full((self.phase_count,), 1.0 / max(self.phase_count, 1))

        compactness_scores = self.running_compactness.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        margin_scores = self.running_margin.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        valid_mask_device = valid_mask.to(device=reference_tensor.device)

        compactness_z = _zscore_or_zero(compactness_scores[valid_mask_device], eps)
        margin_z = _zscore_or_zero(margin_scores[valid_mask_device], eps)
        combined = (
            PHASE_WEIGHT_COMPACTNESS_COEFF * compactness_z
            + PHASE_WEIGHT_MARGIN_COEFF * margin_z
        )
        valid_weights = torch.softmax(combined / PHASE_WEIGHT_TEMPERATURE, dim=0)

        weights = reference_tensor.new_zeros(self.phase_count)
        weights[valid_mask_device] = valid_weights
        return weights

    def get_logs(self):
        logs = {}
        if self.running_compactness is None or self.running_margin is None or self.valid_counts is None:
            return logs

        for phase_idx in range(self.phase_count):
            logs[f"source_compactness_score_p{phase_idx + 1}"] = float(self.running_compactness[phase_idx].item())
            logs[f"source_margin_score_p{phase_idx + 1}"] = float(self.running_margin[phase_idx].item())
            logs[f"source_valid_count_p{phase_idx + 1}"] = float(self.valid_counts[phase_idx].item())
        return logs


def compute_source_phase_compactness_loss(spatial_feats, positions, labels, weight_tracker=None, eps=1e-6):
    """
    Compute a source-domain phase compactness regularizer on top of per-time-step
    PSE features.

    - uniform 5 phases
    - phase weights come from a source-level running summary
    - compactness remains the optimized quantity
    - source-level weights are built from detached reliability estimates:
      - inverse within-class scatter
      - nearest-class phase margin
    """
    if spatial_feats.ndim != 3:
        raise ValueError(f"Expected spatial_feats to have shape [B, T, D], got {tuple(spatial_feats.shape)}")

    batch_size, sequence_length, _ = spatial_feats.shape
    if batch_size < 2 or sequence_length < UNIFORM_PHASE_COUNT:
        zero = spatial_feats.sum() * 0.0
        return zero, {"compactness_loss": 0.0}

    ordered_feats = _sorted_sequence_features(spatial_feats, positions)
    phase_slices = _uniform_phase_slices(sequence_length, UNIFORM_PHASE_COUNT)
    total_loss = spatial_feats.sum() * 0.0
    phase_logs = {}
    phase_structures = []

    for phase_idx, phase_indices in enumerate(phase_slices):
        if phase_indices.numel() == 0:
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = 0.0
            phase_structures.append({
                "phase_loss": total_loss,
                "valid_class_count": 0,
                "compactness_score": spatial_feats.new_tensor(0.0),
                "margin_score": spatial_feats.new_tensor(0.0),
            })
            continue

        phase_feats = ordered_feats[:, phase_indices, :].mean(dim=1)
        phase_loss = phase_feats.sum() * 0.0
        valid_class_count = 0
        class_centers = []
        class_withins = []

        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            class_count = int(class_mask.sum().item())
            if class_count < 2:
                continue

            class_phase_feats = phase_feats[class_mask]
            class_center = class_phase_feats.mean(dim=0, keepdim=True)
            class_within = (class_phase_feats - class_center).pow(2).sum(dim=1).mean()
            phase_loss = phase_loss + class_within
            valid_class_count += 1
            class_centers.append(class_center.squeeze(0))
            class_withins.append(class_within)

        if valid_class_count > 0:
            phase_loss = phase_loss / (valid_class_count + eps)
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = float(phase_loss.detach().item())
            compactness_score = 1.0 / (phase_loss.detach() + eps)

            if len(class_centers) >= 2:
                centers = torch.stack(class_centers, dim=0)
                center_distances = torch.cdist(centers, centers, p=2)
                center_distances.fill_diagonal_(float("inf"))
                margin_score = center_distances.min(dim=1).values.mean().detach()
            else:
                margin_score = spatial_feats.new_tensor(0.0)

            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = float(margin_score.item())
            phase_structures.append({
                "phase_loss": phase_loss,
                "valid_class_count": valid_class_count,
                "compactness_score": compactness_score,
                "margin_score": margin_score,
            })
        else:
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            phase_logs[f"phase_margin_score_p{phase_idx + 1}"] = 0.0
            phase_structures.append({
                "phase_loss": phase_loss,
                "valid_class_count": 0,
                "compactness_score": spatial_feats.new_tensor(0.0),
                "margin_score": spatial_feats.new_tensor(0.0),
            })

    if weight_tracker is None:
        weights = _compute_phase_weights(phase_structures, spatial_feats, eps)
    else:
        weight_tracker.update(phase_structures)
        weights = weight_tracker.get_weights(spatial_feats, eps)

    for phase_idx, stats in enumerate(phase_structures):
        if stats["valid_class_count"] > 0:
            total_loss = total_loss + weights[phase_idx] * stats["phase_loss"]
        phase_logs[f"phase_weight_p{phase_idx + 1}"] = float(weights[phase_idx].detach().item())

    if weight_tracker is not None:
        phase_logs.update(weight_tracker.get_logs())

    total_loss = SOURCE_PHASE_COMPACTNESS_LAMBDA * total_loss
    phase_logs["compactness_loss"] = float(total_loss.detach().item())
    return total_loss, phase_logs
