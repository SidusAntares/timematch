import torch


UNIFORM_PHASE_COUNT = 5
SOURCE_PHASE_COMPACTNESS_LAMBDA = 0.05
# Weights come from the absolute correlations observed in the closed-set
# uniform 5-phase analysis, then normalized to sum to 1.
SOURCE_PHASE_COMPACTNESS_WEIGHTS = (
    0.2471577675737799,
    0.2318104906937395,
    0.21267881498951407,
    0.15779025980183336,
    0.1465626669411332,
)


def _uniform_phase_slices(sequence_length, phase_count):
    phase_indices = torch.arange(sequence_length, dtype=torch.long)
    return torch.tensor_split(phase_indices, phase_count)


def _sorted_sequence_features(spatial_feats, positions):
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1])
    return torch.gather(spatial_feats, dim=1, index=expanded)


def compute_source_phase_compactness_loss(spatial_feats, positions, labels, eps=1e-6):
    """
    Compute a source-domain phase compactness regularizer on top of per-time-step
    PSE features. This is the first, minimal idea implementation:

    - uniform 5 phases
    - phase weights hard-coded from the current analysis
    - only compactness is optimized
    - gradients are allowed to flow normally through the whole upstream path
    """
    if spatial_feats.ndim != 3:
        raise ValueError(f"Expected spatial_feats to have shape [B, T, D], got {tuple(spatial_feats.shape)}")

    batch_size, sequence_length, _ = spatial_feats.shape
    if batch_size < 2 or sequence_length < UNIFORM_PHASE_COUNT:
        zero = spatial_feats.sum() * 0.0
        return zero, {"compactness_loss": 0.0}

    ordered_feats = _sorted_sequence_features(spatial_feats, positions)
    phase_slices = _uniform_phase_slices(sequence_length, UNIFORM_PHASE_COUNT)
    weights = spatial_feats.new_tensor(SOURCE_PHASE_COMPACTNESS_WEIGHTS)

    total_loss = spatial_feats.sum() * 0.0
    phase_logs = {}

    for phase_idx, phase_indices in enumerate(phase_slices):
        if phase_indices.numel() == 0:
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            continue

        phase_feats = ordered_feats[:, phase_indices, :].mean(dim=1)
        phase_loss = phase_feats.sum() * 0.0
        valid_class_count = 0

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

        if valid_class_count > 0:
            phase_loss = phase_loss / (valid_class_count + eps)
            total_loss = total_loss + weights[phase_idx] * phase_loss
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = float(phase_loss.detach().item())
        else:
            phase_logs[f"phase_compactness_loss_p{phase_idx + 1}"] = 0.0

    total_loss = SOURCE_PHASE_COMPACTNESS_LAMBDA * total_loss
    phase_logs["compactness_loss"] = float(total_loss.detach().item())
    return total_loss, phase_logs
