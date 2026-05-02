import torch


UNIFORM_PHASE_COUNT = 5


def _uniform_phase_slices(sequence_length, phase_count):
    phase_indices = torch.arange(sequence_length, dtype=torch.long)
    return torch.tensor_split(phase_indices, phase_count)


def _sorted_sequence_features(spatial_feats, positions):
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1])
    return torch.gather(spatial_feats, dim=1, index=expanded)


def compute_target_phase_compactness_loss(spatial_feats, positions, pseudo_labels, eps=1e-6):
    """
    Compute a class-conditional phase compactness loss on high-confidence target
    samples using their pseudo labels.

    This is intentionally minimal for the first DA-stage target-structure probe:
    - uniform 5-phase partition
    - equal weighting over valid phases
    - optimize only compactness, no extra phase weighting yet
    """
    if spatial_feats.ndim != 3:
        raise ValueError(f"Expected spatial_feats to have shape [B, T, D], got {tuple(spatial_feats.shape)}")

    batch_size, sequence_length, _ = spatial_feats.shape
    zero = spatial_feats.sum() * 0.0
    if batch_size < 2 or sequence_length < UNIFORM_PHASE_COUNT:
        return zero, {"target_compactness_loss": 0.0, "target_valid_phase_count": 0.0}

    ordered_feats = _sorted_sequence_features(spatial_feats, positions)
    phase_slices = _uniform_phase_slices(sequence_length, UNIFORM_PHASE_COUNT)

    total_loss = zero
    valid_phase_count = 0
    logs = {"target_struct_valid_samples": float(batch_size)}

    for phase_idx, phase_indices in enumerate(phase_slices):
        if phase_indices.numel() == 0:
            logs[f"target_phase_compactness_loss_p{phase_idx + 1}"] = 0.0
            logs[f"target_phase_valid_classes_p{phase_idx + 1}"] = 0.0
            continue

        phase_feats = ordered_feats[:, phase_indices, :].mean(dim=1)
        phase_loss = phase_feats.sum() * 0.0
        valid_class_count = 0

        for class_id in pseudo_labels.unique(sorted=True):
            class_mask = pseudo_labels == class_id
            class_count = int(class_mask.sum().item())
            if class_count < 2:
                continue

            class_phase_feats = phase_feats[class_mask]
            class_center = class_phase_feats.mean(dim=0, keepdim=True)
            class_within = (class_phase_feats - class_center).pow(2).sum(dim=1).mean()
            phase_loss = phase_loss + class_within
            valid_class_count += 1

        logs[f"target_phase_valid_classes_p{phase_idx + 1}"] = float(valid_class_count)
        if valid_class_count > 0:
            phase_loss = phase_loss / (valid_class_count + eps)
            total_loss = total_loss + phase_loss
            valid_phase_count += 1
            logs[f"target_phase_compactness_loss_p{phase_idx + 1}"] = float(phase_loss.detach().item())
        else:
            logs[f"target_phase_compactness_loss_p{phase_idx + 1}"] = 0.0

    if valid_phase_count > 0:
        total_loss = total_loss / (valid_phase_count + eps)
    else:
        total_loss = zero

    logs["target_valid_phase_count"] = float(valid_phase_count)
    logs["target_compactness_loss"] = float(total_loss.detach().item())
    return total_loss, logs
