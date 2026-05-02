import torch
import torch.nn.functional as F


UNIFORM_PHASE_COUNT = 5


def _uniform_phase_slices(sequence_length, phase_count):
    phase_indices = torch.arange(sequence_length, dtype=torch.long)
    return torch.tensor_split(phase_indices, phase_count)


def _sorted_sequence_features(spatial_feats, positions):
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1])
    return torch.gather(spatial_feats, dim=1, index=expanded)


def _phase_pool(ordered_feats, phase_count):
    _, sequence_length, _ = ordered_feats.shape
    phase_slices = _uniform_phase_slices(sequence_length, phase_count)
    pooled = []
    for phase_indices in phase_slices:
        if phase_indices.numel() == 0:
            pooled.append(ordered_feats.new_zeros(ordered_feats.shape[0], ordered_feats.shape[-1]))
        else:
            pooled.append(ordered_feats[:, phase_indices, :].mean(dim=1))
    return pooled


def compute_target_phase_consistency_loss(
    teacher_spatial_feats_weak,
    positions_weak,
    student_spatial_feats_strong,
    positions_strong,
    eps=1e-6,
):
    """
    Unsupervised target-domain phase structure regularization.

    We avoid pseudo labels entirely and instead preserve the target batch's
    phase-wise geometry across weak/strong views:
    - teacher weak view provides the reference relation graph
    - student strong view is encouraged to match that graph
    - relation graph is the off-diagonal cosine similarity matrix per phase
    """
    if teacher_spatial_feats_weak.ndim != 3 or student_spatial_feats_strong.ndim != 3:
        raise ValueError("Expected phase consistency inputs to have shape [B, T, D].")

    batch_size = teacher_spatial_feats_weak.shape[0]
    zero = student_spatial_feats_strong.sum() * 0.0
    if batch_size < 2:
        return zero, {
            "target_consistency_loss": 0.0,
            "target_consistency_valid_phase_count": 0.0,
            "target_consistency_valid_samples": float(batch_size),
        }

    teacher_ordered = _sorted_sequence_features(teacher_spatial_feats_weak, positions_weak)
    student_ordered = _sorted_sequence_features(student_spatial_feats_strong, positions_strong)
    teacher_phase_feats = _phase_pool(teacher_ordered, UNIFORM_PHASE_COUNT)
    student_phase_feats = _phase_pool(student_ordered, UNIFORM_PHASE_COUNT)

    total_loss = zero
    valid_phase_count = 0
    off_diag_mask = (~torch.eye(batch_size, dtype=torch.bool, device=student_spatial_feats_strong.device)).float()
    logs = {"target_consistency_valid_samples": float(batch_size)}

    for phase_idx, (teacher_phase, student_phase) in enumerate(zip(teacher_phase_feats, student_phase_feats)):
        teacher_norm = F.normalize(teacher_phase, dim=1, eps=eps)
        student_norm = F.normalize(student_phase, dim=1, eps=eps)

        teacher_similarity = teacher_norm @ teacher_norm.T
        student_similarity = student_norm @ student_norm.T

        phase_diff = (student_similarity - teacher_similarity.detach()) * off_diag_mask
        denom = off_diag_mask.sum().clamp_min(1.0)
        phase_loss = phase_diff.pow(2).sum() / denom

        total_loss = total_loss + phase_loss
        valid_phase_count += 1
        logs[f"target_phase_consistency_loss_p{phase_idx + 1}"] = float(phase_loss.detach().item())
        logs[f"target_phase_teacher_sim_mean_p{phase_idx + 1}"] = float(
            (teacher_similarity.detach() * off_diag_mask).sum().item() / float(denom.item())
        )
        logs[f"target_phase_student_sim_mean_p{phase_idx + 1}"] = float(
            (student_similarity.detach() * off_diag_mask).sum().item() / float(denom.item())
        )

    total_loss = total_loss / (valid_phase_count + eps)
    logs["target_consistency_valid_phase_count"] = float(valid_phase_count)
    logs["target_consistency_loss"] = float(total_loss.detach().item())
    return total_loss, logs
