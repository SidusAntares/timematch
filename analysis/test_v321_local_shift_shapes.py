"""Shape checks for the v3.2.1 local-shift scaffold."""

import torch

from methods.local_shift.local_position import (
    check_temporal_positions_in_range,
    compute_stage_residual_shift,
    expand_stage_shift_to_time,
)
from methods.local_shift.soft_alignment import SoftStageAligner
from methods.local_shift.source_reference import build_reference_from_temporal_features
from methods.local_shift.target_partition import (
    BudgetedFeatureChangePartitioner,
    partition_target_temporal_features,
)


def main() -> None:
    torch.manual_seed(7)
    batch, steps, dim, classes = 5, 12, 8, 3
    source_features = torch.randn(batch * 3, steps, dim)
    source_labels = torch.arange(batch * 3) % classes
    target_features = torch.randn(batch, steps, dim)
    times = torch.linspace(0, 110, steps)

    partitioner = BudgetedFeatureChangePartitioner(kmax=5, min_stage_len=2, change_quantile=0.65)
    reference = build_reference_from_temporal_features(
        source_features,
        source_labels,
        times,
        num_classes=classes,
        partitioner=partitioner,
    )
    intervals, target_pooled, target_stage_times = partition_target_temporal_features(
        target_features,
        times,
        partitioner=partitioner,
    )

    target_stage_features = target_pooled.unsqueeze(0).expand(batch, -1, -1).contiguous()
    pseudo = torch.tensor([0, 1, 2, 0, 1])
    aligner = SoftStageAligner(temperature=0.2, top_m=2)
    alignment = aligner(
        target_stage_features,
        reference.prototypes,
        reference.times,
        pseudo_labels=pseudo,
        source_mask=reference.mask,
    )
    target_times = target_stage_times.unsqueeze(0).expand(batch, -1)
    shifts = compute_stage_residual_shift(target_times, alignment.expected_source_times, clip=60)
    positions = expand_stage_shift_to_time(times.unsqueeze(0).expand(batch, -1), intervals, shifts)
    positions = check_temporal_positions_in_range(positions, min_position=0, max_position=365)

    assert reference.prototypes.ndim == 3
    assert reference.mask.shape[:2] == reference.prototypes.shape[:2]
    assert target_stage_features.shape[1] == len(intervals)
    assert alignment.weights.shape[:2] == target_stage_features.shape[:2]
    assert shifts.shape == target_times.shape
    assert positions.shape == (batch, steps)
    assert torch.isfinite(alignment.weights).all()
    assert torch.isfinite(positions).all()
    print("v321 local-shift shape checks passed")


if __name__ == "__main__":
    main()
