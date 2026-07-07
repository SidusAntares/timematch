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
    target_features[0].zero_()
    times = torch.linspace(0, 110, steps)

    partitioner = BudgetedFeatureChangePartitioner(kmax=5, min_stage_len=2, change_quantile=0.65)
    reference = build_reference_from_temporal_features(
        source_features,
        source_labels,
        times,
        num_classes=classes,
        partitioner=partitioner,
    )
    target_partition = partition_target_temporal_features(
        target_features,
        times.unsqueeze(0).expand(batch, -1),
        partitioner=partitioner,
    )
    target_pooled = target_partition["stage_feats"]
    target_stage_times = target_partition["stage_centers"]
    target_mask = target_partition["stage_mask"]
    stage_to_time = target_partition["stage_to_time"]

    pseudo = torch.tensor([0, 1, 2, 0, 1])
    pseudo_mask = torch.tensor([True, True, True, False, True])
    aligner = SoftStageAligner(temperature=0.2, top_m=2)
    alignment = aligner(
        target_stage_feats=target_pooled,
        target_stage_centers=target_stage_times,
        target_stage_durations=target_partition["stage_durations"],
        target_stage_mask=target_mask,
        pseudo_labels=pseudo,
        pseudo_mask=pseudo_mask,
        source_reference=reference,
        global_shift=torch.zeros(batch),
    )
    shifts = compute_stage_residual_shift(
        target_stage_centers=target_stage_times,
        expected_source_centers=alignment["expected_source_centers"],
        global_shift=torch.zeros(batch),
        local_shift_clip=60,
    )
    positions, position_logs = expand_stage_shift_to_time(
        base_positions=times.unsqueeze(0).expand(batch, -1),
        stage_to_time=stage_to_time,
        stage_shift=shifts,
        stage_mask=target_mask,
        global_shift=torch.zeros(batch),
    )
    positions, clamp_logs = check_temporal_positions_in_range(
        positions,
        min_position=0,
        max_position=365,
        round_to_long=False,
    )

    assert reference["class_stage_feats"].ndim == 3
    assert reference["class_stage_mask"].shape[:2] == reference["class_stage_feats"].shape[:2]
    assert target_pooled.shape == (batch, partitioner.kmax, dim)
    assert target_mask.shape == (batch, partitioner.kmax)
    assert target_partition["stage_count"].unique().numel() > 1
    assert alignment["weights"].shape == (batch, partitioner.kmax, 2)
    assert alignment["source_indices"].shape == (batch, partitioner.kmax, 2)
    assert shifts.shape == target_stage_times.shape
    assert positions.shape == (batch, steps)
    assert torch.isfinite(alignment["weights"]).all()
    assert torch.isfinite(positions).all()
    assert position_logs["residual_clip_fraction"] >= 0.0
    assert clamp_logs["position_clamp_ratio"] == 0.0
    print("v321 local-shift shape checks passed")


if __name__ == "__main__":
    main()
