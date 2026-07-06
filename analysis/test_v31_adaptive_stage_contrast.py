import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from ideas.adaptive_stage_contrast import (
    AdaptiveTemporalStageExtractor,
    ShiftAwareStageCorrespondence,
    StageContrastiveLoss,
    compute_adaptive_stage_contrast_loss,
)


def main():
    torch.manual_seed(31)
    batch_size = 8
    time_steps = 24
    feat_dim = 64
    stage_count = 6

    h_s = torch.randn(batch_size, time_steps, feat_dim, requires_grad=True)
    h_t = torch.randn(batch_size, time_steps, feat_dim, requires_grad=True)
    positions = torch.arange(time_steps).view(1, -1).repeat(batch_size, 1).float()
    source_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    target_pseudo = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    target_conf = torch.tensor([0.95, 0.91, 0.88, 0.97, 0.93, 0.99, 0.92, 0.94])
    target_mask = target_conf >= 0.9

    extractor = AdaptiveTemporalStageExtractor(num_stages=stage_count, min_stage_len=2)
    source_stage = extractor(h_s, positions)
    target_stage = extractor(h_t, positions)

    assert source_stage["stage_feats"].shape == (batch_size, stage_count, feat_dim)
    assert source_stage["stage_start_idx"].shape == (batch_size, stage_count)
    assert source_stage["stage_end_idx"].shape == (batch_size, stage_count)
    assert source_stage["stage_center_pos"].shape == (batch_size, stage_count)

    correspondence = ShiftAwareStageCorrespondence(stage_time_radius=8.0, stage_time_temperature=10.0)
    weights, corr_logs = correspondence(
        source_stage["stage_center_pos"],
        target_stage["stage_center_pos"],
        source_stage["stage_mask"],
        target_stage["stage_mask"],
        target_to_source_shift=2,
    )
    assert weights.shape == (batch_size, batch_size, stage_count, stage_count)
    assert corr_logs["correspondence_valid_candidate_mean"] > 0.0

    criterion = StageContrastiveLoss(temperature=0.1)
    loss, logs = criterion(
        source_stage["stage_feats"],
        source_stage["stage_mask"],
        source_labels,
        target_stage["stage_feats"],
        target_stage["stage_mask"],
        target_pseudo,
        target_conf,
        target_mask,
        weights,
    )
    assert loss.dim() == 0
    assert logs["stage_valid_queries"] > 0.0
    loss.backward()
    assert h_s.grad is not None
    assert h_t.grad is not None

    zero_loss, zero_logs = compute_adaptive_stage_contrast_loss(
        h_s.detach().clone().requires_grad_(True),
        positions,
        source_labels,
        h_t.detach().clone().requires_grad_(True),
        positions,
        target_pseudo,
        target_conf,
        torch.zeros_like(target_mask),
        target_to_source_shift=0,
        num_stages=stage_count,
        stage_min_len=2,
        stage_time_radius=8.0,
        stage_time_temperature=10.0,
        temperature=0.1,
    )
    assert zero_loss.dim() == 0
    assert zero_logs["stage_valid_queries"] == 0.0

    print("V31_UNIT_TEST_OK")
    print(f"loss={float(loss.detach().item()):.6f}")
    print(f"valid_queries={logs['stage_valid_queries']:.0f}")
    print(f"zero_mask_loss={float(zero_loss.detach().item()):.6f}")


if __name__ == "__main__":
    main()
