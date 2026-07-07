import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from ideas.adaptive_stage_contrast import (
    FastClassPrototypeStageContrastiveLoss,
    FastShiftAwareClassStageCorrespondence,
    build_class_stage_prototypes,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    torch.manual_seed(310)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, bt, k, d, c = 128, 128, 6, 128, 12

    source_stage_feats = torch.randn(bs, k, d, device=device, requires_grad=True)
    target_stage_feats = torch.randn(bt, k, d, device=device, requires_grad=True)
    source_stage_mask = torch.ones(bs, k, device=device, dtype=torch.bool)
    target_stage_mask = torch.ones(bt, k, device=device, dtype=torch.bool)
    base_centers = torch.linspace(0.0, 29.0, steps=k, device=device)
    source_stage_center_pos = base_centers.view(1, k).repeat(bs, 1)
    target_stage_center_pos = (base_centers + 2.0).view(1, k).repeat(bt, 1)
    source_labels = torch.arange(bs, device=device) % c
    target_pseudo_labels = torch.arange(bt, device=device) % c
    target_conf = torch.rand(bt, device=device) * 0.1 + 0.9
    target_mask = target_conf >= 0.9

    prototypes = build_class_stage_prototypes(
        source_stage_feats,
        source_stage_mask,
        source_stage_center_pos,
        source_labels,
        num_classes=c,
    )
    assert prototypes["source_proto"].shape == (c, k, d)
    assert prototypes["source_center"].shape == (c, k)
    assert prototypes["source_proto_mask"].shape == (c, k)
    assert bool(prototypes["source_proto_mask"].all().item())

    correspondence = FastShiftAwareClassStageCorrespondence(
        stage_time_radius=8.0,
        stage_time_temperature=10.0,
    )
    a, corr_logs = correspondence(
        target_stage_center_pos,
        target_stage_mask,
        prototypes["source_center"],
        prototypes["source_proto_mask"],
        target_to_source_shift=-2,
    )
    assert a.shape == (bt, k, c, k)
    assert not torch.isnan(a).any()
    valid_sum = a.sum(dim=-1)[target_stage_mask[:, :, None].expand(bt, k, c)]
    assert torch.allclose(valid_sum, torch.ones_like(valid_sum), atol=1e-5)
    assert corr_logs["correspondence_valid_class_mean"] > 0.0

    criterion = FastClassPrototypeStageContrastiveLoss(temperature=0.1)
    loss, logs = criterion(
        source_stage_feats,
        source_stage_mask,
        source_stage_center_pos,
        source_labels,
        target_stage_feats,
        target_stage_mask,
        target_stage_center_pos,
        target_pseudo_labels,
        target_conf,
        target_mask,
        target_to_source_shift=-2,
        num_classes=c,
        stage_time_radius=8.0,
        stage_time_temperature=10.0,
    )
    assert loss.dim() == 0
    assert logs["stage_valid_queries"] > 0
    loss.backward()
    assert source_stage_feats.grad is not None
    assert target_stage_feats.grad is not None

    zero_loss, zero_logs = criterion(
        source_stage_feats.detach().clone().requires_grad_(True),
        source_stage_mask,
        source_stage_center_pos,
        source_labels,
        target_stage_feats.detach().clone().requires_grad_(True),
        target_stage_mask,
        target_stage_center_pos,
        target_pseudo_labels,
        target_conf,
        torch.zeros_like(target_mask),
        target_to_source_shift=-2,
        num_classes=c,
        stage_time_radius=8.0,
        stage_time_temperature=10.0,
    )
    assert zero_loss.dim() == 0
    assert zero_logs["stage_valid_queries"] == 0.0

    # Microbenchmark: fast loss forward only, shape matches intended training batch.
    repeats = 20
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            criterion(
                source_stage_feats.detach(),
                source_stage_mask,
                source_stage_center_pos,
                source_labels,
                target_stage_feats.detach(),
                target_stage_mask,
                target_stage_center_pos,
                target_pseudo_labels,
                target_conf,
                target_mask,
                target_to_source_shift=-2,
                num_classes=c,
                stage_time_radius=8.0,
                stage_time_temperature=10.0,
            )
    _sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats

    print("V31_FAST_UNIT_TEST_OK")
    print(f"device={device}")
    print(f"loss={float(loss.detach().item()):.6f}")
    print(f"valid_queries={logs['stage_valid_queries']:.0f}")
    print(f"zero_mask_loss={float(zero_loss.detach().item()):.6f}")
    print(f"fast_forward_mean_ms={elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
