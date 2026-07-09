import os
import tempfile

import torch
import torch.nn as nn

from methods.local_shift.train_local_shift import (
    compute_local_shift_positions,
    load_source_stage_reference,
    validate_source_stage_reference,
)


class TinyTemporalModel(nn.Module):
    def __init__(self, dim=4, num_classes=3):
        super().__init__()
        self.temporal_encoder = nn.Linear(dim, dim)
        self.decoder = nn.Linear(dim, num_classes)

    def forward_from_temporal_features(self, temporal_features, positions, return_feats=False):
        pooled = self.temporal_encoder(temporal_features + positions.float().unsqueeze(-1) * 0.01).mean(dim=1)
        logits = self.decoder(pooled)
        if return_feats:
            return logits, pooled
        return logits


def _reference(num_classes=3, kmax=4, dim=4):
    return {
        "class_stage_feats": torch.randn(num_classes, kmax, dim),
        "class_stage_centers": torch.linspace(0, 9, kmax).view(1, kmax).expand(num_classes, -1).clone(),
        "class_stage_durations": torch.ones(num_classes, kmax),
        "class_stage_mask": torch.ones(num_classes, kmax, dtype=torch.bool),
        "class_counts": torch.ones(num_classes),
    }


def main():
    torch.manual_seed(11)
    reference = _reference()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "source_stage_reference.pt")
        torch.save(reference, path)
        loaded = load_source_stage_reference(path, device=torch.device("cpu"))

    validate_source_stage_reference(loaded, num_classes=3, feature_dim=4, kmax=4)

    batch, steps, dim = 5, 10, 4
    temporal_features = torch.randn(batch, steps, dim, requires_grad=True)
    positions = torch.arange(steps).view(1, steps).expand(batch, -1)
    pseudo_labels = torch.tensor([0, 1, 2, 1, 0])
    pseudo_mask = torch.tensor([True, True, True, False, True])

    positions_local, logs = compute_local_shift_positions(
        temporal_features=temporal_features,
        positions=positions,
        pseudo_labels=pseudo_labels,
        pseudo_mask=pseudo_mask,
        source_reference=loaded,
        global_shift=2,
        kmax=4,
        min_stage_len=2,
        top_m=2,
        change_quantile=0.75,
        nms_radius=1,
        local_shift_clip=5.0,
        mode="residual",
        detach_correspondence=True,
    )
    assert positions_local.shape == positions.shape
    assert "local_shift_abs_mean" in logs
    assert "alignment_valid_ratio" in logs

    model = TinyTemporalModel(dim=dim, num_classes=3)
    logits = model.forward_from_temporal_features(temporal_features, positions_local)
    loss = nn.CrossEntropyLoss()(logits[pseudo_mask], pseudo_labels[pseudo_mask])
    loss.backward()
    assert temporal_features.grad is not None
    assert float(temporal_features.grad.abs().sum()) > 0.0
    print("PASS|test_v321_train_local_shift_smoke")


if __name__ == "__main__":
    main()
