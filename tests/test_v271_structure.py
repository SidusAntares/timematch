import json

import torch

from ideas.source_phase_compactness import compute_source_structure_loss
from ideas.v271_decomposition import smooth_temporal_trend
from ideas.v271_adaptive_structure import (
    compute_v271_adaptive_support_loss,
    load_v271_adaptive_supports,
)


def _toy_batch(batch_size=6, sequence_length=7, feature_dim=4):
    torch.manual_seed(7)
    base = torch.linspace(0.0, 1.0, steps=sequence_length).view(1, sequence_length, 1)
    class_offsets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.float32).view(batch_size, 1, 1)
    noise = 0.03 * torch.randn(batch_size, sequence_length, feature_dim)
    feats = base + 0.2 * class_offsets + noise
    positions = torch.arange(sequence_length).view(1, sequence_length).repeat(batch_size, 1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return feats, positions, labels


def test_v271_global_structure_loss_is_finite():
    feats, positions, labels = _toy_batch()

    loss, logs = compute_source_structure_loss(
        feats,
        positions,
        labels,
        version="v271_global",
        intra_trade_off=1.0,
        v271_trend_kernel_size=3,
        v271_trend_smoothing_mode="time",
        v271_trend_bandwidth=0.0,
        v271_trend_kernel="gaussian",
        v271_trend_dynamics_trade_off=0.05,
        v271_residual_variance_trade_off=0.10,
        v271_residual_energy_trade_off=0.05,
        v271_residual_energy_margin=1.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert logs["source_structure_v271_global_active"] == 1.0
    assert logs["source_structure_v271_global_trend_classes"] == 3.0
    assert logs["structure_loss"] >= 0.0


def test_time_aware_smoothing_handles_irregular_positions():
    curves = torch.tensor(
        [
            [[0.0], [1.0], [4.0], [5.0]],
            [[2.0], [3.0], [6.0], [7.0]],
        ]
    )
    irregular_positions = torch.tensor([[0, 1, 10, 11], [0, 1, 10, 11]])

    trend = smooth_temporal_trend(
        curves,
        positions=irregular_positions,
        kernel_size=3,
        mode="time",
        bandwidth=1.0,
        kernel="boxcar",
    )

    assert torch.isfinite(trend).all()
    assert trend.shape == curves.shape
    assert torch.allclose(trend[:, 0], curves[:, :2].mean(dim=1))
    assert torch.allclose(trend[:, 2], curves[:, 2:].mean(dim=1))


def test_v271_adaptive_support_loss_is_class_conditioned_and_finite():
    feats, positions, labels = _toy_batch()
    supports = [
        {
            "classes": [0, 1],
            "start": 1,
            "end": 5,
            "score": 0.8,
            "gate": 1.0,
        }
    ]

    loss, logs = compute_v271_adaptive_support_loss(
        feats,
        positions,
        labels,
        supports,
        trend_kernel_size=3,
        trend_smoothing_mode="time",
        trend_bandwidth=0.0,
        trend_kernel="gaussian",
        min_points=2,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert logs["timematch_v271_adaptive_active"] == 1.0
    assert logs["timematch_v271_adaptive_support_count"] == 1.0
    assert logs["timematch_v271_adaptive_class_count"] >= 2.0


def test_v271_adaptive_support_loader_filters_low_reliability(tmp_path):
    support_file = tmp_path / "supports.json"
    support_file.write_text(
        json.dumps(
            {
                "adaptive_supports": [
                    {"class_pair": [0, 1], "start": 1, "end": 4, "score": 0.8, "gate": 0.9},
                    {"class_pair": [1, 2], "start": 2, "end": 5, "score": 0.1, "gate": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )

    supports, logs = load_v271_adaptive_supports(str(support_file), min_score=0.5, min_gate=0.5)

    assert len(supports) == 1
    assert supports[0]["classes"] == [0, 1]
    assert logs["timematch_v271_adaptive_support_count"] == 1.0
