import json

import torch

from ideas.source_phase_compactness import compute_source_structure_loss
from ideas.v271_decomposition import smooth_temporal_trend
from ideas.v271_adaptive_structure import (
    compute_v271_adaptive_support_loss,
    load_v271_adaptive_supports,
)
from ideas.v271_adaptive_support_discovery import construct_pair_adaptive_supports


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


def test_v271_adaptive_support_loss_accepts_boundary_taper():
    feats, positions, labels = _toy_batch()
    supports = [
        {
            "classes": [0, 1],
            "start": 2,
            "end": 4,
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
        taper_mode="triangular",
        taper_ratio=1.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert logs["timematch_v271_adaptive_taper_active"] == 1.0
    assert logs["timematch_v271_adaptive_taper_radius"] > 0.0
    assert logs["timematch_v271_adaptive_support_weight_mean"] > 0.0


def test_v271_adaptive_support_loss_skips_all_off_gates_without_nan():
    feats, positions, labels = _toy_batch()
    supports = [
        {
            "classes": [0, 1],
            "start": 1,
            "end": 5,
            "score": 0.8,
            "gate": 0.0,
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
    assert loss.item() == 0.0
    assert logs["timematch_v271_adaptive_candidate_support_count"] == 1.0
    assert logs["timematch_v271_adaptive_gate_off_count"] == 1.0
    assert logs["timematch_v271_adaptive_active"] == 0.0


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


def test_construct_pair_adaptive_supports_merges_by_class_pair():
    pair_segment_rows = [
        {
            "pair": "0:1",
            "segment": 1,
            "start": 0,
            "end": 9,
            "score": 0.60,
            "support_count": 4,
            "source_separability": 1.2,
            "target_explainability": 0.7,
            "ambiguity": 0.8,
            "shift_stability": 1.0,
            "actual_raw_score": 0.70,
            "baseline_raw_score": 0.10,
            "relative_score": 0.60,
            "ratio_score": 7.0,
        },
        {
            "pair": "0:1",
            "segment": 2,
            "start": 10,
            "end": 19,
            "score": 0.50,
            "support_count": 3,
            "source_separability": 1.1,
            "target_explainability": 0.6,
            "ambiguity": 0.7,
            "shift_stability": 1.0,
            "actual_raw_score": 0.55,
            "baseline_raw_score": 0.05,
            "relative_score": 0.50,
            "ratio_score": 11.0,
        },
        {
            "pair": "1:2",
            "segment": 1,
            "start": 0,
            "end": 9,
            "score": 0.02,
            "support_count": 3,
            "source_separability": 0.2,
            "target_explainability": 0.3,
            "ambiguity": 0.4,
            "shift_stability": 0.5,
            "actual_raw_score": 0.03,
            "baseline_raw_score": 0.01,
            "relative_score": 0.02,
            "ratio_score": 3.0,
        },
    ]

    threshold, supports = construct_pair_adaptive_supports(
        pair_segment_rows,
        score_quantile=0.0,
        min_score=0.10,
        min_ratio=1.0,
        top_m_per_pair=2,
        max_supports=8,
    )

    assert threshold == 0.10
    assert len(supports) == 1
    support = supports[0]
    assert support["classes"] == [0, 1]
    assert support["class_pair"] == [0, 1]
    assert support["start"] == 0
    assert support["end"] == 19
    assert support["atomic_segments"] == [1, 2]
    assert 0.0 <= support["gate"] <= 1.0


def test_construct_pair_adaptive_supports_applies_strict_reliability_filters():
    pair_segment_rows = [
        {
            "pair": "0:1",
            "segment": 1,
            "start": 0,
            "end": 9,
            "score": 0.60,
            "support_count": 2,
            "source_separability": 1.2,
            "target_explainability": 0.7,
            "ambiguity": 0.8,
            "shift_stability": 1.0,
            "actual_raw_score": 0.70,
            "baseline_raw_score": 0.10,
            "relative_score": 0.60,
            "ratio_score": 7.0,
        },
        {
            "pair": "0:1",
            "segment": 2,
            "start": 10,
            "end": 19,
            "score": 0.50,
            "support_count": 20,
            "source_separability": 1.1,
            "target_explainability": 0.6,
            "ambiguity": 0.7,
            "shift_stability": 0.33,
            "actual_raw_score": 0.55,
            "baseline_raw_score": 0.05,
            "relative_score": 0.50,
            "ratio_score": 11.0,
        },
        {
            "pair": "1:2",
            "segment": 3,
            "start": 20,
            "end": 29,
            "score": 0.40,
            "support_count": 20,
            "source_separability": 1.0,
            "target_explainability": 0.6,
            "ambiguity": 0.7,
            "shift_stability": 1.0,
            "actual_raw_score": 0.45,
            "baseline_raw_score": 0.05,
            "relative_score": 0.40,
            "ratio_score": 9.0,
        },
    ]

    _, supports = construct_pair_adaptive_supports(
        pair_segment_rows,
        score_quantile=0.0,
        min_score=0.10,
        min_ratio=1.0,
        min_support_count=10,
        min_shift_stability=0.66,
        gate_score_high=0.80,
    )

    assert len(supports) == 1
    assert supports[0]["classes"] == [1, 2]
    assert supports[0]["gate"] == 0.5


def test_construct_pair_adaptive_supports_can_use_conservative_discrete_gate():
    pair_segment_rows = [
        {
            "pair": "0:1",
            "segment": 1,
            "start": 0,
            "end": 9,
            "score": 1.00,
            "support_count": 40,
            "source_separability": 3.0,
            "target_explainability": 0.90,
            "ambiguity": 0.9,
            "shift_stability": 1.0,
            "actual_raw_score": 1.10,
            "baseline_raw_score": 0.10,
            "relative_score": 1.00,
            "ratio_score": 11.0,
        },
        {
            "pair": "1:2",
            "segment": 2,
            "start": 10,
            "end": 19,
            "score": 0.55,
            "support_count": 20,
            "source_separability": 1.0,
            "target_explainability": 0.60,
            "ambiguity": 0.7,
            "shift_stability": 0.83,
            "actual_raw_score": 0.60,
            "baseline_raw_score": 0.15,
            "relative_score": 0.45,
            "ratio_score": 4.0,
        },
    ]

    _, supports = construct_pair_adaptive_supports(
        pair_segment_rows,
        score_quantile=0.0,
        min_score=0.10,
        min_ratio=1.2,
        min_support_count=10,
        min_shift_stability=0.66,
        max_supports=8,
        gate_mode="conservative",
        gate_low=0.30,
        gate_high=0.70,
        gate_light=0.30,
        gate_full=0.80,
        gate_ratio_high=4.0,
        gate_source_sep_high=2.0,
        gate_target_explain_high=0.70,
    )

    assert len(supports) == 2
    assert supports[0]["classes"] == [0, 1]
    assert supports[0]["gate"] == 0.80
    assert supports[0]["gate_level"] == "full"
    assert supports[0]["reliability"] >= 0.70
    assert supports[1]["classes"] == [1, 2]
    assert supports[1]["gate"] == 0.30
    assert supports[1]["gate_level"] == "light"
    assert 0.30 <= supports[1]["reliability"] < 0.70
