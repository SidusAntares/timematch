import unittest

import torch

from ideas.source_phase_compactness import (
    SourceSegmentWeightTracker,
    compute_source_structure_loss,
)


class SourceElasticCompactnessTest(unittest.TestCase):
    def test_elastic_compactness_reduces_locally_shifted_peak_penalty(self):
        features = torch.tensor(
            [
                [[0.0], [10.0], [0.0]],
                [[0.0], [10.0], [0.0]],
                [[10.0], [0.0], [0.0]],
                [[100.0], [110.0], [100.0]],
                [[100.0], [110.0], [100.0]],
                [[110.0], [100.0], [100.0]],
            ],
            dtype=torch.float32,
        )
        positions = torch.tensor([[0, 1, 2]] * features.shape[0])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        partition = {"mode": "uniform", "phase_count": 3, "intervals": None, "date_positions": [0, 1, 2]}

        base_loss, base_logs = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=SourceSegmentWeightTracker(
                3,
                phase_partition_spec=partition,
                min_sample_points_per_phase=1,
            ),
            version="segment_boundary_window_residual",
            trend_trade_off=0.0,
            segment_inter_trade_off=0.0,
            boundary_window_trade_off=0.0,
            elastic_compactness=False,
        )
        elastic_loss, elastic_logs = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=SourceSegmentWeightTracker(
                3,
                phase_partition_spec=partition,
                min_sample_points_per_phase=1,
            ),
            version="segment_boundary_window_residual",
            trend_trade_off=0.0,
            segment_inter_trade_off=0.0,
            boundary_window_trade_off=0.0,
            elastic_compactness=True,
            elastic_radius=1,
            elastic_temperature=0.10,
        )

        self.assertLess(elastic_loss.item(), base_loss.item())
        self.assertEqual(elastic_logs["source_structure_elastic_compactness_active"], 1.0)
        self.assertEqual(base_logs["source_structure_elastic_compactness_active"], 0.0)
        self.assertGreater(elastic_logs["source_structure_elastic_classes"], 0.0)

    def test_elastic_blend_keeps_hard_intra_anchor(self):
        features = torch.tensor(
            [
                [[0.0], [10.0], [0.0]],
                [[0.0], [10.0], [0.0]],
                [[10.0], [0.0], [0.0]],
                [[100.0], [110.0], [100.0]],
                [[100.0], [110.0], [100.0]],
                [[110.0], [100.0], [100.0]],
            ],
            dtype=torch.float32,
        )
        positions = torch.tensor([[0, 1, 2]] * features.shape[0])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        partition = {"mode": "uniform", "phase_count": 3, "intervals": None, "date_positions": [0, 1, 2]}
        tracker_args = {
            "phase_count": 3,
            "phase_partition_spec": partition,
            "min_sample_points_per_phase": 1,
        }

        base_loss, _ = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=SourceSegmentWeightTracker(**tracker_args),
            version="segment_boundary_window_residual",
            trend_trade_off=0.0,
            segment_inter_trade_off=0.0,
            boundary_window_trade_off=0.0,
            elastic_compactness=False,
        )
        blend_loss, blend_logs = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=SourceSegmentWeightTracker(**tracker_args),
            version="segment_boundary_window_residual",
            trend_trade_off=0.0,
            segment_inter_trade_off=0.0,
            boundary_window_trade_off=0.0,
            elastic_compactness=True,
            elastic_radius=1,
            elastic_temperature=0.10,
            elastic_blend=0.25,
        )
        full_loss, _ = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=SourceSegmentWeightTracker(**tracker_args),
            version="segment_boundary_window_residual",
            trend_trade_off=0.0,
            segment_inter_trade_off=0.0,
            boundary_window_trade_off=0.0,
            elastic_compactness=True,
            elastic_radius=1,
            elastic_temperature=0.10,
            elastic_blend=1.00,
        )

        self.assertLess(full_loss.item(), blend_loss.item())
        self.assertLess(blend_loss.item(), base_loss.item())
        self.assertEqual(blend_logs["source_structure_elastic_blend"], 0.25)


if __name__ == "__main__":
    unittest.main()
