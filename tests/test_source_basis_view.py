import unittest

import torch

from ideas.source_phase_compactness import (
    SourceSegmentWeightTracker,
    compute_source_structure_loss,
)


class SourceBasisViewTest(unittest.TestCase):
    def _fixture(self):
        features = torch.tensor(
            [
                [[0.0], [1.0], [2.0]],
                [[0.1], [1.1], [2.1]],
                [[10.0], [11.0], [12.0]],
                [[10.2], [11.2], [12.2]],
            ],
            dtype=torch.float32,
        )
        positions = torch.tensor([[0, 1, 2]] * features.shape[0])
        labels = torch.tensor([0, 0, 1, 1])
        partition = {"mode": "uniform", "phase_count": 3, "intervals": None, "date_positions": [0, 1, 2]}
        return features, positions, labels, partition

    def _tracker(self, partition):
        return SourceSegmentWeightTracker(
            3,
            phase_partition_spec=partition,
            min_sample_points_per_phase=1,
        )

    def test_basis_segment_only_matches_segment_intra(self):
        features, positions, labels, partition = self._fixture()
        segment_loss, _ = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=self._tracker(partition),
            version="segment_boundary_window_residual",
            trend_trade_off=0.0,
            segment_inter_trade_off=0.0,
            boundary_window_trade_off=0.0,
        )
        basis_loss, basis_logs = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=self._tracker(partition),
            version="basis_view",
            basis_global_trade_off=0.0,
            basis_segment_trade_off=1.0,
            basis_trend_trade_off=0.0,
            basis_boundary_trade_off=0.0,
            basis_dynamics_trade_off=0.0,
            basis_trajectory_trade_off=0.0,
        )

        self.assertTrue(torch.allclose(segment_loss, basis_loss, atol=1e-6))
        self.assertEqual(basis_logs["source_structure_view_family_basis"], 1.0)
        self.assertGreater(basis_logs["source_structure_basis_segment_weighted_loss"], 0.0)

    def test_basis_global_only_uses_whole_series_compactness(self):
        features, positions, labels, partition = self._fixture()
        basis_loss, basis_logs = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=self._tracker(partition),
            version="v271",
            basis_global_trade_off=1.0,
            basis_segment_trade_off=0.0,
            basis_trend_trade_off=0.0,
            basis_boundary_trade_off=0.0,
            basis_dynamics_trade_off=0.0,
            basis_trajectory_trade_off=0.0,
        )

        self.assertGreater(basis_loss.item(), 0.0)
        self.assertGreater(basis_logs["source_structure_global_intra_loss"], 0.0)
        self.assertGreater(basis_logs["source_structure_basis_global_weighted_loss"], 0.0)
        self.assertEqual(basis_logs["source_structure_basis_segment_weighted_loss"], 0.0)

    def test_basis_adaptive_segment_only_uses_discovered_partition(self):
        features, positions, labels, partition = self._fixture()
        adaptive_partition = {
            "mode": "adaptive_segment",
            "phase_count": 2,
            "segment_count": 2,
            "intervals": [(0, 1), (2, 2)],
        }
        basis_loss, basis_logs = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=self._tracker(partition),
            version="basis_view",
            basis_global_trade_off=0.0,
            basis_segment_trade_off=0.0,
            basis_trend_trade_off=0.0,
            basis_boundary_trade_off=0.0,
            basis_dynamics_trade_off=0.0,
            basis_trajectory_trade_off=0.0,
            basis_adaptive_segment_trade_off=1.0,
            adaptive_segment_partition_spec=adaptive_partition,
            adaptive_segment_min_sample_points=1,
        )

        self.assertGreater(basis_loss.item(), 0.0)
        self.assertEqual(basis_logs["source_structure_adaptive_segment_active"], 1.0)
        self.assertEqual(basis_logs["source_structure_adaptive_segment_count"], 2.0)
        self.assertGreater(basis_logs["source_structure_basis_adaptive_segment_weighted_loss"], 0.0)
        self.assertEqual(basis_logs["source_structure_basis_segment_weighted_loss"], 0.0)

    def test_basis_adaptive_segment_empty_partition_is_safe(self):
        features, positions, labels, partition = self._fixture()
        basis_loss, basis_logs = compute_source_structure_loss(
            features,
            positions,
            labels,
            weight_tracker=self._tracker(partition),
            version="basis_view",
            basis_global_trade_off=0.0,
            basis_segment_trade_off=0.0,
            basis_trend_trade_off=0.0,
            basis_boundary_trade_off=0.0,
            basis_dynamics_trade_off=0.0,
            basis_trajectory_trade_off=0.0,
            basis_adaptive_segment_trade_off=1.0,
            adaptive_segment_partition_spec=None,
        )

        self.assertEqual(basis_loss.item(), 0.0)
        self.assertEqual(basis_logs["source_structure_adaptive_segment_active"], 0.0)
        self.assertEqual(basis_logs["source_structure_basis_adaptive_segment_weighted_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
