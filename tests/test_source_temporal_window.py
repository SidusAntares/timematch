import unittest

import torch

from ideas.source_temporal_window import compute_source_target_soft_support_mask


class SourceTemporalWindowTest(unittest.TestCase):
    def test_soft_support_sorts_target_features_by_target_positions(self):
        source_features = torch.tensor(
            [
                [[0.0], [0.0]],
                [[0.0], [0.0]],
                [[10.0], [0.0]],
                [[10.0], [0.0]],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1])
        target_ordered = torch.tensor(
            [
                [[0.1], [5.0]],
                [[9.9], [5.0]],
            ],
            dtype=torch.float32,
        )
        target_shuffled = target_ordered[:, [1, 0], :]
        target_positions = torch.tensor([[1, 0], [1, 0]])

        ordered_weights, ordered_logs = compute_source_target_soft_support_mask(
            source_features,
            labels,
            target_ordered,
            smooth_kernel_size=1,
            reliability_gate=False,
            min_weight=0.2,
        )
        shuffled_weights, shuffled_logs = compute_source_target_soft_support_mask(
            source_features,
            labels,
            target_shuffled,
            target_positions=target_positions,
            smooth_kernel_size=1,
            reliability_gate=False,
            min_weight=0.2,
        )

        self.assertTrue(torch.allclose(ordered_weights, shuffled_weights, atol=1e-6))
        self.assertIn("source_structure_support_entropy", shuffled_logs)
        self.assertIn("source_structure_support_effective_size", shuffled_logs)
        self.assertIn("source_structure_support_scores", shuffled_logs)
        self.assertEqual(ordered_logs["source_structure_support_weights"], shuffled_logs["source_structure_support_weights"])


if __name__ == "__main__":
    unittest.main()
