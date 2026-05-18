import unittest

import torch

from ideas.source_structure_reliability import _energy_reliability, compute_svd_structure_reliability_factors


class SourceStructureReliabilityTest(unittest.TestCase):
    def test_energy_reliability_is_higher_for_low_rank_samples(self):
        base = torch.arange(1, 7, dtype=torch.float32).unsqueeze(1)
        low_rank = torch.cat([base, 2.0 * base, -base], dim=1)
        full_rank = torch.eye(6, dtype=torch.float32)

        self.assertGreater(_energy_reliability(low_rank, zeta=0.90), _energy_reliability(full_rank, zeta=0.90))

    def test_reliability_factors_stay_in_conservative_bounds(self):
        spatial_feats = torch.randn(8, 6, 4)
        positions = torch.arange(6).repeat(8, 1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        spec = {"mode": "uniform", "phase_count": 3, "date_positions": list(range(6)), "intervals": None}

        factors, logs = compute_svd_structure_reliability_factors(
            spatial_feats,
            positions,
            labels,
            phase_partition_spec=spec,
            min_factor=0.7,
            max_factor=1.2,
        )

        self.assertEqual(set(factors), {"intra", "trend", "segment_inter", "boundary_window"})
        for value in factors.values():
            self.assertGreaterEqual(value, 0.7)
            self.assertLessEqual(value, 1.2)
        self.assertIn("source_structure_svd_intra_reliability", logs)


if __name__ == "__main__":
    unittest.main()
