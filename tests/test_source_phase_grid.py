import unittest

from ideas.source_phase_grid import project_to_phase_grid

import torch


class SourcePhaseGridProjectionTest(unittest.TestCase):
    def test_projects_irregular_observations_to_fixed_phase_slots(self):
        feats = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [3.0, 0.0],
                    [9.0, 0.0],
                ]
            ]
        )
        positions = torch.tensor([[10, 30, 90]])

        grid, support = project_to_phase_grid(
            feats,
            positions,
            grid_count=5,
            kernel="linear",
            min_support=0.20,
        )

        self.assertEqual(tuple(grid.shape), (1, 5, 2))
        self.assertEqual(tuple(support.shape), (1, 5))
        self.assertTrue(torch.all(support >= 0.0))
        self.assertTrue(torch.all(support <= 1.0))
        self.assertGreater(float(support[0, 0]), 0.0)
        self.assertLess(float(support[0, 2]), 0.20)
        self.assertTrue(torch.allclose(grid[0, 0], feats[0, 0], atol=1e-4))
        self.assertTrue(torch.allclose(grid[0, -1], feats[0, -1], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
