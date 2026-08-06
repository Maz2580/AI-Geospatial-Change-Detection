from __future__ import annotations

import unittest

import numpy as np
from rasterio.transform import Affine

from building_change.expansion import expand_existing_candidates


class ExpansionTests(unittest.TestCase):
    def test_growth_cannot_create_an_unrelated_candidate(self) -> None:
        score = np.zeros((50, 50), dtype=np.float32)
        seeds = np.zeros_like(score, dtype=bool)
        seeds[20:25, 20:25] = True
        score[20:25, 20:25] = 1.0
        score[19:28, 19:30] = np.maximum(score[19:28, 19:30], 0.7)  # Connected missing section.
        score[2:10, 2:10] = 0.95  # Strong but unrelated change.
        expanded, _ = expand_existing_candidates(
            seeds,
            score,
            np.ones_like(seeds, dtype=bool),
            {"transform": Affine.identity(), "crs": None},
            grow_percentile=80,
            max_distance_m=6,
            closing_m=0,
        )
        self.assertTrue(expanded[19:28, 19:30].any())
        self.assertFalse(expanded[2:10, 2:10].any())


if __name__ == "__main__":
    unittest.main()
