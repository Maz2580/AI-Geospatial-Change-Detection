from __future__ import annotations

import unittest

from affine import Affine
import numpy as np
from rasterio.crs import CRS

from building_change.dino_buildings import (
    DinoBuildingsConfig,
    _building_probability,
    _footprint_collection,
    _segment_probability,
    _window_starts,
)


class _AlwaysBuildingSession:
    def run(self, output_names, input_feed):
        del output_names, input_feed
        logits = np.zeros((1, 3, 256, 256), dtype=np.float32)
        logits[:, 0] = 4.0
        return [logits]


class DinoBuildingsTests(unittest.TestCase):
    def test_sliding_windows_cover_each_edge(self) -> None:
        self.assertEqual(_window_starts(256, 256, 192), [0])
        self.assertEqual(_window_starts(600, 256, 192), [0, 192, 344])

    def test_first_softmax_channel_is_building_probability(self) -> None:
        logits = np.array([[[[2.0]], [[0.0]], [[0.0]]]], dtype=np.float32)
        self.assertGreater(float(_building_probability(logits)[0, 0]), 0.75)

    def test_segment_handles_an_image_smaller_than_one_model_window(self) -> None:
        probability = _segment_probability(
            _AlwaysBuildingSession(),
            np.full((3, 40, 50), 120.0, dtype=np.float32),
            np.ones((40, 50), dtype=bool),
            DinoBuildingsConfig(),
        )
        self.assertEqual(probability.shape, (40, 50))
        self.assertGreater(float(probability.min()), 0.9)

    def test_vectorised_footprints_keep_dated_model_provenance(self) -> None:
        probability = np.zeros((16, 16), dtype=np.float32)
        probability[4:8, 4:8] = 0.9
        collection = _footprint_collection(
            probability,
            np.ones((16, 16), dtype=bool),
            {"crs": CRS.from_epsg(3857), "transform": Affine(1, 0, 0, 0, -1, 16)},
            DinoBuildingsConfig(min_area_m2=10.0, simplify_m=0.0),
            capture_date="2026-04-18",
        )
        self.assertEqual(len(collection["features"]), 1)
        feature = collection["features"][0]
        self.assertEqual(feature["properties"]["capture_date"], "2026-04-18")
        self.assertEqual(feature["properties"]["source_model"], "hotosm/dinov3s-buildings")


if __name__ == "__main__":
    unittest.main()
