from __future__ import annotations

import unittest

from affine import Affine
import numpy as np
from rasterio.crs import CRS

from building_change.dino_buildings import (
    DinoBuildingsConfig,
    _building_probability,
    _footprint_collection,
    _normalise_tile,
    _segment_probability,
    _window_starts,
)


class _AlwaysBuildingSession:
    def run(self, output_names, input_feed):
        del output_names, input_feed
        logits = np.zeros((1, 3, 256, 256), dtype=np.float32)
        logits[:, 0] = 4.0
        return [logits]


class _RecordingSession(_AlwaysBuildingSession):
    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def run(self, output_names, input_feed):
        self.seen.append(input_feed["image"])
        return super().run(output_names, input_feed)


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

    def test_tiles_reach_the_model_imagenet_normalised_not_as_raw_bytes(self) -> None:
        # The frozen DINOv3 backbone saturates on raw 0-255 input.
        white = _normalise_tile(np.full((3, 4, 4), 255.0, dtype=np.float32))
        black = _normalise_tile(np.zeros((3, 4, 4), dtype=np.float32))

        self.assertAlmostEqual(float(white[0].mean()), (1.0 - 0.485) / 0.229, places=4)
        self.assertAlmostEqual(float(black[0].mean()), -0.485 / 0.229, places=4)

        session = _RecordingSession()
        _segment_probability(
            session,
            np.full((3, 40, 50), 255.0, dtype=np.float32),
            np.ones((40, 50), dtype=bool),
            DinoBuildingsConfig(),
        )
        self.assertTrue(session.seen)
        self.assertLessEqual(float(session.seen[0].max()), 3.0)

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
