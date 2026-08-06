from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import transform

from building_change.review import write_candidate_review


class ReviewTests(unittest.TestCase):
    def test_review_accepts_a_fused_multipolygon_candidate(self) -> None:
        longitude, latitude = transform("EPSG:3857", "EPSG:4326", [10, 30], [90, 70])
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[longitude[0], latitude[0]], [longitude[0] + 0.00001, latitude[0]], [longitude[0], latitude[0] - 0.00001], [longitude[0], latitude[0]]]],
                    [[[longitude[1], latitude[1]], [longitude[1] + 0.00001, latitude[1]], [longitude[1], latitude[1] - 0.00001], [longitude[1], latitude[1]]]],
                ],
            },
            "properties": {"candidate_id": 1, "classification": "new_building"},
        }
        profile = {"crs": "EPSG:3857", "transform": from_origin(0, 100, 1, 1)}
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            index = write_candidate_review(image, image, profile, [feature], directory)
            self.assertTrue(index.is_file())
            self.assertTrue((Path(directory) / "review" / "candidate_001_before.png").is_file())


if __name__ == "__main__":
    unittest.main()
