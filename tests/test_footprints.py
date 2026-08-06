from __future__ import annotations

import unittest

from building_change.footprints import compare_footprints


def _feature(candidate_id: int, west: float, south: float, east: float, north: float) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "properties": {"candidate_id": candidate_id},
    }


class FootprintComparisonTests(unittest.TestCase):
    def test_identifies_new_footprint_and_extension_without_reclassifying_stable_building(self) -> None:
        before = {
            "type": "FeatureCollection",
            "features": [
                _feature(1, 145.4000, -36.3400, 145.4001, -36.3399),
                _feature(2, 145.4010, -36.3400, 145.4011, -36.3399),
            ],
            "metadata": {"requested_detector": "segformer"},
        }
        after = {
            "type": "FeatureCollection",
            "features": [
                _feature(11, 145.4000, -36.3400, 145.4001, -36.3399),
                _feature(12, 145.4010, -36.3400, 145.4012, -36.3399),
                _feature(13, 145.4030, -36.3400, 145.4031, -36.3399),
            ],
            "metadata": {"requested_detector": "segformer"},
        }

        result = compare_footprints(before, after, min_area_m2=20, extension_outside_fraction=0.2)

        classifications = [feature["properties"]["classification"] for feature in result["features"]]
        self.assertEqual(classifications.count("new_building_footprint_candidate"), 1)
        self.assertEqual(classifications.count("building_extension_footprint_candidate"), 1)
        self.assertEqual(result["metadata"]["evidence_role"], "object_footprint_refinement")


if __name__ == "__main__":
    unittest.main()
