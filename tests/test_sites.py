from __future__ import annotations

import unittest

from building_change.sites import build_sites


def _feature(candidate_id: int, west: float, south: float, east: float, north: float, classification: str, **properties: object) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]},
        "properties": {"candidate_id": candidate_id, "classification": classification, **properties},
    }


class SiteGroupingTests(unittest.TestCase):
    def test_attaches_nearby_footprint_without_counting_it_as_an_independent_vote(self) -> None:
        changes = {
            "type": "FeatureCollection",
            "features": [
                _feature(1, 145.4000, -36.3400, 145.4001, -36.3399, "new building", candidate_sources=["dfine", "segformer"], source_count=2),
                _feature(2, 145.4020, -36.3400, 145.4021, -36.3399, "paved", candidate_sources=["pixel"], source_count=1),
            ],
        }
        footprints = {
            "type": "FeatureCollection",
            "features": [
                _feature(10, 145.40015, -36.3400, 145.40025, -36.3399, "new_building_footprint_candidate"),
                _feature(11, 145.4050, -36.3400, 145.4051, -36.3399, "new_building_footprint_candidate"),
            ],
        }

        result = build_sites(changes, footprints, anchor_merge_distance_m=3, site_radius_m=25)

        self.assertEqual(len(result["features"]), 3)
        attached = next(feature for feature in result["features"] if feature["properties"]["change_candidate_ids"] == [1])
        self.assertEqual(attached["properties"]["footprint_candidate_ids"], [10])
        self.assertEqual(attached["properties"]["review_priority"], "medium")
        standalone = next(feature for feature in result["features"] if feature["properties"]["classification"] == "footprint_only_site_candidate")
        self.assertEqual(standalone["properties"]["footprint_candidate_ids"], [11])


if __name__ == "__main__":
    unittest.main()
