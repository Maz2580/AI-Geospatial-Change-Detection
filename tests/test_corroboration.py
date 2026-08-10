from __future__ import annotations

import unittest

from building_change.corroboration import (
    CorroborationConfig,
    CorroborationError,
    corroborate_footprints,
)


def _square(west: float, south: float, size: float, candidate_id: int = 1) -> dict:
    east, north = west + size, south + size
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "properties": {"candidate_id": candidate_id, "classification": "new_building_footprint_candidate"},
    }


def _collection(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


# Roughly 100 m at this latitude; enough to separate features unambiguously.
DEG = 0.001


class CorroborationTests(unittest.TestCase):
    def test_overlapping_evidence_from_two_sources_is_corroborated(self) -> None:
        footprints = _collection(_square(145.400, -36.330, DEG))
        evidence = {
            "pixel_change": _collection(_square(145.400, -36.330, DEG / 2)),
            "dinov2_semantic": _collection(_square(145.4004, -36.3296, DEG / 2)),
        }

        report = corroborate_footprints(footprints, evidence)

        support = footprints["features"][0]["properties"]["change_support"]
        self.assertEqual(support["tier"], "corroborated")
        self.assertEqual(support["supporting_sources"], ["dinov2_semantic", "pixel_change"])
        self.assertEqual(report["tier_counts"]["corroborated"], 1)

    def test_a_single_nearby_source_is_only_weakly_corroborated(self) -> None:
        footprints = _collection(_square(145.400, -36.330, DEG))
        evidence = {"pixel_change": _collection(_square(145.400, -36.330, DEG / 4))}

        corroborate_footprints(footprints, evidence)

        self.assertEqual(footprints["features"][0]["properties"]["change_support"]["tier"], "weakly_corroborated")

    def test_distant_evidence_leaves_a_footprint_unsupported(self) -> None:
        footprints = _collection(_square(145.400, -36.330, DEG))
        evidence = {"pixel_change": _collection(_square(145.450, -36.380, DEG))}

        corroborate_footprints(footprints, evidence)

        support = footprints["features"][0]["properties"]["change_support"]
        self.assertEqual(support["tier"], "footprint_only")
        self.assertEqual(support["supporting_sources"], [])

    def test_no_candidate_is_ever_dropped(self) -> None:
        footprints = _collection(
            _square(145.400, -36.330, DEG, 1),
            _square(145.450, -36.380, DEG, 2),
            _square(145.470, -36.390, DEG, 3),
        )
        evidence = {"pixel_change": _collection(_square(145.400, -36.330, DEG / 2))}

        report = corroborate_footprints(footprints, evidence)

        self.assertEqual(len(footprints["features"]), 3)
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(sum(report["tier_counts"].values()), 3)

    def test_records_per_source_overlap_and_distance(self) -> None:
        footprints = _collection(_square(145.400, -36.330, DEG))
        evidence = {"pixel_change": _collection(_square(145.400, -36.330, DEG))}

        corroborate_footprints(footprints, evidence)

        per_source = footprints["features"][0]["properties"]["change_support"]["per_source"]["pixel_change"]
        self.assertGreater(per_source["overlap_fraction"], 0.9)
        self.assertEqual(per_source["distance_m"], 0.0)

    def test_runs_without_any_evidence_sources(self) -> None:
        footprints = _collection(_square(145.400, -36.330, DEG))

        report = corroborate_footprints(footprints, {})

        self.assertEqual(report["evidence_sources"], [])
        self.assertEqual(footprints["features"][0]["properties"]["change_support"]["tier"], "footprint_only")

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(CorroborationError):
            CorroborationConfig(support_distance_m=-1).validate()
        with self.assertRaises(CorroborationError):
            CorroborationConfig(strong_support_fraction=1.5).validate()
        with self.assertRaises(CorroborationError):
            CorroborationConfig(min_sources_for_corroborated=0).validate()

    def test_rejects_input_that_is_not_a_feature_collection(self) -> None:
        with self.assertRaises(CorroborationError):
            corroborate_footprints({"type": "Feature"}, {})
        with self.assertRaises(CorroborationError):
            corroborate_footprints(_collection(_square(145.4, -36.33, DEG)), {"bad": {"type": "Feature"}})


if __name__ == "__main__":
    unittest.main()
