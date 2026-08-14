from __future__ import annotations

import copy
import unittest

from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

from building_change.identity import (
    IdentityError,
    LinkConfig,
    assign_identities,
    location_key,
    provenance,
    site_history,
)

_TO_WGS84 = Transformer.from_crs("EPSG:7855", "EPSG:4326", always_xy=True).transform
_ORIGIN_E, _ORIGIN_N = 320_000.0, 5_812_000.0


def _feature(east: float, north: float, width: float = 12.0, depth: float = 10.0, **properties) -> dict:
    metric = box(_ORIGIN_E + east, _ORIGIN_N + north, _ORIGIN_E + east + width, _ORIGIN_N + north + depth)
    return {
        "type": "Feature",
        "geometry": mapping(shapely_transform(_TO_WGS84, metric)),
        "properties": {"classification": "new_building_footprint_candidate", **properties},
    }


def _collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features, "metadata": {}}


class LocationKeyTests(unittest.TestCase):
    def test_the_key_is_readable_and_encodes_the_zone(self) -> None:
        key = location_key(_feature(0, 0)["geometry"])
        self.assertTrue(key.startswith("UTM55S-"), key)
        self.assertEqual(len(key.split("-")), 3)

    def test_the_same_place_gives_the_same_key(self) -> None:
        self.assertEqual(location_key(_feature(0, 0)["geometry"]), location_key(_feature(0, 0)["geometry"]))

    def test_places_far_apart_give_different_keys(self) -> None:
        self.assertNotEqual(location_key(_feature(0, 0)["geometry"]), location_key(_feature(500, 500)["geometry"]))

    def test_an_empty_geometry_is_refused(self) -> None:
        with self.assertRaises(IdentityError):
            location_key(box(0, 0, 0, 0))


class IdentityLinkTests(unittest.TestCase):
    def test_a_first_survey_creates_new_sites(self) -> None:
        result = assign_identities(_collection([_feature(0, 0), _feature(100, 100)]), survey_date="2022-02-02")

        for feature in result["features"]:
            self.assertEqual(feature["properties"]["site_status"], "new_site")
            self.assertEqual(feature["properties"]["observation_count"], 1)
            self.assertEqual(feature["properties"]["first_seen"], "2022-02-02")
        ids = {feature["properties"]["site_id"] for feature in result["features"]}
        self.assertEqual(len(ids), 2)

    def test_the_same_building_keeps_its_id_in_a_later_survey(self) -> None:
        first = assign_identities(_collection([_feature(0, 0)]), survey_date="2022-02-02")
        # Remeasured on the later image: same roof, outline a little different.
        second = assign_identities(
            _collection([_feature(0.6, -0.4, width=12.5, depth=10.3)]),
            previous=first,
            survey_date="2026-04-18",
        )

        properties = second["features"][0]["properties"]
        self.assertEqual(properties["site_id"], first["features"][0]["properties"]["site_id"])
        self.assertEqual(properties["first_seen"], "2022-02-02")
        self.assertEqual(properties["observation_count"], 2)
        self.assertEqual(properties["site_status"], "seen_before")

    def test_the_id_does_not_drift_when_the_outline_is_remeasured(self) -> None:
        """The id must come from the first observation, not be recomputed."""
        first = assign_identities(_collection([_feature(0, 0)]), survey_date="2022-02-02")
        original = first["features"][0]["properties"]["site_id"]

        current = first
        for date in ("2023-01-01", "2024-01-01", "2026-04-18"):
            current = assign_identities(
                _collection([_feature(1.2, 0.9, width=13.0, depth=11.0)]), previous=current, survey_date=date
            )
        properties = current["features"][0]["properties"]
        self.assertEqual(properties["site_id"], original)
        self.assertEqual(properties["observation_count"], 4)
        # Its own location has moved on, which is exactly why the key is not the id.
        self.assertNotEqual(properties["location_key"], original)

    def test_a_genuinely_new_building_gets_its_own_id(self) -> None:
        first = assign_identities(_collection([_feature(0, 0)]), survey_date="2022-02-02")
        second = assign_identities(
            _collection([_feature(0, 0), _feature(300, 300)]), previous=first, survey_date="2026-04-18"
        )

        statuses = {f["properties"]["site_status"] for f in second["features"]}
        self.assertEqual(statuses, {"seen_before", "new_site"})

    def test_one_earlier_observation_is_not_claimed_twice(self) -> None:
        """Two candidates near one old site must not both inherit its id."""
        first = assign_identities(_collection([_feature(0, 0, width=30, depth=30)]), survey_date="2022-02-02")
        second = assign_identities(
            _collection([_feature(0, 0, width=14, depth=30), _feature(16, 0, width=14, depth=30)]),
            previous=first,
            survey_date="2026-04-18",
        )

        ids = [f["properties"]["site_id"] for f in second["features"]]
        self.assertEqual(len(set(ids)), 2)

    def test_a_distant_building_is_not_linked(self) -> None:
        first = assign_identities(_collection([_feature(0, 0)]), survey_date="2022-02-02")
        second = assign_identities(
            _collection([_feature(0, 60)]),
            previous=first,
            survey_date="2026-04-18",
            config=LinkConfig(max_centroid_distance_m=8.0),
        )
        self.assertEqual(second["features"][0]["properties"]["site_status"], "new_site")

    def test_an_empty_collection_is_returned_unchanged(self) -> None:
        empty = _collection([])
        self.assertEqual(assign_identities(copy.deepcopy(empty)), empty)

    def test_a_bad_link_configuration_is_refused(self) -> None:
        with self.assertRaises(IdentityError):
            LinkConfig(min_overlap_fraction=1.5).validate()


class HistoryTests(unittest.TestCase):
    def test_observations_group_by_site(self) -> None:
        first = assign_identities(_collection([_feature(0, 0)]), survey_date="2022-02-02")
        second = assign_identities(_collection([_feature(0.5, 0.5)]), previous=first, survey_date="2026-04-18")

        history = site_history([first, second])
        self.assertEqual(len(history), 1)
        self.assertEqual([entry["survey_date"] for entry in next(iter(history.values()))],
                         ["2022-02-02", "2026-04-18"])


class ProvenanceTests(unittest.TestCase):
    def test_provenance_records_what_is_needed_to_defend_a_candidate(self) -> None:
        record = provenance(
            before_image="before.tif",
            after_image="after.tif",
            before_date="2022-02-02",
            after_date="2026-04-18",
            model="hotosm/dinov3s-buildings",
            threshold=0.4371,
            commit="abc1234",
        )
        self.assertEqual(record["status"], "review_candidate_not_a_finding")
        self.assertEqual(record["code_commit"], "abc1234")
        self.assertEqual(record["before_capture_date"], "2022-02-02")

    def test_an_unrecorded_commit_is_explicit_rather_than_missing(self) -> None:
        record = provenance(
            before_image="a.tif", after_image="b.tif", before_date=None, after_date=None,
            model="m", threshold=0.5,
        )
        self.assertEqual(record["code_commit"], "unrecorded")
        self.assertEqual(record["before_capture_date"], "unknown")


if __name__ == "__main__":
    unittest.main()
