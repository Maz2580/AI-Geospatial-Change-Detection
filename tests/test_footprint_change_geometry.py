"""An extension candidate must describe the extension, not the whole house."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform

from building_change.footprints import (
    compare_footprints,
    substantial_change,
    write_comparison_outputs,
)

# A block in Melbourne, projected so the test can state sizes in real metres.
_TO_WGS84 = Transformer.from_crs("EPSG:7855", "EPSG:4326", always_xy=True).transform
_ORIGIN_E, _ORIGIN_N = 320_000.0, 5_812_000.0


def _feature(east: float, north: float, width: float, depth: float, candidate_id: int) -> dict:
    metric = box(_ORIGIN_E + east, _ORIGIN_N + north, _ORIGIN_E + east + width, _ORIGIN_N + north + depth)
    return {
        "type": "Feature",
        "geometry": mapping(shapely_transform(_TO_WGS84, metric)),
        "properties": {"candidate_id": candidate_id, "classification": "building_footprint"},
    }


def _collection(features: list[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {"requested_detector": "test"},
    }


class ExtensionGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        # A 20 x 20 m house that grows a 20 x 10 m rear wing: 400 m2 becomes 600.
        # The wing is a third of the result, clearing the 0.25 extension fraction.
        self.before = _collection([_feature(0, 0, 20, 20, 1)])
        self.after = _collection([_feature(0, 0, 20, 30, 1)])

    def test_an_extension_reports_only_the_new_wing(self) -> None:
        collection = compare_footprints(self.before, self.after, min_area_m2=10.0)

        self.assertEqual(len(collection["features"]), 1)
        properties = collection["features"][0]["properties"]
        self.assertEqual(properties["classification"], "building_extension_footprint_candidate")

        # The change is 200 m2. The house is 600 m2. Before this fix the
        # candidate reported 600 -- three times what was actually built.
        self.assertAlmostEqual(properties["changed_area_m2"], 200.0, delta=3.0)
        self.assertAlmostEqual(properties["object_area_m2"], 600.0, delta=6.0)
        self.assertEqual(properties["area_m2"], properties["changed_area_m2"])

    def test_the_candidate_geometry_is_the_change_itself(self) -> None:
        collection = compare_footprints(self.before, self.after, min_area_m2=10.0)
        geometry = shape(collection["features"][0]["geometry"])

        to_metric = Transformer.from_crs("EPSG:4326", "EPSG:7855", always_xy=True).transform
        self.assertAlmostEqual(shapely_transform(to_metric, geometry).area, 200.0, delta=3.0)

    def test_a_new_building_reports_its_whole_footprint(self) -> None:
        """With nothing there before, the change and the object coincide."""
        after = _collection([_feature(200, 200, 12, 10, 1)])
        collection = compare_footprints(self.before, after, min_area_m2=10.0)

        properties = collection["features"][0]["properties"]
        self.assertEqual(properties["classification"], "new_building_footprint_candidate")
        self.assertAlmostEqual(properties["changed_area_m2"], properties["object_area_m2"], delta=0.1)

    def test_the_containing_object_is_written_to_its_own_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = write_comparison_outputs(directory, self.before, self.after, min_area_m2=10.0)

            candidates = json.loads(Path(report["outputs"]["footprint_candidates"]).read_text(encoding="utf-8"))
            objects = json.loads(Path(report["outputs"]["footprint_change_objects"]).read_text(encoding="utf-8"))

        # The candidate layer must not leak the out-of-band carrier.
        self.assertNotIn("_object_geometry", candidates["features"][0])
        self.assertEqual(len(objects["features"]), 1)
        self.assertEqual(
            objects["features"][0]["properties"]["candidate_id"],
            candidates["features"][0]["properties"]["candidate_id"],
        )
        self.assertGreater(
            objects["features"][0]["properties"]["area_m2"],
            candidates["features"][0]["properties"]["area_m2"],
        )
        self.assertAlmostEqual(report["changed_area_m2_total"], 200.0, delta=3.0)
        self.assertAlmostEqual(report["containing_object_area_m2_total"], 600.0, delta=6.0)


class SliverRejectionTests(unittest.TestCase):
    """Two segmentations of one unchanged roof must not become an extension."""

    def test_a_perimeter_ribbon_is_discarded(self) -> None:
        # A 20 x 20 m roof outlined 0.4 m larger on the after date: the
        # difference is a 16 m2 ring, wider than the 10 m2 floor but only
        # 0.4 m thick.
        roof = box(0, 0, 20, 20)
        ribbon = box(-0.4, -0.4, 20.4, 20.4).difference(roof)
        self.assertGreater(ribbon.area, 10.0)

        kept = substantial_change(ribbon, min_change_width_m=1.5, min_change_area_m2=10.0)
        self.assertTrue(kept.is_empty)

    def test_a_real_wing_survives(self) -> None:
        wing = box(0, 0, 20, 10)
        kept = substantial_change(wing, min_change_width_m=1.5, min_change_area_m2=10.0)
        self.assertAlmostEqual(kept.area, 200.0, delta=0.1)

    def test_scattered_islands_are_dropped_but_a_wing_beside_them_is_kept(self) -> None:
        from shapely.ops import unary_union

        islands = unary_union([box(30 + 3 * i, 0, 30 + 3 * i + 0.5, 4) for i in range(4)])
        mixed = unary_union([box(0, 0, 12, 12), islands])

        kept = substantial_change(mixed, min_change_width_m=1.5, min_change_area_m2=10.0)
        self.assertAlmostEqual(kept.area, 144.0, delta=0.1)

    def test_an_unchanged_building_produces_no_extension_candidate(self) -> None:
        """The Murchison holdout failure, end to end."""
        before = _collection([_feature(0, 0, 20, 20, 1)])
        # The same roof, outlined 0.4 m larger. Nothing was built.
        after = _collection([_feature(-0.4, -0.4, 20.8, 20.8, 1)])

        collection = compare_footprints(before, after, min_area_m2=10.0)
        self.assertEqual(collection["features"], [])

    def test_the_discarded_sliver_area_is_reported(self) -> None:
        """A reviewer must be able to see that the difference was edge noise."""
        before = _collection([_feature(0, 0, 20, 20, 1)])
        after = _collection([_feature(0, 0, 20, 30, 1)])

        collection = compare_footprints(before, after, min_area_m2=10.0)
        self.assertIn("discarded_sliver_area_m2", collection["features"][0]["properties"])


if __name__ == "__main__":
    unittest.main()
