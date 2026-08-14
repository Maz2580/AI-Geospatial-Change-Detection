from __future__ import annotations

import unittest

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from building_change.geodesy import (
    GeodesyError,
    geodesic_area_m2,
    ground_area_m2,
    metric_crs_for,
    to_metric,
    utm_epsg_for,
    web_mercator_area_inflation,
)

# A block in Melbourne's western CBD, where the gold set and pilot imagery sit.
MELBOURNE_BLOCK = box(144.9500, -37.8100, 144.9512, -37.8091)
MELBOURNE_TRUE_AREA_M2 = 10_555.9


class GroundAreaTests(unittest.TestCase):
    def test_geodesic_area_matches_the_local_projected_area(self) -> None:
        projected, _ = to_metric(MELBOURNE_BLOCK)
        self.assertAlmostEqual(geodesic_area_m2(MELBOURNE_BLOCK), projected.area, delta=5.0)
        self.assertAlmostEqual(geodesic_area_m2(MELBOURNE_BLOCK), MELBOURNE_TRUE_AREA_M2, delta=5.0)

    def test_web_mercator_area_is_not_ground_area(self) -> None:
        """The bug this module exists to stop: 3857 is projected but not metres."""
        to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        in_mercator = shapely_transform(to_mercator, MELBOURNE_BLOCK)

        # What the old code did: trust `is_projected` and read the area off.
        self.assertGreater(in_mercator.area, MELBOURNE_TRUE_AREA_M2 * 1.5)

        # What the helper does.
        self.assertAlmostEqual(
            ground_area_m2(in_mercator, "EPSG:3857"), MELBOURNE_TRUE_AREA_M2, delta=5.0
        )

    def test_inflation_factor_is_reported_for_the_measured_latitudes(self) -> None:
        self.assertAlmostEqual(web_mercator_area_inflation(-37.81), 1.602, places=2)
        self.assertAlmostEqual(web_mercator_area_inflation(-36.34), 1.541, places=2)

    def test_a_true_metre_crs_is_left_alone(self) -> None:
        projected, crs = to_metric(MELBOURNE_BLOCK, "EPSG:7855")
        self.assertAlmostEqual(ground_area_m2(projected, crs), MELBOURNE_TRUE_AREA_M2, delta=5.0)

    def test_geographic_geometry_is_measured_geodesically(self) -> None:
        self.assertAlmostEqual(
            ground_area_m2(MELBOURNE_BLOCK, "EPSG:4326"), MELBOURNE_TRUE_AREA_M2, delta=5.0
        )

    def test_empty_geometry_has_no_area(self) -> None:
        self.assertEqual(ground_area_m2(box(0, 0, 0, 0), "EPSG:4326"), 0.0)

    def test_area_without_a_crs_is_refused(self) -> None:
        with self.assertRaises(GeodesyError):
            ground_area_m2(MELBOURNE_BLOCK, None)


class MetricCrsTests(unittest.TestCase):
    def test_victorian_geometry_lands_in_utm_zone_55_south(self) -> None:
        self.assertEqual(utm_epsg_for(144.96, -37.81), 32755)
        self.assertEqual(metric_crs_for(MELBOURNE_BLOCK).to_epsg(), 32755)

    def test_zone_and_hemisphere_follow_the_coordinate(self) -> None:
        self.assertEqual(utm_epsg_for(-0.1, 51.5), 32630)  # London, zone 30 north
        self.assertEqual(utm_epsg_for(151.2, -33.9), 32756)  # Sydney, zone 56 south

    def test_distances_are_ground_distances(self) -> None:
        """0.0008 degrees of latitude is 88.8 m of ground, and must measure so.

        The same separation read off EPSG:3857 would come back as 112 m.
        """
        south, north = box(144.9500, -37.8100, 144.9501, -37.8099), box(144.9500, -37.8091, 144.9501, -37.8090)
        crs = metric_crs_for(south)
        projected_south, _ = to_metric(south, crs)
        projected_north, _ = to_metric(north, crs)
        self.assertAlmostEqual(projected_south.distance(projected_north), 88.8, delta=0.5)

        in_mercator_south, _ = to_metric(south, "EPSG:3857")
        in_mercator_north, _ = to_metric(north, "EPSG:3857")
        self.assertGreater(in_mercator_south.distance(in_mercator_north), 110.0)

    def test_an_out_of_range_coordinate_is_refused(self) -> None:
        with self.assertRaises(GeodesyError):
            utm_epsg_for(400.0, 0.0)


if __name__ == "__main__":
    unittest.main()
