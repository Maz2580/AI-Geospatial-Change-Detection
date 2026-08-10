from __future__ import annotations

import math
import unittest

from pyproj import CRS
from shapely.geometry import Polygon

from building_change.regularisation import (
    RegularisationConfig,
    RegularisationError,
    regularise_geometries,
    tolerance_for_pixel_size,
)


def _staircase_rectangle(step: float = 0.1) -> Polygon:
    """A 10 x 6 m rectangle whose long edges are broken into pixel-sized steps."""
    coords = []
    for index in range(100):
        x = index * step
        coords.append((x, 0.0 if index % 2 == 0 else step))
    coords.append((10.0, 6.0))
    for index in range(100):
        x = 10.0 - index * step
        coords.append((x, 6.0 if index % 2 == 0 else 6.0 - step))
    coords.append((0.0, 0.0))
    return Polygon(coords).buffer(0)


def _edge_angle_deviation(geometry: Polygon) -> float:
    """Length-weighted mean deviation of edges from the polygon's dominant angle."""
    coords = list(geometry.exterior.coords)
    segments = []
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 1e-9:
            segments.append((math.degrees(math.atan2(y1 - y0, x1 - x0)) % 90.0, length))
    dominant = max(segments, key=lambda item: item[1])[0]
    total = sum(length for _, length in segments)
    weighted = sum(min(abs(angle - dominant), 90.0 - abs(angle - dominant)) * length for angle, length in segments)
    return weighted / total


class ToleranceTests(unittest.TestCase):
    def test_tolerance_scales_with_ground_sample_distance(self) -> None:
        self.assertAlmostEqual(tolerance_for_pixel_size(0.075), 0.1875)

    def test_rejects_non_positive_pixel_size(self) -> None:
        with self.assertRaises(RegularisationError):
            tolerance_for_pixel_size(0)


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_non_positive_tolerance(self) -> None:
        with self.assertRaises(RegularisationError):
            RegularisationConfig(simplify_tolerance_m=0).validate()

    def test_rejects_zero_cores(self) -> None:
        with self.assertRaises(RegularisationError):
            RegularisationConfig(num_cores=0).validate()


class RegulariseGeometriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projected = CRS.from_epsg(7855)
        self.geographic = CRS.from_epsg(4326)

    def test_straightens_staircase_edges_while_preserving_area(self) -> None:
        original = _staircase_rectangle()

        result = regularise_geometries([original], self.projected)

        self.assertEqual(len(result), 1)
        self.assertLess(_edge_angle_deviation(result[0]), _edge_angle_deviation(original))
        self.assertLess(len(result[0].exterior.coords), len(original.exterior.coords))
        self.assertAlmostEqual(result[0].area / original.area, 1.0, delta=0.1)

    def test_returns_input_unchanged_when_disabled(self) -> None:
        original = _staircase_rectangle()

        result = regularise_geometries([original], self.projected, RegularisationConfig(enabled=False))

        self.assertEqual(result[0].wkt, original.wkt)

    def test_skips_regularisation_for_geographic_crs(self) -> None:
        original = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])

        result = regularise_geometries([original], self.geographic)

        self.assertEqual(result[0].wkt, original.wkt)

    def test_preserves_positional_alignment_across_the_batch(self) -> None:
        geometries = [
            _staircase_rectangle(),
            Polygon([(20, 0), (26, 0), (26, 4), (20, 4)]),
            Polygon([(40, 0), (48, 0), (48, 5), (40, 5)]),
        ]

        result = regularise_geometries(geometries, self.projected)

        self.assertEqual(len(result), len(geometries))
        for source, regularised in zip(geometries, result):
            self.assertLess(source.centroid.distance(regularised.centroid), 1.0)

    def test_handles_empty_input(self) -> None:
        self.assertEqual(regularise_geometries([], self.projected), [])


if __name__ == "__main__":
    unittest.main()
