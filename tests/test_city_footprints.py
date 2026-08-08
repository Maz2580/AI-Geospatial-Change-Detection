from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from building_change.city_footprints import (
    FootprintBoundingBox,
    fetch_city_melbourne_footprints,
    write_city_footprint_benchmark,
)


def _record(west: float, south: float, east: float, north: float) -> dict:
    return {
        "geo_shape": {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
            },
        }
    }


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, pages: dict[tuple[str, int], dict]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, *, params: dict, timeout: float) -> _Response:
        del timeout
        self.calls.append((url, params))
        year = "2020" if "/2020-" in url else "2023"
        return _Response(self.pages[(year, int(params["offset"]))])


class CityFootprintsTests(unittest.TestCase):
    def test_bbox_uses_documented_lat_lon_filter_order(self) -> None:
        bbox = FootprintBoundingBox(144.951, -37.811, 144.953, -37.809)
        self.assertEqual(bbox.ods_where_clause(), "in_bbox(geo_point_2d, -37.811, 144.951, -37.809, 144.953)")

    def test_fetches_multiple_pages_and_preserves_reference_provenance(self) -> None:
        session = _Session(
            {
                ("2020", 0): {"total_count": 2, "results": [_record(144.951, -37.811, 144.9512, -37.8108)]},
                ("2020", 1): {"total_count": 2, "results": [_record(144.952, -37.811, 144.9522, -37.8108)]},
            }
        )
        collection = fetch_city_melbourne_footprints(
            2020, FootprintBoundingBox(144.95, -37.82, 144.96, -37.80), session=session, page_size=1
        )
        self.assertEqual(len(collection["features"]), 2)
        self.assertEqual(collection["metadata"]["license"], "CC BY 4.0")
        self.assertEqual([call[1]["offset"] for call in session.calls], [0, 1])

    def test_writes_dated_reference_and_comparison_outputs(self) -> None:
        pages = {
            ("2020", 0): {"total_count": 1, "results": [_record(144.951, -37.811, 144.9512, -37.8108)]},
            ("2023", 0): {"total_count": 2, "results": [_record(144.951, -37.811, 144.9512, -37.8108), _record(144.952, -37.811, 144.9522, -37.8108)]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = write_city_footprint_benchmark(
                temporary,
                before_year=2020,
                after_year=2023,
                bbox=FootprintBoundingBox(144.95, -37.82, 144.96, -37.80),
                session=_Session(pages),
                min_area_m2=20,
            )
            self.assertEqual(report["before_footprint_count"], 1)
            self.assertEqual(report["after_footprint_count"], 2)
            self.assertTrue(Path(report["outputs"]["footprint_candidates"]).is_file())


if __name__ == "__main__":
    unittest.main()
