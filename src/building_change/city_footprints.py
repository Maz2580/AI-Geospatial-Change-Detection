"""Dated City of Melbourne building footprints for benchmark construction.

The City publishes separate snapshot datasets, which are useful for assembling
an Australian test set.  They are *reference evidence*, not automatic truth:
the polygons map structure walls while aerial imagery often sees roof edges,
and snapshot differences can include mapping improvements as well as building
work.  Every inferred change still needs image review.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

from .footprints import write_comparison_outputs


CITY_FOOTPRINTS_BASE_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
CITY_FOOTPRINTS_LICENSE = "CC BY 4.0"


class CityFootprintError(ValueError):
    """Raised when an official footprint snapshot cannot be obtained safely."""


class HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...

    def raise_for_status(self) -> None: ...


class HttpSession(Protocol):
    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> HttpResponse: ...


@dataclass(frozen=True)
class FootprintBoundingBox:
    """A WGS84 rectangle in normal west/south/east/north order."""

    west: float
    south: float
    east: float
    north: float

    def validate(self) -> None:
        if not -180 <= self.west < self.east <= 180 or not -90 <= self.south < self.north <= 90:
            raise CityFootprintError("The City footprint bounding box must be valid WGS84 west/south/east/north coordinates.")

    def ods_where_clause(self) -> str:
        """Return the documented Opendatasoft bbox filter for its Geo Point field."""
        self.validate()
        return f"in_bbox(geo_point_2d, {self.south}, {self.west}, {self.north}, {self.east})"


def _request_session(session: HttpSession | None) -> HttpSession:
    if session is not None:
        return session
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a core dependency
        raise CityFootprintError("The core requests dependency is required to download City footprint snapshots.") from exc
    return requests.Session()


def _geometry_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("geo_shape")
    if not isinstance(value, dict):
        return None
    geometry = value.get("geometry") if value.get("type") == "Feature" else value
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        return None
    if not isinstance(geometry.get("coordinates"), list):
        return None
    return geometry


def fetch_city_melbourne_footprints(
    snapshot_year: int,
    bbox: FootprintBoundingBox,
    *,
    session: HttpSession | None = None,
    page_size: int = 100,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Fetch all valid official footprint polygons that intersect a small AOI.

    The public Explore API caps one request at 100 records. Its spatial filter
    uses each record's representative point, so a footprint may extend slightly
    beyond the requested rectangle. Keeping it avoids slicing a building in
    half; the imagery AOI should include a small buffer around the reference
    rectangle.
    """
    if not 2010 <= snapshot_year <= 2100:
        raise CityFootprintError("Snapshot year must be a four-digit year.")
    bbox.validate()
    if not 1 <= page_size <= 100 or timeout_seconds <= 0:
        raise CityFootprintError("City API page size must be 1--100 and timeout must be positive.")
    active_session = _request_session(session)
    url = f"{CITY_FOOTPRINTS_BASE_URL}/{snapshot_year}-building-footprints/records"
    where = bbox.ods_where_clause()
    offset = 0
    total_count: int | None = None
    features: list[dict[str, Any]] = []
    try:
        while total_count is None or offset < total_count:
            response = active_session.get(
                url,
                params={"limit": page_size, "offset": offset, "where": where},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise CityFootprintError("City footprint API returned an unexpected response.")
            count = payload.get("total_count")
            if not isinstance(count, int) or count < 0:
                raise CityFootprintError("City footprint API response has no valid total_count.")
            total_count = count
            records = payload["results"]
            for record in records:
                if not isinstance(record, dict):
                    continue
                geometry = _geometry_from_record(record)
                if geometry is None:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            "candidate_id": len(features) + 1,
                            "snapshot_year": snapshot_year,
                            "source": "City of Melbourne Building Footprints",
                            "evidence_role": "dated_reference_footprint",
                        },
                    }
                )
            offset += len(records)
            if not records and offset < total_count:
                raise CityFootprintError("City footprint API returned an empty page before its declared total.")
    except CityFootprintError:
        raise
    except Exception as exc:  # pragma: no cover - network/provider dependent
        raise CityFootprintError(f"Could not fetch City of Melbourne {snapshot_year} building footprints: {exc}") from exc
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "crs": "EPSG:4326",
            "requested_detector": f"city_melbourne_{snapshot_year}_reference",
            "source": "City of Melbourne Building Footprints",
            "source_dataset": f"{snapshot_year}-building-footprints",
            "source_url": url,
            "license": CITY_FOOTPRINTS_LICENSE,
            "query_bbox_wgs84": {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north},
            "evidence_role": "dated_reference_footprint",
            "warning": "Snapshot changes require imagery review; footprint updates can reflect mapping changes and map wall footprints rather than roof edges.",
        },
    }


def write_city_footprint_benchmark(
    output_dir: str | Path,
    *,
    before_year: int,
    after_year: int,
    bbox: FootprintBoundingBox,
    session: HttpSession | None = None,
    match_distance_m: float = 6.0,
    match_iou: float = 0.10,
    extension_outside_fraction: float = 0.25,
    min_area_m2: float = 20.0,
) -> dict[str, Any]:
    """Download two snapshots and write reference changes for human validation."""
    if before_year >= after_year:
        raise CityFootprintError("The benchmark before year must be earlier than the after year.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    before = fetch_city_melbourne_footprints(before_year, bbox, session=session)
    after = fetch_city_melbourne_footprints(after_year, bbox, session=session)
    before_path = directory / f"city_melbourne_{before_year}_building_footprints.geojson"
    after_path = directory / f"city_melbourne_{after_year}_building_footprints.geojson"
    before_path.write_text(json.dumps(before, indent=2), encoding="utf-8")
    after_path.write_text(json.dumps(after, indent=2), encoding="utf-8")
    comparison = write_comparison_outputs(
        directory,
        before,
        after,
        match_distance_m=match_distance_m,
        match_iou=match_iou,
        extension_outside_fraction=extension_outside_fraction,
        min_area_m2=min_area_m2,
    )
    report = {
        "reference_source": "City of Melbourne Building Footprints",
        "reference_license": CITY_FOOTPRINTS_LICENSE,
        "before_year": before_year,
        "after_year": after_year,
        "before_footprint_count": len(before["features"]),
        "after_footprint_count": len(after["features"]),
        "bbox_wgs84": {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north},
        "comparison": comparison,
        "outputs": {
            "before_reference_footprints": str(before_path),
            "after_reference_footprints": str(after_path),
            **comparison["outputs"],
        },
        "warning": "This is a labelled benchmark candidate set, not automatic construction ground truth. Review each reference difference against the matching aerial dates.",
    }
    (directory / "reference_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
