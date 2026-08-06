"""Compare WGS84 building-footprint proposals from two imagery dates.

This is an object-evidence stage. It must not be counted as an independent
vote beside the change model that proposed the same location.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


class FootprintComparisonError(ValueError):
    """Raised when footprint inputs cannot be compared safely."""


def _metric_geometry(geometry: dict[str, Any]) -> BaseGeometry:
    try:
        result = shape(transform_geom("EPSG:4326", "EPSG:3857", geometry, precision=6))
    except Exception as exc:
        raise FootprintComparisonError("Footprint geometry must be valid WGS84 GeoJSON.") from exc
    if not result.is_valid:
        result = result.buffer(0)
    if result.is_empty or result.area <= 0:
        raise FootprintComparisonError("Footprint geometry is empty or has no area.")
    return result


def _wgs84_geometry(geometry: BaseGeometry) -> dict[str, Any]:
    return transform_geom("EPSG:3857", "EPSG:4326", mapping(geometry), precision=8)


def _records(collection: dict[str, Any], *, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
        raise FootprintComparisonError(f"{name} is not a GeoJSON FeatureCollection.")
    records: list[dict[str, Any]] = []
    for ordinal, feature in enumerate(collection["features"], start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature" or not isinstance(feature.get("geometry"), dict):
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        records.append(
            {
                "candidate_id": properties.get("candidate_id", ordinal),
                "geometry": _metric_geometry(feature["geometry"]),
            }
        )
    metadata = collection.get("metadata") if isinstance(collection.get("metadata"), dict) else {}
    return records, metadata


def compare_footprints(
    before_collection: dict[str, Any],
    after_collection: dict[str, Any],
    *,
    match_distance_m: float = 6.0,
    match_iou: float = 0.10,
    extension_outside_fraction: float = 0.25,
    min_area_m2: float = 20.0,
) -> dict[str, Any]:
    """Return new and extension candidates from before/after footprint proposals.

    ``match_distance_m`` absorbs small segmentation/registration differences.
    A nearby after footprint is only labelled as an extension when a material
    fraction lies outside its matched before footprint(s). All results require
    review: footprint proposals can merge neighbouring buildings or omit roofs.
    """
    if match_distance_m < 0 or not 0 <= match_iou <= 1 or not 0 <= extension_outside_fraction <= 1 or min_area_m2 <= 0:
        raise FootprintComparisonError("Footprint comparison parameters are outside their valid ranges.")
    before, before_metadata = _records(before_collection, name="before footprints")
    after, after_metadata = _records(after_collection, name="after footprints")
    features: list[dict[str, Any]] = []
    for after_record in after:
        geometry = after_record["geometry"]
        if geometry.area < min_area_m2:
            continue
        comparisons = []
        for before_record in before:
            before_geometry = before_record["geometry"]
            intersection_area = geometry.intersection(before_geometry).area
            union_area = geometry.union(before_geometry).area
            comparisons.append(
                {
                    "record": before_record,
                    "distance": geometry.distance(before_geometry),
                    "iou": 0.0 if union_area <= 0 else intersection_area / union_area,
                }
            )
        matched = [value for value in comparisons if value["distance"] <= match_distance_m or value["iou"] >= match_iou]
        nearest_distance = min((value["distance"] for value in comparisons), default=None)
        best_iou = max((value["iou"] for value in comparisons), default=0.0)
        outside_area = geometry.area
        overlap_fraction = 0.0
        classification: str | None
        if not matched:
            classification = "new_building_footprint_candidate"
        else:
            matched_before = unary_union([value["record"]["geometry"] for value in matched])
            outside_area = geometry.difference(matched_before).area
            overlap_fraction = max(0.0, min(1.0, 1.0 - outside_area / geometry.area))
            classification = "building_extension_footprint_candidate" if outside_area / geometry.area >= extension_outside_fraction else None
        if classification is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": _wgs84_geometry(geometry),
                "properties": {
                    "candidate_id": len(features) + 1,
                    "classification": classification,
                    "area_m2": round(float(geometry.area), 1),
                    "outside_before_area_m2": round(float(outside_area), 1),
                    "before_overlap_fraction": round(float(overlap_fraction), 3),
                    "nearest_before_footprint_distance_m": round(float(nearest_distance), 1) if nearest_distance is not None else None,
                    "best_before_footprint_iou": round(float(best_iou), 3),
                    "matched_before_footprint_ids": [value["record"]["candidate_id"] for value in matched],
                    "evidence_role": "object_footprint_refinement",
                    "footprint_detector": after_metadata.get("requested_detector", "unknown"),
                },
            }
        )
    features.sort(key=lambda feature: (-feature["properties"]["area_m2"], feature["properties"]["candidate_id"]))
    for index, feature in enumerate(features, start=1):
        feature["properties"]["candidate_id"] = index
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "crs": "EPSG:4326",
            "evidence_role": "object_footprint_refinement",
            "before_footprint_count": len(before),
            "after_footprint_count": len(after),
            "match_distance_m": match_distance_m,
            "match_iou": match_iou,
            "extension_outside_fraction": extension_outside_fraction,
            "before_detector": before_metadata.get("requested_detector", "unknown"),
            "after_detector": after_metadata.get("requested_detector", "unknown"),
        },
    }


def load_feature_collection(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        collection = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FootprintComparisonError(f"Could not read footprint GeoJSON: {source}") from exc
    if not isinstance(collection, dict):
        raise FootprintComparisonError(f"Footprint input {source} is not a JSON object.")
    return collection


def write_comparison_outputs(
    output_dir: str | Path,
    before_collection: dict[str, Any],
    after_collection: dict[str, Any],
    **options: Any,
) -> dict[str, Any]:
    """Write review candidates and an auditable footprint-comparison report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    collection = compare_footprints(before_collection, after_collection, **options)
    candidates_path = directory / "footprint_change_candidates.geojson"
    report_path = directory / "footprint_comparison_report.json"
    candidates_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    classes = {"new_building_footprint_candidate": 0, "building_extension_footprint_candidate": 0}
    for feature in collection["features"]:
        classes[feature["properties"]["classification"]] += 1
    report = {
        "candidate_count": len(collection["features"]),
        "class_counts": classes,
        "metadata": collection["metadata"],
        "outputs": {"footprint_candidates": str(candidates_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "outputs": {**report["outputs"], "report": str(report_path)}}
