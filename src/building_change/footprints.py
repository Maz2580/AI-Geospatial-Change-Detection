"""Compare WGS84 building-footprint proposals from two imagery dates.

This is an object-evidence stage. It must not be counted as an independent
vote beside the change model that proposed the same location.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geodesy import from_metric, metric_crs_for, to_metric


class FootprintComparisonError(ValueError):
    """Raised when footprint inputs cannot be compared safely."""


def _metric_geometry(geometry: dict[str, Any], crs: Any) -> BaseGeometry:
    """Project one WGS84 footprint into true ground metres.

    Every footprint in a comparison must share ``crs``, or distances between
    them are meaningless.
    """
    try:
        result, _ = to_metric(shape(geometry), crs)
    except Exception as exc:
        raise FootprintComparisonError("Footprint geometry must be valid WGS84 GeoJSON.") from exc
    if result.is_empty or result.area <= 0:
        raise FootprintComparisonError("Footprint geometry is empty or has no area.")
    return result


def _wgs84_geometry(geometry: BaseGeometry, crs: Any) -> dict[str, Any]:
    return from_metric(geometry, crs)


def _comparison_crs(*collections: dict[str, Any]) -> Any:
    """Choose one ground-metre CRS for a whole comparison.

    Taken from the first usable geometry rather than per feature, so that two
    footprints either side of a UTM zone boundary are still measured against
    each other in one plane.
    """
    for collection in collections:
        for feature in collection.get("features", []) if isinstance(collection.get("features"), list) else []:
            if isinstance(feature, dict) and isinstance(feature.get("geometry"), dict):
                try:
                    return metric_crs_for(feature["geometry"])
                except Exception:  # noqa: BLE001 - try the next candidate geometry
                    continue
    raise FootprintComparisonError("No usable footprint geometry to establish a measurement CRS.")


def _records(collection: dict[str, Any], crs: Any, *, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                "geometry": _metric_geometry(feature["geometry"], crs),
            }
        )
    metadata = collection.get("metadata") if isinstance(collection.get("metadata"), dict) else {}
    return records, metadata


def substantial_change(
    geometry: BaseGeometry,
    *,
    min_change_width_m: float,
    min_change_area_m2: float,
) -> BaseGeometry:
    """Drop change fragments too thin or too small to be construction.

    Subtracting one date's footprint from another's leaves two very different
    kinds of shape. A real extension is compact and attached to one side of the
    building. Disagreement between two independent segmentations of the *same*
    unchanged roof is a thin ribbon around the perimeter plus scattered islands,
    and on the Murchison holdout that ribbon was being reported as a 112 m2
    extension of a house nobody had touched.

    Eroding by half the minimum width separates them: a ribbon a few pixels wide
    disappears, a real wing survives. Parts that survive are returned whole, so
    the reported edge is still the measured edge rather than a rounded one.
    """
    if min_change_width_m < 0 or min_change_area_m2 < 0:
        raise FootprintComparisonError("Change size limits cannot be negative.")
    if geometry.is_empty or min_change_width_m == 0:
        return geometry
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    kept = [
        part
        for part in parts
        if part.area >= min_change_area_m2
        and not part.buffer(-min_change_width_m / 2.0, join_style="mitre").is_empty
    ]
    if not kept:
        return geometry.intersection(geometry.buffer(-geometry.length))  # an empty geometry of the same type
    result = unary_union(kept)
    return result if result.is_valid else result.buffer(0)


def compare_footprints(
    before_collection: dict[str, Any],
    after_collection: dict[str, Any],
    *,
    match_distance_m: float = 6.0,
    match_iou: float = 0.10,
    extension_outside_fraction: float = 0.25,
    min_area_m2: float = 20.0,
    # A wing narrower than this is segmentation disagreement, not construction.
    min_change_width_m: float = 1.5,
    # Matches the gold set's "anything with a roof, about 10 m2 and up".
    min_change_area_m2: float = 10.0,
) -> dict[str, Any]:
    """Return new and extension candidates from before/after footprint proposals.

    Each candidate's geometry is **what changed**, not the object containing it.
    An extension is the new wing, not the whole house it was added to. The
    containing footprint travels alongside under ``_object_geometry`` and is
    written to its own layer by ``write_comparison_outputs``.

    ``match_distance_m`` absorbs small segmentation/registration differences.
    A nearby after footprint is only labelled as an extension when a material
    fraction lies outside its matched before footprint(s). All results require
    review: footprint proposals can merge neighbouring buildings or omit roofs.
    """
    if match_distance_m < 0 or not 0 <= match_iou <= 1 or not 0 <= extension_outside_fraction <= 1 or min_area_m2 <= 0:
        raise FootprintComparisonError("Footprint comparison parameters are outside their valid ranges.")
    crs = _comparison_crs(after_collection, before_collection)
    before, before_metadata = _records(before_collection, crs, name="before footprints")
    after, after_metadata = _records(after_collection, crs, name="after footprints")
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
        # The changed part: what is present on the after date and was not there
        # before. For a new building that is the whole footprint; for an
        # extension it is only the new wing.
        changed_geometry = geometry
        overlap_fraction = 0.0
        sliver_area = 0.0
        classification: str | None
        if not matched:
            classification = "new_building_footprint_candidate"
        else:
            matched_before = unary_union([value["record"]["geometry"] for value in matched])
            changed_geometry = geometry.difference(matched_before)
            if not changed_geometry.is_valid:
                changed_geometry = changed_geometry.buffer(0)
            raw_change_area = changed_geometry.area
            changed_geometry = substantial_change(
                changed_geometry,
                min_change_width_m=min_change_width_m,
                min_change_area_m2=min_change_area_m2,
            )
            sliver_area = raw_change_area - changed_geometry.area
            overlap_fraction = max(0.0, min(1.0, 1.0 - changed_geometry.area / geometry.area))
            classification = (
                "building_extension_footprint_candidate"
                if changed_geometry.area / geometry.area >= extension_outside_fraction
                else None
            )
        if classification is None or changed_geometry.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                # The candidate's geometry is the change, not the object that
                # contains it. Reporting a 450 m2 house as an "extension" made
                # every extension candidate overstate what was built.
                "geometry": _wgs84_geometry(changed_geometry, crs),
                "properties": {
                    "candidate_id": len(features) + 1,
                    "classification": classification,
                    # area_m2 always describes this feature's own geometry.
                    "area_m2": round(float(changed_geometry.area), 1),
                    "changed_area_m2": round(float(changed_geometry.area), 1),
                    "object_area_m2": round(float(geometry.area), 1),
                    # Retained under its original name for existing consumers.
                    "outside_before_area_m2": round(float(changed_geometry.area), 1),
                    "before_overlap_fraction": round(float(overlap_fraction), 3),
                    # How much of the raw difference was too thin to be building.
                    # A large value here means the two dates disagree about the
                    # roof edge, not that something was built.
                    "discarded_sliver_area_m2": round(float(sliver_area), 1),
                    "nearest_before_footprint_distance_m": round(float(nearest_distance), 1) if nearest_distance is not None else None,
                    "best_before_footprint_iou": round(float(best_iou), 3),
                    "matched_before_footprint_ids": [value["record"]["candidate_id"] for value in matched],
                    "evidence_role": "object_footprint_refinement",
                    "footprint_detector": after_metadata.get("requested_detector", "unknown"),
                },
                # The containing object, carried so it can be written to its own
                # layer. Reviewers need the whole roof for context even though
                # the candidate is only the changed part.
                "_object_geometry": _wgs84_geometry(geometry, crs),
            }
        )
    features.sort(key=lambda feature: (-feature["properties"]["object_area_m2"], feature["properties"]["candidate_id"]))
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
            # Recorded because it is passed by the caller, not taken from the
            # signature default: a report without it cannot be reproduced.
            "min_area_m2": min_area_m2,
            "min_change_width_m": min_change_width_m,
            "min_change_area_m2": min_change_area_m2,
            "measurement_crs": str(crs),
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
    """Write review candidates, their containing objects, and an audit report.

    Two layers, sharing ``candidate_id``:

    ``footprint_change_candidates.geojson``
        What changed. For an extension this is the new wing only.
    ``footprint_change_objects.geojson``
        The whole roof each change sits on, for review context.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    collection = compare_footprints(before_collection, after_collection, **options)

    object_features: list[dict[str, Any]] = []
    for feature in collection["features"]:
        object_geometry = feature.pop("_object_geometry", None)
        if object_geometry is None:
            continue
        object_features.append(
            {
                "type": "Feature",
                "geometry": object_geometry,
                "properties": {
                    "candidate_id": feature["properties"]["candidate_id"],
                    "classification": "containing_building_footprint",
                    "area_m2": feature["properties"]["object_area_m2"],
                    "changed_area_m2": feature["properties"]["changed_area_m2"],
                    "evidence_role": "review_context_for_the_matching_candidate",
                },
            }
        )

    candidates_path = directory / "footprint_change_candidates.geojson"
    objects_path = directory / "footprint_change_objects.geojson"
    report_path = directory / "footprint_comparison_report.json"
    candidates_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    objects_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": object_features,
                "metadata": {**collection["metadata"], "evidence_role": "review_context"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    classes = {"new_building_footprint_candidate": 0, "building_extension_footprint_candidate": 0}
    for feature in collection["features"]:
        classes[feature["properties"]["classification"]] += 1
    changed_total = sum(feature["properties"]["changed_area_m2"] for feature in collection["features"])
    object_total = sum(feature["properties"]["object_area_m2"] for feature in collection["features"])
    report = {
        "candidate_count": len(collection["features"]),
        "class_counts": classes,
        "changed_area_m2_total": round(changed_total, 1),
        "containing_object_area_m2_total": round(object_total, 1),
        "metadata": collection["metadata"],
        "outputs": {
            "footprint_candidates": str(candidates_path),
            "footprint_change_objects": str(objects_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "outputs": {**report["outputs"], "report": str(report_path)}}
