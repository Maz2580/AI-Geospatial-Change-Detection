"""Fuse independent WGS84 construction-change candidate sources.

Fusion is deliberately conservative: it records agreement between sources but
keeps unconfirmed single-source candidates for human review.  It does not
promote a remote detector to ground truth.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


class FusionError(ValueError):
    """Raised when a candidate input is not usable WGS84 GeoJSON."""


def _metric_geometry(geometry: dict[str, Any]) -> BaseGeometry:
    try:
        projected = transform_geom("EPSG:4326", "EPSG:3857", geometry, precision=6)
        result = shape(projected)
    except Exception as exc:  # Rasterio/Shapely expose several parse exception types.
        raise FusionError("Candidate geometry must be valid WGS84 GeoJSON.") from exc
    if result.is_empty:
        raise FusionError("Candidate geometry is empty.")
    return result


def _wgs84_geometry(geometry: BaseGeometry) -> dict[str, Any]:
    return transform_geom("EPSG:3857", "EPSG:4326", mapping(geometry), precision=8)


def _preferred_classification(classifications: Iterable[str]) -> str:
    counts = Counter(classifications)
    if not counts:
        return "construction_change_candidate"

    def key(value: str) -> tuple[int, int, str]:
        lowered = value.lower()
        construction_priority = int(any(word in lowered for word in ("building", "extension", "pool", "driveway", "pav")))
        return (counts[value], construction_priority, value)

    return max(counts, key=key)


def _feature_record(source: str, feature: dict[str, Any], ordinal: int) -> dict[str, Any] | None:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    metric = _metric_geometry(geometry)
    properties = feature.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    candidate_id = properties.get("candidate_id", ordinal)
    classification = str(properties.get("classification", "construction_change_candidate"))
    model = properties.get("requested_detector") or properties.get("model") or properties.get("candidate_source")
    return {
        "source": source,
        "candidate_id": candidate_id,
        "classification": classification,
        "model": str(model) if model else None,
        "metric_geometry": metric,
    }


def _input_records(candidate_inputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source, collection in candidate_inputs.items():
        if not source.strip():
            raise FusionError("Each candidate source needs a non-empty name.")
        if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
            raise FusionError(f"Candidate source {source!r} is not a GeoJSON FeatureCollection.")
        for ordinal, feature in enumerate(collection["features"], start=1):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                continue
            record = _feature_record(source, feature, ordinal)
            if record is not None:
                records.append(record)
    return records


def fuse_candidates(candidate_inputs: dict[str, dict[str, Any]], *, merge_distance_m: float = 2.0) -> dict[str, Any]:
    """Fuse nearby/intersecting candidates from *different* sources.

    Input GeoJSON must be WGS84. A single-source candidate is deliberately
    retained so that the reviewer can assess potential omissions by other
    models. Only spatial agreement from independent source labels forms a
    multi-source cluster.
    """
    if merge_distance_m < 0:
        raise FusionError("merge_distance_m cannot be negative.")
    records = _input_records(candidate_inputs)
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left]["source"] == records[right]["source"]:
                continue
            first, second = records[left]["metric_geometry"], records[right]["metric_geometry"]
            if first.intersects(second) or first.distance(second) <= merge_distance_m:
                join(left, right)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        clusters.setdefault(find(index), []).append(record)

    features: list[dict[str, Any]] = []
    for candidate_id, members in enumerate(clusters.values(), start=1):
        footprints = [m["metric_geometry"] for m in members if m.get("classification") == "building_footprint"]
        merged = unary_union(footprints) if footprints else unary_union([m["metric_geometry"] for m in members])
        sources = sorted({member["source"] for member in members})
        classifications = [member["classification"] for member in members]
        source_models = sorted({member["model"] for member in members if member["model"]})
        features.append(
            {
                "type": "Feature",
                "geometry": _wgs84_geometry(merged),
                "properties": {
                    "candidate_id": candidate_id,
                    "classification": _preferred_classification(classifications),
                    "area_m2": round(float(merged.area), 1),
                    "candidate_sources": sources,
                    "source_count": len(sources),
                    "raw_candidate_count": len(members),
                    "source_models": source_models,
                    "source_candidate_ids": [f"{member['source']}:{member['candidate_id']}" for member in members],
                    "source_classifications": sorted(set(classifications)),
                    "agreement": "multi_source_agreement" if len(sources) > 1 else "single_source_candidate",
                },
            }
        )
    features.sort(key=lambda feature: (-feature["properties"]["source_count"], -feature["properties"]["area_m2"]))
    for candidate_id, feature in enumerate(features, start=1):
        feature["properties"]["candidate_id"] = candidate_id
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "crs": "EPSG:4326",
            "merge_distance_m": merge_distance_m,
            "input_sources": sorted(candidate_inputs),
            "single_source_candidates_are_retained": True,
        },
    }


def load_candidate_input(value: str) -> tuple[str, dict[str, Any]]:
    """Parse one ``source=path`` CLI value and read its FeatureCollection."""
    if "=" not in value:
        raise FusionError("Each --candidate value must use source=path.geojson.")
    source, path_text = value.split("=", 1)
    path = Path(path_text)
    if not source.strip() or not path.is_file():
        raise FusionError(f"Candidate input is invalid: {value!r}")
    try:
        collection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FusionError(f"Could not read candidate GeoJSON: {path}") from exc
    if not isinstance(collection, dict):
        raise FusionError(f"Candidate input {path} is not a JSON object.")
    return source.strip(), collection


def write_fusion_outputs(output_dir: str | Path, candidate_inputs: dict[str, dict[str, Any]], *, merge_distance_m: float = 2.0) -> dict[str, Any]:
    """Fuse candidates and write GeoJSON plus an auditable source-count report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    collection = fuse_candidates(candidate_inputs, merge_distance_m=merge_distance_m)
    geojson_path = directory / "fused_candidates.geojson"
    report_path = directory / "fusion_report.json"
    geojson_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    source_counts = Counter(feature["properties"]["source_count"] for feature in collection["features"])
    report = {
        "input_sources": collection["metadata"]["input_sources"],
        "merge_distance_m": merge_distance_m,
        "fused_candidate_count": len(collection["features"]),
        "by_source_agreement": {str(count): amount for count, amount in sorted(source_counts.items())},
        "outputs": {"fused_candidates": str(geojson_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "outputs": {**report["outputs"], "report": str(report_path)}}
