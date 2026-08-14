"""Create construction-site review candidates from change and footprint evidence.

The result intentionally represents a site, not a confirmed building. A site
can contain a building footprint, a driveway, a pool, vegetation change, or a
combination of these objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geodesy import from_metric, metric_crs_for, to_metric


class SiteGroupingError(ValueError):
    """Raised when source candidates cannot be grouped into WGS84 sites."""


def _metric_geometry(geometry: dict[str, Any], crs: Any) -> BaseGeometry:
    """Project into true ground metres so the grouping radii mean metres."""
    try:
        result, _ = to_metric(shape(geometry), crs)
    except Exception as exc:
        raise SiteGroupingError("Candidate geometry must be valid WGS84 GeoJSON.") from exc
    if result.is_empty:
        raise SiteGroupingError("Candidate geometry is empty.")
    return result


def _grouping_crs(*collections: dict[str, Any]) -> Any:
    """Choose one ground-metre CRS for every collection being grouped."""
    for collection in collections:
        features = collection.get("features")
        for feature in features if isinstance(features, list) else []:
            if isinstance(feature, dict) and isinstance(feature.get("geometry"), dict):
                try:
                    return metric_crs_for(feature["geometry"])
                except Exception:  # noqa: BLE001 - try the next candidate geometry
                    continue
    raise SiteGroupingError("No usable candidate geometry to establish a measurement CRS.")


def _surface_union(geometries: list[BaseGeometry]) -> BaseGeometry:
    surfaces: list[BaseGeometry] = []
    for geometry in geometries:
        if isinstance(geometry, (Polygon, MultiPolygon)):
            surfaces.append(geometry)
        elif isinstance(geometry, GeometryCollection):
            surfaces.extend(part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon)))
    result = unary_union(surfaces)
    if result.is_empty or not isinstance(result, (Polygon, MultiPolygon)):
        raise SiteGroupingError("Site has no polygonal geometry for review.")
    return result


def _wgs84_geometry(geometry: BaseGeometry, crs: Any) -> dict[str, Any]:
    return from_metric(geometry, crs)


def _records(collection: dict[str, Any], crs: Any, *, role: str) -> list[dict[str, Any]]:
    if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
        raise SiteGroupingError(f"{role} input is not a GeoJSON FeatureCollection.")
    records: list[dict[str, Any]] = []
    for ordinal, feature in enumerate(collection["features"], start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature" or not isinstance(feature.get("geometry"), dict):
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        records.append(
            {
                "role": role,
                "candidate_id": properties.get("candidate_id", ordinal),
                "geometry": _metric_geometry(feature["geometry"], crs),
                "properties": properties,
            }
        )
    return records


def _change_clusters(changes: list[dict[str, Any]], merge_distance_m: float) -> list[list[dict[str, Any]]]:
    parents = list(range(len(changes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        first, second = find(left), find(right)
        if first != second:
            parents[second] = first

    for left in range(len(changes)):
        for right in range(left + 1, len(changes)):
            first, second = changes[left]["geometry"], changes[right]["geometry"]
            if first.intersects(second) or first.distance(second) <= merge_distance_m:
                join(left, right)
    clusters: dict[int, list[dict[str, Any]]] = {}
    for index, change in enumerate(changes):
        clusters.setdefault(find(index), []).append(change)
    return list(clusters.values())


def _source_names(change_records: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for record in change_records:
        value = record["properties"].get("candidate_sources", [])
        if isinstance(value, list):
            names.update(str(item) for item in value)
        elif value:
            names.add(str(value))
    return sorted(names)


def _priority(changes: list[dict[str, Any]], footprints: list[dict[str, Any]]) -> str:
    agreement = max((int(record["properties"].get("source_count", 1)) for record in changes), default=0)
    if agreement >= 3 and footprints:
        return "high"
    if agreement >= 2 or footprints:
        return "medium"
    return "low"


def _site_feature(
    candidate_id: int,
    changes: list[dict[str, Any]],
    footprints: list[dict[str, Any]],
    crs: Any,
) -> dict[str, Any]:
    geometry = _surface_union([record["geometry"] for record in [*changes, *footprints]])
    change_classifications = sorted({str(record["properties"].get("classification", "change_candidate")) for record in changes})
    footprint_classifications = sorted({str(record["properties"].get("classification", "footprint_candidate")) for record in footprints})
    classification = "construction_site_candidate" if changes else "footprint_only_site_candidate"
    return {
        "type": "Feature",
        "geometry": _wgs84_geometry(geometry, crs),
        "properties": {
            "candidate_id": candidate_id,
            "classification": classification,
            "site_area_m2": round(float(geometry.area), 1),
            "review_priority": _priority(changes, footprints),
            "change_evidence_count": len(changes),
            "footprint_evidence_count": len(footprints),
            "change_candidate_ids": [record["candidate_id"] for record in changes],
            "footprint_candidate_ids": [record["candidate_id"] for record in footprints],
            "change_classifications": change_classifications,
            "footprint_classifications": footprint_classifications,
            "change_sources": _source_names(changes),
            "evidence_role": "site_context_for_manual_object_classification",
            "manual_review_question": "Which permanent objects changed here: building, extension, pool, hardscape, garden, or none?",
        },
    }


def build_sites(
    change_collection: dict[str, Any],
    footprint_collection: dict[str, Any],
    *,
    anchor_merge_distance_m: float = 10.0,
    site_radius_m: float = 25.0,
) -> dict[str, Any]:
    """Group nearby evidence into reviewable construction sites.

    Footprints are attached to the nearest change cluster within
    ``site_radius_m``. This avoids treating a footprint comparison as another
    independent model vote and preserves footprint-only cases for review.
    """
    if anchor_merge_distance_m < 0 or site_radius_m < 0:
        raise SiteGroupingError("Site grouping distances cannot be negative.")
    crs = _grouping_crs(change_collection, footprint_collection)
    changes = _records(change_collection, crs, role="change")
    footprints = _records(footprint_collection, crs, role="footprint")
    clusters = _change_clusters(changes, anchor_merge_distance_m)
    cluster_geometries = [_surface_union([record["geometry"] for record in cluster]) for cluster in clusters]
    attached: list[list[dict[str, Any]]] = [[] for _ in clusters]
    unassigned: list[dict[str, Any]] = []
    for footprint in footprints:
        distances = [footprint["geometry"].distance(geometry) for geometry in cluster_geometries]
        if distances and min(distances) <= site_radius_m:
            attached[distances.index(min(distances))].append(footprint)
        else:
            unassigned.append(footprint)
    features = [_site_feature(index, cluster, attached[index - 1], crs) for index, cluster in enumerate(clusters, start=1)]
    features.extend(
        _site_feature(len(features) + index, [], [footprint], crs) for index, footprint in enumerate(unassigned, start=1)
    )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    features.sort(key=lambda feature: (priority_order[feature["properties"]["review_priority"]], -feature["properties"]["site_area_m2"]))
    for candidate_id, feature in enumerate(features, start=1):
        feature["properties"]["candidate_id"] = candidate_id
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "crs": "EPSG:4326",
            "anchor_merge_distance_m": anchor_merge_distance_m,
            "site_radius_m": site_radius_m,
            "change_candidate_count": len(changes),
            "footprint_candidate_count": len(footprints),
            "site_count": len(features),
        },
    }


def load_feature_collection(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        collection = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteGroupingError(f"Could not read candidate GeoJSON: {source}") from exc
    if not isinstance(collection, dict):
        raise SiteGroupingError(f"Candidate input {source} is not a JSON object.")
    return collection


def write_site_outputs(
    output_dir: str | Path,
    change_collection: dict[str, Any],
    footprint_collection: dict[str, Any],
    **options: Any,
) -> dict[str, Any]:
    """Write site candidates and their provenance report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    collection = build_sites(change_collection, footprint_collection, **options)
    sites_path = directory / "construction_site_candidates.geojson"
    report_path = directory / "construction_site_report.json"
    sites_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    priority_counts = {priority: 0 for priority in ("high", "medium", "low")}
    for feature in collection["features"]:
        priority_counts[feature["properties"]["review_priority"]] += 1
    report = {
        "site_count": len(collection["features"]),
        "priority_counts": priority_counts,
        "metadata": collection["metadata"],
        "outputs": {"site_candidates": str(sites_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "outputs": {**report["outputs"], "report": str(report_path)}}
