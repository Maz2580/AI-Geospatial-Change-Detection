"""Persistent identity for a change candidate across surveys.

``candidate_id`` is assigned by descending area on every run, so candidate 7 in
one survey is not candidate 7 in the next. That is fine for reading a single
report and useless for the thing this tool is meant to grow into: following an
unpermitted structure across four surveys and showing when it appeared.

Two separate ideas are needed, and conflating them is the trap:

``location_key``
    A deterministic, readable label derived from where the candidate is. Good
    for citing a candidate in an email or sorting a list. **Not** a join key --
    two observations of one building can round to different cells, and two
    genuinely different structures on one lot can round to the same one.

``assign_identities``
    The actual cross-survey link, by geometry. A candidate inherits the
    ``site_id`` of the observation it overlaps in the previous survey, so the id
    survives the outline changing shape between dates -- which it will, because
    the outline is remeasured from each image independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from .geodesy import GeodesyError, metric_crs_for, to_metric

# One metre. Finer implies a precision the outline does not have; coarser starts
# merging neighbouring structures into one label.
LOCATION_KEY_PRECISION_M = 1.0


class IdentityError(ValueError):
    """Raised when candidate identity cannot be established."""


@dataclass(frozen=True)
class LinkConfig:
    """How an observation is recognised as the same site seen again."""

    # Generous, because the outline is remeasured from each image: the same roof
    # can shift by a metre between dates without anything being built.
    max_centroid_distance_m: float = 8.0
    # Or overlapping at all, which catches a building that grew a wing and moved
    # its centroid further than the distance allows.
    min_overlap_fraction: float = 0.10

    def validate(self) -> None:
        if self.max_centroid_distance_m < 0:
            raise IdentityError("max_centroid_distance_m cannot be negative.")
        if not 0 <= self.min_overlap_fraction <= 1:
            raise IdentityError("min_overlap_fraction must be between zero and one.")


def location_key(geometry: BaseGeometry | dict[str, Any], *, precision_m: float = LOCATION_KEY_PRECISION_M) -> str:
    """Return a deterministic, readable label for where a candidate sits.

    Format is ``UTM<zone><hemisphere>-<easting>-<northing>`` in whole metres, for
    example ``UTM55S-0334521-5813442``. Readable, sortable, and stable for a
    fixed location -- but a label, not a join key. Use ``assign_identities`` to
    decide whether two observations are the same site.
    """
    if precision_m <= 0:
        raise IdentityError("precision_m must be positive.")
    resolved = shape(geometry) if isinstance(geometry, dict) else geometry
    if resolved.is_empty:
        raise IdentityError("Cannot key an empty geometry.")
    try:
        crs = metric_crs_for(resolved)
        projected, _ = to_metric(resolved, crs)
    except GeodesyError as exc:
        # A degenerate polygon -- zero width, or collapsing on projection -- is
        # not empty by Shapely's definition but cannot be located.
        raise IdentityError("Cannot key a degenerate geometry.") from exc
    point = projected.representative_point()
    epsg = crs.to_epsg()
    zone = epsg % 100
    hemisphere = "S" if 32700 <= epsg < 32800 else "N"
    easting = int(round(point.x / precision_m) * precision_m)
    northing = int(round(point.y / precision_m) * precision_m)
    return f"UTM{zone}{hemisphere}-{easting:07d}-{northing:07d}"


def _metric_features(collection: dict[str, Any], crs: Any) -> list[tuple[dict[str, Any], BaseGeometry]]:
    features = collection.get("features")
    if collection.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise IdentityError("Candidates must be a GeoJSON FeatureCollection.")
    result: list[tuple[dict[str, Any], BaseGeometry]] = []
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            continue
        geometry, _ = to_metric(shape(feature["geometry"]), crs)
        if not geometry.is_empty:
            result.append((feature, geometry))
    return result


def assign_identities(
    collection: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
    survey_date: str | None = None,
    config: LinkConfig | None = None,
) -> dict[str, Any]:
    """Attach a ``site_id`` to every candidate, carrying it across surveys.

    A candidate that overlaps, or sits close to, an observation in ``previous``
    inherits its ``site_id``, ``first_seen`` date, and observation count. One
    that does not is a new site and takes a fresh id from its location.

    The id is deliberately taken from the *first* observation and never
    recomputed, so it does not drift as the outline is remeasured on later dates.
    """
    active = config or LinkConfig()
    active.validate()

    features = collection.get("features")
    if collection.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise IdentityError("Candidates must be a GeoJSON FeatureCollection.")
    if not features:
        return collection

    crs = metric_crs_for(next(
        feature["geometry"] for feature in features
        if isinstance(feature, dict) and isinstance(feature.get("geometry"), dict)
    ))
    current = _metric_features(collection, crs)
    earlier = _metric_features(previous, crs) if previous else []

    claimed: set[int] = set()
    for feature, geometry in current:
        properties = feature.setdefault("properties", {})
        best_index: int | None = None
        best_score = 0.0
        for index, (earlier_feature, earlier_geometry) in enumerate(earlier):
            if index in claimed:
                continue
            overlap = geometry.intersection(earlier_geometry).area
            smaller = min(geometry.area, earlier_geometry.area)
            overlap_fraction = overlap / smaller if smaller > 0 else 0.0
            distance = geometry.centroid.distance(earlier_geometry.centroid)
            if overlap_fraction < active.min_overlap_fraction and distance > active.max_centroid_distance_m:
                continue
            # Prefer real overlap; fall back to proximity when nothing overlaps.
            score = overlap_fraction if overlap_fraction > 0 else 1.0 / (1.0 + distance)
            if score > best_score:
                best_score, best_index = score, index

        if best_index is None:
            properties["site_id"] = location_key(feature["geometry"])
            properties["first_seen"] = survey_date or "unknown"
            properties["observation_count"] = 1
            properties["site_status"] = "new_site"
        else:
            claimed.add(best_index)
            earlier_properties = earlier[best_index][0].get("properties", {})
            properties["site_id"] = earlier_properties.get("site_id") or location_key(feature["geometry"])
            properties["first_seen"] = earlier_properties.get("first_seen", "unknown")
            properties["observation_count"] = int(earlier_properties.get("observation_count", 1)) + 1
            properties["site_status"] = "seen_before"
        properties["location_key"] = location_key(feature["geometry"])

    metadata = collection.setdefault("metadata", {})
    metadata["identity"] = {
        "survey_date": survey_date or "unknown",
        "linked_to_previous": bool(earlier),
        "max_centroid_distance_m": active.max_centroid_distance_m,
        "min_overlap_fraction": active.min_overlap_fraction,
        "note": "site_id links observations across surveys; location_key is a label, not a join key.",
    }
    return collection


def provenance(
    *,
    before_image: str,
    after_image: str,
    before_date: str | None,
    after_date: str | None,
    model: str,
    threshold: float,
    settings: dict[str, Any] | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    """Describe how a candidate set was produced.

    A candidate that may support enforcement has to say what imagery and what
    model version produced it. Without this a saved layer cannot be defended
    later, or reproduced.
    """
    return {
        "before_image": before_image,
        "after_image": after_image,
        "before_capture_date": before_date or "unknown",
        "after_capture_date": after_date or "unknown",
        "model": model,
        "threshold": threshold,
        "settings": settings or {},
        "code_commit": commit or "unrecorded",
        "status": "review_candidate_not_a_finding",
    }


def site_history(collections: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group observations by ``site_id`` across surveys, oldest first."""
    history: dict[str, list[dict[str, Any]]] = {}
    for collection in collections:
        for feature in collection.get("features", []):
            properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
            site = properties.get("site_id")
            if not site:
                continue
            history.setdefault(str(site), []).append(
                {
                    "survey_date": collection.get("metadata", {}).get("identity", {}).get("survey_date", "unknown"),
                    "classification": properties.get("classification"),
                    "area_m2": properties.get("area_m2"),
                    "changed_area_m2": properties.get("changed_area_m2"),
                }
            )
    return history
