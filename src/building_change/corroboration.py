"""Corroborate footprint candidates with independent change evidence.

Measured on the Murchison estate, the pixel change detector covers 0% of 35 of
the 38 new buildings the footprint channel finds, and 85% of its output falls
outside every building footprint. A radiometric delta peaks at boundaries, so it
marks roof edges and surface transitions rather than whole buildings.

Fusing the two as equal votes therefore emits slivers as standalone building
candidates and splits single buildings across several fragments. This module
inverts the relationship: footprints supply the geometry, and the change
channels are asked only whether something changed there.

Nothing is dropped. Every footprint candidate is returned, tagged with which
evidence supports it, so a reviewer can sort by confidence rather than trust a
silent filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geodesy import metric_crs_for, to_metric

logger = logging.getLogger(__name__)

SUPPORT_TIERS = ("corroborated", "weakly_corroborated", "footprint_only")


class CorroborationError(ValueError):
    """Raised when footprint candidates cannot be corroborated."""


@dataclass(frozen=True)
class CorroborationConfig:
    """Rules for deciding whether change evidence backs a footprint candidate."""

    # Absorbs segmentation and registration differences between channels.
    support_distance_m: float = 2.0
    # Area fraction of the footprint that evidence must cover to count as strong.
    strong_support_fraction: float = 0.10
    min_sources_for_corroborated: int = 2

    def validate(self) -> None:
        if self.support_distance_m < 0:
            raise CorroborationError("support_distance_m cannot be negative.")
        if not 0 <= self.strong_support_fraction <= 1:
            raise CorroborationError("strong_support_fraction must be between 0 and 1.")
        if self.min_sources_for_corroborated < 1:
            raise CorroborationError("min_sources_for_corroborated must be at least one.")


def _metric(geometry: dict[str, Any], crs: Any) -> BaseGeometry:
    """Project into true ground metres, so support_distance_m means metres."""
    result, _ = to_metric(shape(geometry), crs)
    return result


def _measurement_crs(collection: dict[str, Any]) -> Any:
    """Pick one ground-metre CRS for the candidates and all their evidence."""
    features = collection.get("features")
    for feature in features if isinstance(features, list) else []:
        if isinstance(feature, dict) and isinstance(feature.get("geometry"), dict):
            try:
                return metric_crs_for(feature["geometry"])
            except Exception:  # noqa: BLE001 - try the next candidate geometry
                continue
    return None


def _evidence_union(collection: dict[str, Any], crs: Any) -> BaseGeometry | None:
    features = collection.get("features")
    if collection.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise CorroborationError("Evidence must be a GeoJSON FeatureCollection.")
    geometries = []
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            continue
        geometry = _metric(feature["geometry"], crs)
        if not geometry.is_empty and geometry.area > 0:
            geometries.append(geometry)
    return unary_union(geometries) if geometries else None


def corroborate_footprints(
    footprint_candidates: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    *,
    config: CorroborationConfig | None = None,
) -> dict[str, Any]:
    """Tag each footprint candidate with the change evidence that supports it."""
    active = config or CorroborationConfig()
    active.validate()

    features = footprint_candidates.get("features")
    if footprint_candidates.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise CorroborationError("Footprint candidates must be a GeoJSON FeatureCollection.")

    crs = _measurement_crs(footprint_candidates)
    if crs is None:
        raise CorroborationError("No usable candidate geometry to establish a measurement CRS.")

    unions = {name: _evidence_union(collection, crs) for name, collection in evidence.items()}
    unions = {name: geometry for name, geometry in unions.items() if geometry is not None}
    combined = unary_union(list(unions.values())) if unions else None

    tier_counts: dict[str, int] = {tier: 0 for tier in SUPPORT_TIERS}
    for feature in features:
        geometry = _metric(feature["geometry"], crs)
        if geometry.is_empty or geometry.area <= 0:
            continue

        supporting: list[str] = []
        details: dict[str, Any] = {}
        for name, union in unions.items():
            overlap = geometry.intersection(union).area / geometry.area
            distance = geometry.distance(union)
            if overlap > 0 or distance <= active.support_distance_m:
                supporting.append(name)
            details[name] = {
                "overlap_fraction": round(float(overlap), 4),
                "distance_m": round(float(distance), 2),
            }

        support_fraction = geometry.intersection(combined).area / geometry.area if combined else 0.0
        if len(supporting) >= active.min_sources_for_corroborated and support_fraction >= active.strong_support_fraction:
            tier = "corroborated"
        elif supporting:
            tier = "weakly_corroborated"
        else:
            tier = "footprint_only"
        tier_counts[tier] += 1

        feature.setdefault("properties", {})["change_support"] = {
            "tier": tier,
            "supporting_sources": sorted(supporting),
            "source_count": len(supporting),
            "support_fraction": round(float(support_fraction), 4),
            "per_source": details,
        }

    return {
        "candidate_count": len(features),
        "evidence_sources": sorted(unions),
        "tier_counts": tier_counts,
        "config": {
            "support_distance_m": active.support_distance_m,
            "strong_support_fraction": active.strong_support_fraction,
            "min_sources_for_corroborated": active.min_sources_for_corroborated,
        },
        "note": (
            "Footprints supply the geometry; change channels only indicate that something changed there. "
            "No candidate is removed, so review order should follow tier rather than assume filtering."
        ),
    }
