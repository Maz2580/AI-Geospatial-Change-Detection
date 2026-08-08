"""Annotate candidate polygons with before/after cast-shadow risk.

High-recall change extraction should not discard a whole real construction site
because one fragment overlaps a shadow. This module keeps the candidate and
records shadow evidence for ranking and human review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

from .detector import DetectionConfig, _read_new_image, _reproject_rgb, _shadow_mask


class ShadowRiskError(ValueError):
    """Raised when candidates cannot be annotated on the after-image grid."""


def shadow_risk_level(overlap_fraction: float) -> str:
    """Classify a combined before/after shadow overlap conservatively."""
    if not 0 <= overlap_fraction <= 1:
        raise ShadowRiskError("Shadow overlap fraction must be between zero and one.")
    if overlap_fraction >= 0.5:
        return "high"
    if overlap_fraction >= 0.15:
        return "medium"
    return "low"


def _load_collection(path: str | Path) -> dict[str, Any]:
    try:
        collection = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowRiskError(f"Could not read candidate GeoJSON: {path}") from exc
    if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
        raise ShadowRiskError("Candidates must be a GeoJSON FeatureCollection.")
    return collection


def annotate_shadow_risk(
    before_image: str | Path,
    after_image: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    *,
    config: DetectionConfig | None = None,
) -> dict[str, Any]:
    """Write candidates with independent before/after shadow-overlap fields."""
    before_path, after_path = Path(before_image), Path(after_image)
    if not before_path.is_file() or not after_path.is_file():
        raise FileNotFoundError("Both before and after RGB GeoTIFFs must exist.")
    collection = _load_collection(candidate_path)
    active_config = config or DetectionConfig()
    after_rgb, after_valid, profile = _read_new_image(after_path)
    before_rgb, before_valid = _reproject_rgb(before_path, profile)
    before_shadow = _shadow_mask(before_rgb, before_valid, profile, active_config)
    after_shadow = _shadow_mask(after_rgb, after_valid, profile, active_config)
    source_crs = profile.get("crs")
    if source_crs is None:
        raise ShadowRiskError("The after image has no CRS; candidates cannot be aligned for shadow annotation.")
    annotated: list[dict[str, Any]] = []
    counts = {"low": 0, "medium": 0, "high": 0, "invalid_geometry": 0}
    for feature in collection["features"]:
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            continue
        properties = dict(feature.get("properties") or {})
        try:
            image_geometry = transform_geom("EPSG:4326", source_crs, feature["geometry"], precision=6)
            footprint = geometry_mask([image_geometry], out_shape=after_valid.shape, transform=profile["transform"], invert=True)
        except Exception:
            properties.update({"shadow_risk": "invalid_geometry", "before_shadow_fraction": None, "after_shadow_fraction": None, "combined_shadow_fraction": None})
            counts["invalid_geometry"] += 1
        else:
            valid_footprint = footprint & (before_valid | after_valid)
            pixel_count = int(valid_footprint.sum())
            before_fraction = float((before_shadow & valid_footprint).sum()) / pixel_count if pixel_count else 0.0
            after_fraction = float((after_shadow & valid_footprint).sum()) / pixel_count if pixel_count else 0.0
            combined = float(((before_shadow | after_shadow) & valid_footprint).sum()) / pixel_count if pixel_count else 0.0
            risk = shadow_risk_level(combined)
            properties.update(
                {
                    "shadow_risk": risk,
                    "before_shadow_fraction": round(before_fraction, 4),
                    "after_shadow_fraction": round(after_fraction, 4),
                    "combined_shadow_fraction": round(combined, 4),
                }
            )
            counts[risk] += 1
        annotated.append({"type": "Feature", "geometry": feature["geometry"], "properties": properties})
    result = {
        "type": "FeatureCollection",
        "features": annotated,
        "metadata": {
            "evidence_role": "change_candidate_with_shadow_risk",
            "before_image": str(before_path),
            "after_image": str(after_path),
            "shadow_risk_counts": counts,
            "warning": "Shadow risk is a review signal. It does not disprove a real construction change.",
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
