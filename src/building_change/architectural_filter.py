"""Post-fusion architectural filter.

Applies calibrated thresholds (min_area_m2, min_rectangularity,
require_vlm_confirmation) to eliminate false positives from the fused
candidate set.  Designed to be called at the end of the enhanced pipeline
or standalone.
"""

from __future__ import annotations

import logging
from typing import Any

from shapely.geometry import shape
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def apply_architectural_filter(
    collection: dict[str, Any],
    calibrated_params: dict[str, Any],
) -> dict[str, Any]:
    """Filter a fused FeatureCollection using calibrated thresholds.

    Parameters
    ----------
    collection : dict
        A GeoJSON FeatureCollection.
    calibrated_params : dict
        The ``optimal_parameters`` section from
        ``config/calibrated_detection_config.json``.

    Returns
    -------
    dict
        A new FeatureCollection containing only features that pass all
        calibrated thresholds.
    """
    min_area = float(calibrated_params.get("min_area_m2", 0.0))
    min_rect = float(calibrated_params.get("min_rectangularity", 0.0))
    require_vlm = bool(calibrated_params.get("require_vlm_confirmation", False))

    features = collection.get("features", [])
    accepted: list[dict[str, Any]] = []

    for feat in features:
        props = feat.get("properties", {})

        # --- Area filter ---
        area = props.get("area_m2", 0.0)
        if area < min_area:
            logger.debug(
                "Rejected candidate %s: area %.1f < %.1f",
                props.get("candidate_id"), area, min_area,
            )
            continue

        # --- Rectangularity filter ---
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)
        min_rotated_rect = geom.minimum_rotated_rectangle
        rectangularity = (
            geom.area / min_rotated_rect.area
            if min_rotated_rect.area > 0
            else 0.0
        )
        if rectangularity < min_rect:
            logger.debug(
                "Rejected candidate %s: rectangularity %.3f < %.3f",
                props.get("candidate_id"), rectangularity, min_rect,
            )
            continue

        # --- VLM confirmation filter ---
        if require_vlm:
            vlm_verdict = props.get("vlm_verdict")
            vlm_type = props.get("vlm_type")
            has_vlm = vlm_verdict is not None or vlm_type is not None
            if not has_vlm:
                logger.debug(
                    "Rejected candidate %s: require_vlm_confirmation is true "
                    "but vlm_verdict=%s, vlm_type=%s",
                    props.get("candidate_id"), vlm_verdict, vlm_type,
                )
                continue

        # --- Passed all filters ---
        accepted.append(feat)

    # Re-number candidates
    for idx, feat in enumerate(accepted, 1):
        feat["properties"]["candidate_id"] = idx

    logger.info(
        "Architectural filter: %d → %d candidates (eliminated %d)",
        len(features), len(accepted), len(features) - len(accepted),
    )

    return {
        "type": "FeatureCollection",
        "features": accepted,
        "metadata": {
            **collection.get("metadata", {}),
            "architectural_filter_applied": True,
            "filter_config": calibrated_params,
        },
    }
