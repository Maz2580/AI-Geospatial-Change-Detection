"""Smart Fusion module for Detector v5 + Local Multi-Source Pipeline.

Combines the high-level VLM location intelligence of Detector v5 with the
7.5cm sub-pixel 90°/45° shadow-cleared geometry of the local pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from shapely.geometry import mapping, shape
from shapely.validation import make_valid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def smart_fuse_candidates(
    v5_geojson_path: str | Path,
    local_geojson_path: str | Path,
    output_geojson_path: str | Path,
    *,
    iou_threshold: float = 0.10,
) -> dict[str, Any]:
    v5_path = Path(v5_geojson_path)
    local_path = Path(local_geojson_path)
    out_path = Path(output_geojson_path)

    if not v5_path.is_file() or not local_path.is_file():
        raise FileNotFoundError("Both input GeoJSON files must exist.")

    v5_data = json.loads(v5_path.read_text(encoding="utf-8"))
    local_data = json.loads(local_path.read_text(encoding="utf-8"))

    v5_features = v5_data.get("features", [])
    local_features = local_data.get("features", [])

    logger.info("Loaded %d Detector v5 candidates", len(v5_features))
    logger.info("Loaded %d Local Pipeline candidates", len(local_features))

    # Parse geometries
    v5_geoms = []
    for idx, feat in enumerate(v5_features):
        try:
            g = shape(feat["geometry"])
            if not g.is_valid:
                g = make_valid(g)
            v5_geoms.append((idx, g, feat["properties"]))
        except Exception:
            continue

    local_geoms = []
    for idx, feat in enumerate(local_features):
        try:
            g = shape(feat["geometry"])
            if not g.is_valid:
                g = make_valid(g)
            local_geoms.append((idx, g, feat["properties"]))
        except Exception:
            continue

    matched_local_indices = set()
    matched_v5_indices = set()
    fused_features: list[dict[str, Any]] = []

    # 1. Multi-source Agreement Matching (V5 + Local)
    for v_idx, v_g, v_props in v5_geoms:
        best_iou = 0.0
        best_local_match = None
        best_l_idx = -1

        for l_idx, l_g, l_props in local_geoms:
            if not v_g.intersects(l_g):
                continue
            inter = v_g.intersection(l_g).area
            union = v_g.union(l_g).area
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_local_match = (l_g, l_props)
                best_l_idx = l_idx

        if best_iou >= iou_threshold and best_local_match is not None:
            l_g, l_props = best_local_match
            matched_v5_indices.add(v_idx)
            matched_local_indices.add(best_l_idx)

            # Use local 7.5cm shadow-cleared geometry + attach V5 VLM metadata
            fused_feat = {
                "type": "Feature",
                "geometry": mapping(l_g),
                "properties": {
                    "candidate_id": len(fused_features) + 1,
                    "classification": "high_confidence_fused_building",
                    "agreement_type": "multi_model_agreement",
                    "vlm_verdict": v_props.get("vlm_verdict"),
                    "vlm_note": v_props.get("vlm_note"),
                    "vlm_type": v_props.get("vlm_type"),
                    "address": v_props.get("address"),
                    "parcel_spi": v_props.get("prop_propnum"),
                    "clip_type": v_props.get("clip_type"),
                    "v5_score": v_props.get("score"),
                    "local_classification": l_props.get("classification"),
                    "area_m2": l_props.get("area_m2", round(l_g.area, 2)),
                    "candidate_source": "fused_v5_local",
                    "agreement_iou": round(best_iou, 3),
                },
            }
            fused_features.append(fused_feat)

    # 2. V5 High-Confidence Candidates (Confirmed by VLM but no local match)
    for v_idx, v_g, v_props in v5_geoms:
        if v_idx in matched_v5_indices:
            continue
        is_building = (
            v_props.get("vlm_type") == "new building"
            or v_props.get("vlm_verdict") == "construction"
            or v_props.get("kind") == "new_or_cleared"
        )
        if is_building:
            fused_feat = {
                "type": "Feature",
                "geometry": mapping(v_g),
                "properties": {
                    "candidate_id": len(fused_features) + 1,
                    "classification": "v5_confirmed_building",
                    "agreement_type": "v5_exclusive",
                    "vlm_verdict": v_props.get("vlm_verdict"),
                    "vlm_note": v_props.get("vlm_note"),
                    "vlm_type": v_props.get("vlm_type"),
                    "address": v_props.get("address"),
                    "v5_score": v_props.get("score"),
                    "area_m2": v_props.get("area_m2", round(v_g.area, 2)),
                    "candidate_source": "detector_v5",
                },
            }
            fused_features.append(fused_feat)

    # 3. High-Confidence Local Footprint Candidates (Confirmed by WHU/BIT multi-source)
    for l_idx, l_g, l_props in local_geoms:
        if l_idx in matched_local_indices:
            continue
        if l_props.get("agreement") == "multi_source_agreement":
            fused_feat = {
                "type": "Feature",
                "geometry": mapping(l_g),
                "properties": {
                    "candidate_id": len(fused_features) + 1,
                    "classification": l_props.get("classification", "likely_building_footprint"),
                    "agreement_type": "local_multi_source",
                    "area_m2": l_props.get("area_m2", round(l_g.area, 2)),
                    "candidate_source": "local_pipeline",
                },
            }
            fused_features.append(fused_feat)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_collection = {"type": "FeatureCollection", "features": fused_features}
    out_path.write_text(json.dumps(out_collection, indent=2), encoding="utf-8")

    report = {
        "total_fused_candidates": len(fused_features),
        "multi_model_agreement_count": len(matched_v5_indices),
        "v5_exclusive_count": len([f for f in fused_features if f["properties"]["agreement_type"] == "v5_exclusive"]),
        "local_multi_source_count": len([f for f in fused_features if f["properties"]["agreement_type"] == "local_multi_source"]),
        "output_file": str(out_path),
    }

    logger.info("Smart Fusion complete! Results: %s", json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    smart_fuse_candidates(
        "data/output/detector_v5_candidates.geojson",
        "data/output/shadow_subtracted_test/enhanced_fused_candidates.geojson",
        "data/output/smart_fused_test/smart_fused_candidates.geojson",
    )
