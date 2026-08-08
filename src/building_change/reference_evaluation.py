"""Score detector candidates against human-validated reference changes.

Reference footprints supply a compact evaluation set, not pixel-perfect truth.
The output therefore reports reference recall and flags unmatched predictions
for review instead of calling them false positives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rasterio.warp import transform_geom
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


class ReferenceEvaluationError(ValueError):
    """Raised when benchmark labels or candidate GeoJSON are not compatible."""


def _load_json(path: str | Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceEvaluationError(f"Could not read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ReferenceEvaluationError(f"{description} must be a JSON object.")
    return value


def _features(collection: dict[str, Any], *, description: str) -> dict[int, BaseGeometry]:
    values = collection.get("features")
    if collection.get("type") != "FeatureCollection" or not isinstance(values, list):
        raise ReferenceEvaluationError(f"{description} must be a GeoJSON FeatureCollection.")
    results: dict[int, BaseGeometry] = {}
    for ordinal, feature in enumerate(values, start=1):
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        candidate_id = properties.get("candidate_id", ordinal)
        if not isinstance(candidate_id, int) or candidate_id <= 0:
            raise ReferenceEvaluationError(f"{description} candidate IDs must be positive integers.")
        try:
            geometry = shape(transform_geom("EPSG:4326", "EPSG:3857", feature["geometry"], precision=6))
        except Exception as exc:
            raise ReferenceEvaluationError(f"{description} contains invalid WGS84 geometry.") from exc
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty or geometry.area <= 0:
            continue
        results[candidate_id] = geometry
    return results


def _real_reference_ids(labels: dict[str, Any]) -> tuple[set[int], set[int]]:
    values = labels.get("labels")
    if not isinstance(values, list):
        raise ReferenceEvaluationError("Reference labels must contain a labels list.")
    real: set[int] = set()
    rejected: set[int] = set()
    for record in values:
        if not isinstance(record, dict):
            raise ReferenceEvaluationError("Each reference label must be an object.")
        candidate_id = record.get("reference_candidate_id")
        assessment = record.get("assessment")
        if not isinstance(candidate_id, int) or candidate_id <= 0 or not isinstance(assessment, str):
            raise ReferenceEvaluationError("Each reference label needs a positive ID and assessment.")
        if assessment == "real_visible_change":
            real.add(candidate_id)
        elif assessment == "mapping_only_or_not_visible":
            rejected.add(candidate_id)
    if real & rejected:
        raise ReferenceEvaluationError("A reference candidate cannot be both real and rejected.")
    if not real:
        raise ReferenceEvaluationError("No reference labels are marked real_visible_change.")
    return real, rejected


def evaluate_reference_candidates(
    reference_collection: dict[str, Any],
    label_document: dict[str, Any],
    prediction_collection: dict[str, Any],
    *,
    match_distance_m: float = 8.0,
) -> dict[str, Any]:
    """Match detection candidates to human-confirmed visible reference changes."""
    if match_distance_m < 0:
        raise ReferenceEvaluationError("Match distance cannot be negative.")
    reference = _features(reference_collection, description="reference candidates")
    predictions = _features(prediction_collection, description="prediction candidates")
    real_ids, rejected_ids = _real_reference_ids(label_document)
    missing = sorted((real_ids | rejected_ids) - set(reference))
    if missing:
        raise ReferenceEvaluationError(f"Labels refer to reference candidates that do not exist: {missing}")
    hit_predictions: set[int] = set()
    rejected_matches: set[int] = set()
    per_reference: list[dict[str, Any]] = []
    for reference_id in sorted(real_ids):
        geometry = reference[reference_id]
        matched = sorted(
            prediction_id for prediction_id, prediction in predictions.items()
            if geometry.intersects(prediction) or geometry.distance(prediction) <= match_distance_m
        )
        hit_predictions.update(matched)
        per_reference.append({"reference_candidate_id": reference_id, "matched_prediction_ids": matched, "detected": bool(matched)})
    for reference_id in rejected_ids:
        geometry = reference[reference_id]
        for prediction_id, prediction in predictions.items():
            if geometry.intersects(prediction) or geometry.distance(prediction) <= match_distance_m:
                rejected_matches.add(prediction_id)
    unmatched = sorted(set(predictions) - hit_predictions)
    return {
        "human_confirmed_reference_count": len(real_ids),
        "detected_reference_count": sum(item["detected"] for item in per_reference),
        "reference_recall": round(sum(item["detected"] for item in per_reference) / len(real_ids), 4),
        "prediction_candidate_count": len(predictions),
        "prediction_ids_matching_real_reference": sorted(hit_predictions),
        "unmatched_prediction_ids_requiring_review": unmatched,
        "prediction_ids_matching_rejected_reference": sorted(rejected_matches),
        "match_distance_m": match_distance_m,
        "per_reference": per_reference,
        "warning": "Unmatched predictions are not called false positives because this compact reference set is incomplete outside its labelled differences.",
    }


def load_and_evaluate(
    reference_path: str | Path,
    labels_path: str | Path,
    predictions_path: str | Path,
    *,
    match_distance_m: float = 8.0,
) -> dict[str, Any]:
    """Load file inputs and evaluate one detector output reproducibly."""
    return evaluate_reference_candidates(
        _load_json(reference_path, description="reference candidates"),
        _load_json(labels_path, description="reference labels"),
        _load_json(predictions_path, description="prediction candidates"),
        match_distance_m=match_distance_m,
    )
