"""Human label templates for independent building-change benchmark cases.

Reference-map candidates are prompts for review, never copied into roof labels.
This keeps a future model evaluation independent from the detector or map that
suggested a location.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AnnotationError(ValueError):
    """Raised when an annotation template is malformed."""


ASSESSMENTS = {
    "unreviewed",
    "real_visible_change",
    "mapping_only_or_not_visible",
    "inconclusive_due_occlusion",
}
VISIBLE_CHANGE_TYPES = {
    "new_building",
    "building_extension",
    "demolition",
    "pool",
    "driveway",
    "garden_or_soil",
    "other_permanent_change",
}


def _candidate_ids(reference_collection: dict[str, Any]) -> list[int]:
    features = reference_collection.get("features")
    if reference_collection.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise AnnotationError("Reference candidates must be a GeoJSON FeatureCollection.")
    candidate_ids: list[int] = []
    for ordinal, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            raise AnnotationError("Every reference candidate must have a GeoJSON geometry.")
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        candidate_id = properties.get("candidate_id", ordinal)
        if not isinstance(candidate_id, int) or candidate_id <= 0 or candidate_id in candidate_ids:
            raise AnnotationError("Reference candidate IDs must be unique positive integers.")
        candidate_ids.append(candidate_id)
    if not candidate_ids:
        raise AnnotationError("Reference candidates must not be empty.")
    return candidate_ids


def create_reference_label_template(
    *,
    case_id: str,
    reference_collection: dict[str, Any],
    before_date: str,
    after_date: str,
    reference_path: str,
) -> dict[str, Any]:
    """Create an intentionally incomplete review document for one fixed pair."""
    if not isinstance(case_id, str) or not case_id:
        raise AnnotationError("case_id must be a non-empty string.")
    if not isinstance(before_date, str) or not before_date or not isinstance(after_date, str) or not after_date:
        raise AnnotationError("before_date and after_date must be non-empty ISO-date strings.")
    candidate_ids = _candidate_ids(reference_collection)
    metadata = reference_collection.get("metadata") if isinstance(reference_collection.get("metadata"), dict) else {}
    return {
        "schema_version": 1,
        "case_id": case_id,
        "status": "requires_human_review",
        "source_reference": {
            "path": reference_path,
            "source": metadata.get("source", "unknown"),
            "warning": "Reference-map geometry is a review prompt only. Do not copy it into after_roof_labels.",
        },
        "source_imagery": {"before": before_date, "after": after_date},
        "label_definition": {
            "real_visible_change": "A permanent change is visibly supported by the fixed before/after imagery.",
            "mapping_only_or_not_visible": "The reference-map difference is not a visible permanent change in the fixed imagery pair.",
            "inconclusive_due_occlusion": "A fixed-pair review cannot establish whether the reference-map difference is real because cloud, shadow, tree cover, image quality, or another occlusion hides decisive evidence. It is excluded from scoring.",
            "after_roof_labels": "Human-drawn after-date roof outlines. They must remain empty until manually digitised from the after image.",
        },
        "labels": [
            {
                "reference_candidate_id": candidate_id,
                "assessment": "unreviewed",
                "visible_change_type": "unreviewed",
                "review_notes": "",
            }
            for candidate_id in candidate_ids
        ],
        "after_roof_labels": {
            "type": "FeatureCollection",
            "features": [],
            "required_properties": {
                "object_id": "unique stable string, for example docklands-001-roof-01",
                "change_type": "new_building, building_extension, unchanged_building, or not_a_building",
                "reference_candidate_ids": "list of reviewed official-reference candidate IDs, possibly empty",
                "label_source": "human_review",
            },
        },
    }


def validate_reference_label_document(
    document: dict[str, Any], *, require_complete: bool = False
) -> dict[str, Any]:
    """Validate human assessments while allowing an explicit draft state."""
    if document.get("schema_version") != 1:
        raise AnnotationError("Annotation schema_version must be 1.")
    if not isinstance(document.get("case_id"), str) or not document["case_id"]:
        raise AnnotationError("Annotation document needs a case_id.")
    labels = document.get("labels")
    if not isinstance(labels, list) or not labels:
        raise AnnotationError("Annotation document needs at least one reference label.")
    seen: set[int] = set()
    reviewed = 0
    visible = 0
    inconclusive = 0
    for label in labels:
        if not isinstance(label, dict):
            raise AnnotationError("Each reference label must be an object.")
        candidate_id = label.get("reference_candidate_id")
        assessment = label.get("assessment")
        change_type = label.get("visible_change_type")
        if not isinstance(candidate_id, int) or candidate_id <= 0 or candidate_id in seen:
            raise AnnotationError("Reference label IDs must be unique positive integers.")
        seen.add(candidate_id)
        if assessment not in ASSESSMENTS:
            raise AnnotationError(f"Unsupported reference assessment: {assessment!r}")
        if assessment == "unreviewed":
            if change_type != "unreviewed":
                raise AnnotationError("An unreviewed reference label must use visible_change_type=unreviewed.")
            continue
        reviewed += 1
        if assessment == "real_visible_change":
            if change_type not in VISIBLE_CHANGE_TYPES:
                raise AnnotationError("A real visible change needs a supported permanent visible_change_type.")
            visible += 1
        elif assessment == "mapping_only_or_not_visible":
            if change_type != "not_applicable":
                raise AnnotationError("A mapping-only label must use visible_change_type=not_applicable.")
        else:
            if change_type != "not_applicable":
                raise AnnotationError("An inconclusive label must use visible_change_type=not_applicable.")
            review_notes = label.get("review_notes")
            if not isinstance(review_notes, str) or not review_notes.strip():
                raise AnnotationError("An inconclusive label must record a non-empty review_notes explanation.")
            inconclusive += 1
    if require_complete and reviewed != len(labels):
        raise AnnotationError("Annotation review is incomplete: at least one reference candidate is still unreviewed.")
    roof_labels = document.get("after_roof_labels")
    if not isinstance(roof_labels, dict) or roof_labels.get("type") != "FeatureCollection" or not isinstance(roof_labels.get("features"), list):
        raise AnnotationError("after_roof_labels must be a GeoJSON FeatureCollection.")
    return {
        "case_id": document["case_id"],
        "reference_candidate_count": len(labels),
        "reviewed_reference_count": reviewed,
        "unreviewed_reference_count": len(labels) - reviewed,
        "visible_permanent_change_count": visible,
        "inconclusive_reference_count": inconclusive,
        "after_roof_label_count": len(roof_labels["features"]),
        "complete": reviewed == len(labels),
    }


def load_annotation_document(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"Could not read annotation document: {path}") from exc
    if not isinstance(document, dict):
        raise AnnotationError("Annotation document must be a JSON object.")
    return document
