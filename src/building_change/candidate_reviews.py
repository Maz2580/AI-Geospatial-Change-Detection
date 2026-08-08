"""Validation for human review of model-generated change candidates.

These reviews keep target construction, temporary/movable objects, and
ambiguous imagery separate. They are evidence for model comparison, not model
training labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CandidateReviewError(ValueError):
    """Raised when a model-candidate review document is malformed."""


ASSESSMENTS = {
    "confirmed_target_change",
    "temporary_or_movable_change",
    "not_target_change",
    "inconclusive_due_image_ambiguity",
}
TARGET_TYPES = {"new_building", "building_extension", "demolition"}
OUTLINE_QUALITIES = {"good", "acceptable", "poor", "not_assessed"}


def validate_candidate_review_document(document: dict[str, Any]) -> dict[str, Any]:
    """Validate a human review while retaining non-target detections honestly."""
    if document.get("schema_version") != 1:
        raise CandidateReviewError("Candidate review schema_version must be 1.")
    if not isinstance(document.get("case_id"), str) or not document["case_id"]:
        raise CandidateReviewError("Candidate review needs a case_id.")
    model_reviews = document.get("model_reviews")
    if not isinstance(model_reviews, list) or not model_reviews:
        raise CandidateReviewError("Candidate review needs at least one model review.")

    totals = {assessment: 0 for assessment in ASSESSMENTS}
    model_summaries: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for model_review in model_reviews:
        if not isinstance(model_review, dict):
            raise CandidateReviewError("Each model review must be an object.")
        model_id = model_review.get("model_id")
        candidate_path = model_review.get("candidate_path")
        labels = model_review.get("labels")
        if not isinstance(model_id, str) or not model_id or model_id in seen_models:
            raise CandidateReviewError("Model review IDs must be unique non-empty strings.")
        if not isinstance(candidate_path, str) or not candidate_path:
            raise CandidateReviewError(f"Model review {model_id!r} needs a candidate_path.")
        if not isinstance(labels, list) or not labels:
            raise CandidateReviewError(f"Model review {model_id!r} needs at least one candidate label.")
        seen_models.add(model_id)
        seen_candidates: set[int] = set()
        counts = {assessment: 0 for assessment in ASSESSMENTS}
        for label in labels:
            if not isinstance(label, dict):
                raise CandidateReviewError("Each model candidate label must be an object.")
            candidate_id = label.get("candidate_id")
            assessment = label.get("assessment")
            target_type = label.get("target_type")
            outline_quality = label.get("outline_quality")
            notes = label.get("review_notes")
            if not isinstance(candidate_id, int) or candidate_id <= 0 or candidate_id in seen_candidates:
                raise CandidateReviewError("Candidate IDs must be unique positive integers within each model review.")
            if assessment not in ASSESSMENTS:
                raise CandidateReviewError(f"Unsupported candidate assessment: {assessment!r}")
            if outline_quality not in OUTLINE_QUALITIES:
                raise CandidateReviewError(f"Unsupported outline quality: {outline_quality!r}")
            if not isinstance(notes, str) or not notes.strip():
                raise CandidateReviewError("Every candidate review must include non-empty review_notes.")
            if assessment == "confirmed_target_change":
                if target_type not in TARGET_TYPES:
                    raise CandidateReviewError("A confirmed target change needs a supported target_type.")
                if outline_quality == "not_assessed":
                    raise CandidateReviewError("A confirmed target change needs an assessed outline quality.")
            elif target_type != "not_applicable":
                raise CandidateReviewError("Non-target or inconclusive candidates must use target_type=not_applicable.")
            seen_candidates.add(candidate_id)
            counts[assessment] += 1
            totals[assessment] += 1
        model_summaries.append({"model_id": model_id, "reviewed_candidate_count": len(labels), "assessment_counts": counts})
    return {
        "case_id": document["case_id"],
        "model_review_count": len(model_reviews),
        "reviewed_candidate_count": sum(item["reviewed_candidate_count"] for item in model_summaries),
        "assessment_counts": totals,
        "models": model_summaries,
    }


def load_candidate_review_document(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateReviewError(f"Could not read candidate review document: {path}") from exc
    if not isinstance(document, dict):
        raise CandidateReviewError("Candidate review document must be a JSON object.")
    return document
