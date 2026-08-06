"""Summarise human-labelled construction-site review results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class EvaluationError(ValueError):
    """Raised when a human-label file does not follow the expected schema."""


def summarise_labels(labels: dict[str, Any]) -> dict[str, Any]:
    """Calculate transparent baseline metrics from site-review labels.

    Building recall is reported only for labels that provide a known total. The
    output deliberately avoids an overall precision score because a site can
    include valid driveway, pool, garden, and building changes at once.
    """
    records = labels.get("labels")
    if not isinstance(records, list):
        raise EvaluationError("Label file must contain a labels list.")
    known_building_total = 0
    detected_buildings = 0
    assessment_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    outline_counts: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            raise EvaluationError("Each label must be an object.")
        assessment_counts[str(record.get("assessment", "unlabelled"))] += 1
        outline_counts[str(record.get("outline_quality", "unlabelled"))] += 1
        issues = record.get("issues", [])
        if not isinstance(issues, list):
            raise EvaluationError("Each label's issues field must be a list.")
        issue_counts.update(str(issue) for issue in issues)
        buildings = record.get("buildings", {})
        if not isinstance(buildings, dict):
            raise EvaluationError("Each label's buildings field must be an object.")
        detected, total = buildings.get("detected"), buildings.get("total")
        if total is not None:
            if not isinstance(detected, int) or not isinstance(total, int) or detected < 0 or total < 0 or detected > total:
                raise EvaluationError("Known building counts must be non-negative integers with detected <= total.")
            detected_buildings += detected
            known_building_total += total
    return {
        "labelled_site_count": len(records),
        "known_building_total": known_building_total,
        "detected_building_total": detected_buildings,
        "building_recall_on_known_sites": round(detected_buildings / known_building_total, 4) if known_building_total else None,
        "assessment_counts": dict(sorted(assessment_counts.items())),
        "outline_quality_counts": dict(sorted(outline_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def load_and_summarise(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        labels = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read label file: {source}") from exc
    if not isinstance(labels, dict):
        raise EvaluationError("Label file must be a JSON object.")
    return summarise_labels(labels)
