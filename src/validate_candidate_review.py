"""Validate and summarise a human review of model-generated candidates."""

from __future__ import annotations

import argparse
import json

from building_change.candidate_reviews import CandidateReviewError, load_candidate_review_document, validate_candidate_review_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a model-candidate human review document.")
    parser.add_argument("--review", required=True, help="Path to a candidate review JSON document.")
    args = parser.parse_args()
    try:
        report = validate_candidate_review_document(load_candidate_review_document(args.review))
    except CandidateReviewError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
