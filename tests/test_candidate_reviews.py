from __future__ import annotations

import copy
import unittest

from building_change.candidate_reviews import CandidateReviewError, validate_candidate_review_document


def _review() -> dict:
    return {
        "schema_version": 1,
        "case_id": "docklands",
        "model_reviews": [
            {
                "model_id": "example-model",
                "candidate_path": "data/example.geojson",
                "labels": [
                    {
                        "candidate_id": 1,
                        "assessment": "confirmed_target_change",
                        "target_type": "new_building",
                        "outline_quality": "good",
                        "review_notes": "Visible permanent building change.",
                    },
                    {
                        "candidate_id": 2,
                        "assessment": "temporary_or_movable_change",
                        "target_type": "not_applicable",
                        "outline_quality": "not_assessed",
                        "review_notes": "Ship left the scene.",
                    },
                ],
            }
        ],
    }


class CandidateReviewTests(unittest.TestCase):
    def test_preserves_temporary_objects_separately_from_target_changes(self) -> None:
        report = validate_candidate_review_document(_review())
        self.assertEqual(report["reviewed_candidate_count"], 2)
        self.assertEqual(report["assessment_counts"]["confirmed_target_change"], 1)
        self.assertEqual(report["assessment_counts"]["temporary_or_movable_change"], 1)

    def test_confirmed_target_requires_outline_assessment(self) -> None:
        document = _review()
        document["model_reviews"][0]["labels"][0]["outline_quality"] = "not_assessed"
        with self.assertRaisesRegex(CandidateReviewError, "outline quality"):
            validate_candidate_review_document(document)

    def test_non_target_cannot_claim_a_target_type(self) -> None:
        document = copy.deepcopy(_review())
        document["model_reviews"][0]["labels"][1]["target_type"] = "new_building"
        with self.assertRaisesRegex(CandidateReviewError, "not_applicable"):
            validate_candidate_review_document(document)


if __name__ == "__main__":
    unittest.main()
