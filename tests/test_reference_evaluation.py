from __future__ import annotations

import unittest

from building_change.reference_evaluation import evaluate_reference_candidates


def _feature(candidate_id: int, west: float, south: float, east: float, north: float) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]},
        "properties": {"candidate_id": candidate_id},
    }


class ReferenceEvaluationTests(unittest.TestCase):
    def test_reports_real_reference_recall_and_flags_rejected_matches(self) -> None:
        reference = {"type": "FeatureCollection", "features": [_feature(1, 144.9510, -37.8110, 144.9512, -37.8108), _feature(2, 144.9520, -37.8110, 144.9522, -37.8108), _feature(3, 144.9530, -37.8110, 144.9532, -37.8108)]}
        labels = {"labels": [{"reference_candidate_id": 1, "assessment": "real_visible_change"}, {"reference_candidate_id": 2, "assessment": "real_visible_change"}, {"reference_candidate_id": 3, "assessment": "mapping_only_or_not_visible"}]}
        predictions = {"type": "FeatureCollection", "features": [_feature(11, 144.95105, -37.81095, 144.95115, -37.81085), _feature(12, 144.95305, -37.81095, 144.95315, -37.81085), _feature(13, 144.9550, -37.8110, 144.9552, -37.8108)]}
        report = evaluate_reference_candidates(reference, labels, predictions, match_distance_m=2)
        self.assertEqual(report["reference_recall"], 0.5)
        self.assertEqual(report["prediction_ids_matching_real_reference"], [11])
        self.assertEqual(report["prediction_ids_matching_rejected_reference"], [12])
        self.assertEqual(report["unmatched_prediction_ids_requiring_review"], [12, 13])


if __name__ == "__main__":
    unittest.main()
