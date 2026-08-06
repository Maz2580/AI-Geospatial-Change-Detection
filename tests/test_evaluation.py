from __future__ import annotations

import unittest

from building_change.evaluation import summarise_labels


class EvaluationTests(unittest.TestCase):
    def test_reports_building_recall_only_from_known_totals(self) -> None:
        result = summarise_labels(
            {
                "labels": [
                    {"assessment": "partial", "outline_quality": "poor", "issues": ["shadow"], "buildings": {"detected": 1, "total": 3}},
                    {"assessment": "correct", "outline_quality": "good", "issues": [], "buildings": {"detected": 2, "total": 2}},
                    {"assessment": "partial", "outline_quality": "poor", "issues": ["pool"], "buildings": {"detected": 1, "total": None}},
                ]
            }
        )
        self.assertEqual(result["known_building_total"], 5)
        self.assertEqual(result["detected_building_total"], 3)
        self.assertEqual(result["building_recall_on_known_sites"], 0.6)
        self.assertEqual(result["issue_counts"], {"pool": 1, "shadow": 1})


if __name__ == "__main__":
    unittest.main()
