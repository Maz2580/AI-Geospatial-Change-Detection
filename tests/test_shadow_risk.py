from __future__ import annotations

import unittest

from building_change.shadow_risk import ShadowRiskError, shadow_risk_level


class ShadowRiskTests(unittest.TestCase):
    def test_assigns_review_levels_without_discarding_candidates(self) -> None:
        self.assertEqual(shadow_risk_level(0.0), "low")
        self.assertEqual(shadow_risk_level(0.15), "medium")
        self.assertEqual(shadow_risk_level(0.5), "high")

    def test_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ShadowRiskError):
            shadow_risk_level(1.01)


if __name__ == "__main__":
    unittest.main()
