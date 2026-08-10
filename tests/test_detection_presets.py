from __future__ import annotations

import unittest

from building_change.detector import DETECTION_PRESETS, DetectionConfig


class DetectionPresetTests(unittest.TestCase):
    def test_balanced_preset_matches_the_dataclass_defaults(self) -> None:
        self.assertEqual(DetectionConfig.from_preset("balanced"), DetectionConfig())

    def test_high_recall_preset_relaxes_the_filters_that_zeroed_benchmark_recall(self) -> None:
        config = DetectionConfig.from_preset("high-recall")

        self.assertEqual(config.change_percentile, 96.0)
        self.assertEqual(config.min_area_m2, 10.0)
        self.assertEqual(config.morphology_m, 0.2)
        self.assertFalse(config.enable_shadow_filter)

    def test_explicit_overrides_take_precedence_over_the_preset(self) -> None:
        config = DetectionConfig.from_preset("high-recall", change_percentile=97.5, min_area_m2=None)

        self.assertEqual(config.change_percentile, 97.5)
        self.assertEqual(config.min_area_m2, 10.0)

    def test_rejects_an_unknown_preset(self) -> None:
        with self.assertRaises(ValueError):
            DetectionConfig.from_preset("aggressive")

    def test_every_preset_builds_a_valid_config(self) -> None:
        for name in DETECTION_PRESETS:
            self.assertIsInstance(DetectionConfig.from_preset(name), DetectionConfig)


if __name__ == "__main__":
    unittest.main()
