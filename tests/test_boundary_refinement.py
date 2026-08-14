from __future__ import annotations

import unittest

import numpy as np

from building_change.boundary_refinement import (
    BoundaryRefinementConfig,
    BoundaryRefinementError,
    refine_building_mask,
    refinement_report,
)

PIXEL_M = 0.15


SIZE = 140
CENTRE = SIZE // 2


def _scene(roof_half_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A dark square roof on light ground, under a realistically soft probability.

    The probability field decays with distance from the centre rather than
    switching between 0 and 1. Two things depend on that. A hard field leaves no
    ambiguous pixels for refinement to decide; and the field must reach confident
    background *within the window*, or GrabCut has no negative class to estimate
    a colour model from and declines to run. The measured field is soft -- 12.7%
    of a real scene sits in the 0.2-0.8 band -- and falls to zero a few metres
    from a roof, which is what this reproduces.

    Confident core lands at radius 15.5 px, the 0.35 contour at 27.5, and
    background at 35, all inside the default 4 m (27 px) padding. So the
    threshold outline overshoots a 20 px roof and undershoots a 33 px one.
    """
    rows, columns = np.mgrid[0:SIZE, 0:SIZE]
    distance = np.maximum(np.abs(rows - CENTRE), np.abs(columns - CENTRE))
    probability = np.clip(1.0 - (distance - 8.0) / 30.0, 0.0, 1.0).astype(np.float32)
    valid = np.ones((SIZE, SIZE), dtype=bool)

    rgb = np.full((3, SIZE, SIZE), 210.0, dtype=np.float32)
    roof = distance <= roof_half_width
    rgb[:, roof] = 35.0
    return probability, valid, rgb


def _roof_truth(roof_half_width: int) -> np.ndarray:
    rows, columns = np.mgrid[0:SIZE, 0:SIZE]
    return np.maximum(np.abs(rows - CENTRE), np.abs(columns - CENTRE)) <= roof_half_width


class RefinementTests(unittest.TestCase):
    def test_an_over_large_outline_is_pulled_back_to_the_roof(self) -> None:
        """The failure that matters: the threshold contour overshoots the roof."""
        probability, valid, rgb = _scene(roof_half_width=20)
        truth = _roof_truth(20)

        threshold_mask = (probability >= 0.35) & valid
        refined = refine_building_mask(
            probability, valid, rgb, threshold=0.35, pixel_size_m=PIXEL_M,
            config=BoundaryRefinementConfig(enabled=True),
        )

        self.assertLess(refined.sum(), threshold_mask.sum())
        self.assertGreater(
            (refined & truth).sum() / (refined | truth).sum(),
            (threshold_mask & truth).sum() / (threshold_mask | truth).sum(),
        )

    def test_an_under_sized_outline_grows_onto_the_roof(self) -> None:
        probability, valid, rgb = _scene(roof_half_width=33)
        truth = _roof_truth(33)

        threshold_mask = (probability >= 0.35) & valid
        refined = refine_building_mask(
            probability, valid, rgb, threshold=0.35, pixel_size_m=PIXEL_M,
            config=BoundaryRefinementConfig(enabled=True),
        )
        self.assertGreater(refined.sum(), threshold_mask.sum())
        self.assertGreater(
            (refined & truth).sum() / (refined | truth).sum(),
            (threshold_mask & truth).sum() / (threshold_mask | truth).sum(),
        )

    def test_refinement_is_off_unless_asked_for(self) -> None:
        probability, valid, rgb = _scene(roof_half_width=20)
        threshold_mask = (probability >= 0.35) & valid

        result = refine_building_mask(probability, valid, rgb, threshold=0.35, pixel_size_m=PIXEL_M)
        np.testing.assert_array_equal(result, threshold_mask)

    def test_a_building_with_no_confident_core_keeps_its_threshold_outline(self) -> None:
        """Refinement must never delete a detection it cannot model."""
        probability = np.zeros((SIZE, SIZE), dtype=np.float32)
        probability[50:70, 50:70] = 0.45  # above threshold, below any core value
        valid = np.ones((SIZE, SIZE), dtype=bool)
        rgb = np.full((3, SIZE, SIZE), 120.0, dtype=np.float32)

        threshold_mask = (probability >= 0.35) & valid
        refined = refine_building_mask(
            probability, valid, rgb, threshold=0.35, pixel_size_m=PIXEL_M,
            config=BoundaryRefinementConfig(enabled=True),
        )
        np.testing.assert_array_equal(refined, threshold_mask)

    def test_nothing_detected_stays_nothing(self) -> None:
        probability = np.zeros((60, 60), dtype=np.float32)
        valid = np.ones((60, 60), dtype=bool)
        rgb = np.full((3, 60, 60), 200.0, dtype=np.float32)

        refined = refine_building_mask(
            probability, valid, rgb, threshold=0.35, pixel_size_m=PIXEL_M,
            config=BoundaryRefinementConfig(enabled=True),
        )
        self.assertFalse(refined.any())

    def test_invalid_pixels_are_never_building(self) -> None:
        probability, valid, rgb = _scene(roof_half_width=20)
        valid[:, :60] = False

        refined = refine_building_mask(
            probability, valid, rgb, threshold=0.35, pixel_size_m=PIXEL_M,
            config=BoundaryRefinementConfig(enabled=True),
        )
        self.assertFalse(refined[:, :60].any())


class ConfigurationTests(unittest.TestCase):
    def test_probability_ordering_is_enforced(self) -> None:
        with self.assertRaises(BoundaryRefinementError):
            BoundaryRefinementConfig(core_probability=0.2, background_probability=0.5).validate()

    def test_padding_and_iterations_must_be_positive(self) -> None:
        with self.assertRaises(BoundaryRefinementError):
            BoundaryRefinementConfig(padding_m=0.0).validate()
        with self.assertRaises(BoundaryRefinementError):
            BoundaryRefinementConfig(iterations=0).validate()

    def test_a_bad_pixel_size_is_refused(self) -> None:
        probability, valid, rgb = _scene(roof_half_width=20)
        with self.assertRaises(BoundaryRefinementError):
            refine_building_mask(
                probability, valid, rgb, threshold=0.35, pixel_size_m=0.0,
                config=BoundaryRefinementConfig(enabled=True),
            )


class ReportTests(unittest.TestCase):
    def test_movement_is_reported_in_both_directions(self) -> None:
        before = np.zeros((10, 10), dtype=bool)
        before[2:6, 2:6] = True
        after = np.zeros((10, 10), dtype=bool)
        after[3:8, 3:8] = True

        report = refinement_report(before, after, 1.0)
        self.assertEqual(report["threshold_area_m2"], 16.0)
        self.assertEqual(report["refined_area_m2"], 25.0)
        self.assertGreater(report["area_added_m2"], 0)
        self.assertGreater(report["area_removed_m2"], 0)


if __name__ == "__main__":
    unittest.main()
