from __future__ import annotations

import unittest

from shapely.affinity import translate
from shapely.geometry import Polygon, box

from building_change.outline_metrics import (
    OutlineMetricError,
    boundary_f1,
    hausdorff_95,
    intersection_over_union,
    match_instances,
    score_pair,
    summarise,
    vertex_count,
)


class BoundaryMetricTests(unittest.TestCase):
    def test_identical_outlines_score_perfectly(self) -> None:
        roof = box(0, 0, 10, 20)
        self.assertEqual(boundary_f1(roof, roof, 0.25), 1.0)
        self.assertEqual(hausdorff_95(roof, roof), 0.0)

    def test_boundary_f1_separates_what_iou_hides(self) -> None:
        """A metre of offset barely moves IoU but must collapse boundary F1.

        This is the whole reason the metric exists: published IoU stays in the
        range that reads as success while every edge is a metre off the roof.
        """
        roof = box(0, 0, 10, 20)
        offset = translate(roof, 1.0, 0)
        self.assertGreater(intersection_over_union(offset, roof), 0.8)
        self.assertLess(boundary_f1(offset, roof, 0.25), 0.4)

    def test_tolerance_controls_what_counts_as_agreement(self) -> None:
        roof = box(0, 0, 10, 20)
        offset = translate(roof, 0.5, 0)
        self.assertLess(boundary_f1(offset, roof, 0.25), 0.5)
        self.assertEqual(boundary_f1(offset, roof, 1.0), 1.0)

    def test_hausdorff_95_ignores_a_single_spur(self) -> None:
        """Threshold-contour outlines grow spurious tabs; one must not dominate."""
        roof = box(0, 0, 20, 20)
        spur = Polygon([(0, 0), (20, 0), (20, 20), (0, 20), (0, 10.2), (-3, 10.2), (-3, 9.8), (0, 9.8)])
        self.assertLess(hausdorff_95(spur, roof), 3.0)

    def test_vertices_are_counted_including_holes(self) -> None:
        courtyard = Polygon(box(0, 0, 20, 20).exterior.coords, [box(5, 5, 10, 10).exterior.coords])
        self.assertEqual(vertex_count(courtyard), 10)

    def test_boundary_sampling_ignores_vertex_density(self) -> None:
        """A staircase and a clean rectangle on the same edge must score alike."""
        roof = box(0, 0, 10, 10)
        dense = Polygon(roof.exterior).segmentize(0.05)
        self.assertGreater(boundary_f1(dense, roof, 0.25), 0.99)

    def test_empty_geometry_is_rejected(self) -> None:
        with self.assertRaises(OutlineMetricError):
            score_pair(Polygon(), box(0, 0, 1, 1))


class MatchingTests(unittest.TestCase):
    def test_a_sloppy_outline_is_a_match_not_a_miss(self) -> None:
        """IoU 0.5 would call this a miss plus a false alarm; it is one shed."""
        shed = box(0, 0, 3.2, 3.2)  # 10 m2, the gold set's stated draw threshold
        sloppy = box(-1, -1, 4.2, 4.2)  # the same shed outlined a metre too large
        self.assertLess(intersection_over_union(sloppy, shed), 0.5)
        result = match_instances([sloppy], [shed])
        self.assertEqual(result.matches, [(0, 0)])
        self.assertEqual(result.unmatched_labels, [])
        self.assertEqual(result.unmatched_predictions, [])

    def test_iou_matching_would_punish_small_buildings_for_shared_error(self) -> None:
        """One metre of error: fatal to a shed under IoU, invisible on a warehouse.

        Sheds, garages and carports are the structures encroachment detection
        exists to catch, so an IoU gate would excuse the same absolute error on
        the buildings that matter least and condemn it on the ones that matter
        most. Overlap-of-the-smaller treats them alike.
        """
        for side in (3.2, 14.1, 77.0):
            roof = box(0, 0, side, side)
            sloppy = box(-1, -1, side + 1, side + 1)
            result = match_instances([sloppy], [roof])
            self.assertEqual(result.matches, [(0, 0)], f"{side} m roof")
        self.assertLess(intersection_over_union(box(-1, -1, 4.2, 4.2), box(0, 0, 3.2, 3.2)), 0.5)
        self.assertGreater(intersection_over_union(box(-1, -1, 78, 78), box(0, 0, 77, 77)), 0.9)

    def test_a_merged_blob_is_reported_as_a_merge(self) -> None:
        units = [box(0, 0, 10, 10), box(11, 0, 21, 10), box(22, 0, 32, 10)]
        blob = box(0, 0, 32, 10)
        result = match_instances([blob], units)
        self.assertEqual(result.merged_predictions, [0])
        self.assertEqual(len(result.correspondences), 3)
        # Only one boundary exists, so it is scored once, not three times.
        self.assertEqual(len(result.matches), 1)
        # The other two roofs are covered, so they are merged, never misses.
        self.assertEqual(result.unmatched_labels, [])

    def test_a_split_roof_is_reported_as_a_split(self) -> None:
        roof = box(0, 0, 30, 10)
        pieces = [box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10)]
        result = match_instances(pieces, [roof])
        self.assertEqual(result.split_labels, [0])
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.unmatched_predictions, [])

    def test_a_missed_roof_and_a_false_alarm_are_kept_apart(self) -> None:
        result = match_instances([box(100, 100, 110, 110)], [box(0, 0, 10, 10)])
        self.assertEqual(result.matches, [])
        self.assertEqual(result.unmatched_labels, [0])
        self.assertEqual(result.unmatched_predictions, [0])

    def test_a_neighbouring_roof_does_not_correspond(self) -> None:
        """Clipping a neighbour's edge must not count as finding it."""
        result = match_instances([box(0, 0, 10, 10)], [box(9, 0, 19, 10)])
        self.assertEqual(result.matches, [])
        self.assertEqual(result.unmatched_labels, [0])

    def test_an_invalid_overlap_fraction_is_rejected(self) -> None:
        with self.assertRaises(OutlineMetricError):
            match_instances([box(0, 0, 1, 1)], [box(0, 0, 1, 1)], min_overlap_fraction=0)


class SummaryTests(unittest.TestCase):
    def test_summary_of_nothing_reports_nothing(self) -> None:
        self.assertEqual(summarise([]), {"instance_count": 0})

    def test_summary_uses_medians_so_one_warehouse_cannot_dominate(self) -> None:
        roof = box(0, 0, 10, 10)
        warehouse = box(0, 0, 80, 80)
        scores = [
            score_pair(roof, roof),
            score_pair(roof, roof),
            score_pair(translate(warehouse, 5, 0), warehouse),
        ]
        summary = summarise(scores)
        self.assertEqual(summary["instance_count"], 3)
        self.assertEqual(summary["median_boundary_f1"]["0.25"], 1.0)


if __name__ == "__main__":
    unittest.main()
