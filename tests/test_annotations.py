from __future__ import annotations

import unittest

from building_change.annotations import AnnotationError, create_reference_label_template, validate_reference_label_document


def _reference() -> dict:
    return {
        "type": "FeatureCollection",
        "metadata": {"source": "official reference"},
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[144.0, -37.0], [144.1, -37.0], [144.1, -37.1], [144.0, -37.1], [144.0, -37.0]]]},
                "properties": {"candidate_id": 3},
            }
        ],
    }


class AnnotationTemplateTests(unittest.TestCase):
    def _template(self) -> dict:
        return create_reference_label_template(
            case_id="independent-case",
            reference_collection=_reference(),
            before_date="2020-01-01",
            after_date="2023-01-01",
            reference_path="data/benchmarks/case/reference.geojson",
        )

    def test_template_keeps_reference_geometry_out_of_roof_labels(self) -> None:
        document = self._template()
        report = validate_reference_label_document(document)
        self.assertEqual(report["unreviewed_reference_count"], 1)
        self.assertEqual(report["after_roof_label_count"], 0)
        self.assertFalse(report["complete"])

    def test_complete_review_requires_valid_change_type(self) -> None:
        document = self._template()
        document["labels"][0]["assessment"] = "real_visible_change"
        document["labels"][0]["visible_change_type"] = "new_building"
        report = validate_reference_label_document(document, require_complete=True)
        self.assertEqual(report["visible_permanent_change_count"], 1)

    def test_mapping_only_review_cannot_claim_a_visible_change_type(self) -> None:
        document = self._template()
        document["labels"][0].update({"assessment": "mapping_only_or_not_visible", "visible_change_type": "new_building"})
        with self.assertRaisesRegex(AnnotationError, "not_applicable"):
            validate_reference_label_document(document, require_complete=True)

    def test_complete_occluded_review_is_excluded_from_visible_change_counts(self) -> None:
        document = self._template()
        document["labels"][0].update(
            {
                "assessment": "inconclusive_due_occlusion",
                "visible_change_type": "not_applicable",
                "review_notes": "Dense tree cover obscures the decisive before-date roof edge.",
            }
        )
        report = validate_reference_label_document(document, require_complete=True)
        self.assertEqual(report["visible_permanent_change_count"], 0)
        self.assertEqual(report["inconclusive_reference_count"], 1)

    def test_inconclusive_review_requires_an_explanation(self) -> None:
        document = self._template()
        document["labels"][0].update(
            {"assessment": "inconclusive_due_occlusion", "visible_change_type": "not_applicable"}
        )
        with self.assertRaisesRegex(AnnotationError, "review_notes"):
            validate_reference_label_document(document, require_complete=True)


if __name__ == "__main__":
    unittest.main()
