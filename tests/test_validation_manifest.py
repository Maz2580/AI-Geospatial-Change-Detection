from __future__ import annotations

import copy
from pathlib import Path
import unittest

from building_change.validation import ValidationManifestError, load_validation_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "benchmarks" / "validation_manifest.json"


class ValidationManifestTests(unittest.TestCase):
    def test_committed_pilot_evidence_is_locked_and_valid(self) -> None:
        report = validate_manifest(load_validation_manifest(MANIFEST), root=ROOT)
        self.assertEqual(report["benchmark"], "victoria_building_change_pilot_evidence")
        self.assertEqual(report["case_count"], 3)
        self.assertEqual(report["verified_artifact_count"], 8)

    def test_rejects_silent_artifact_hash_change(self) -> None:
        manifest = copy.deepcopy(load_validation_manifest(MANIFEST))
        manifest["cases"][0]["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValidationManifestError, "changed"):
            validate_manifest(manifest, root=ROOT)

    def test_rejects_non_pilot_split_until_complete_holdout_labels_exist(self) -> None:
        manifest = copy.deepcopy(load_validation_manifest(MANIFEST))
        manifest["cases"][0]["split"] = "holdout"
        with self.assertRaisesRegex(ValidationManifestError, "must currently be a pilot"):
            validate_manifest(manifest, root=ROOT)


if __name__ == "__main__":
    unittest.main()
