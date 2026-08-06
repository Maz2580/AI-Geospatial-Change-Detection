from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from building_change.detector import DetectionConfig, run_detection
from building_change.nearmap import NearmapApiError, Survey, select_surveys


class NearmapSelectionTests(unittest.TestCase):
    def test_selects_deterministic_before_and_after_surveys(self) -> None:
        surveys = [Survey("s3", date(2025, 8, 1), {}), Survey("s1", date(2021, 2, 1), {}), Survey("s2", date(2023, 3, 1), {})]
        before, after = select_surveys(surveys, before_date="2021-12-31", after_date="2024-01-01")
        self.assertEqual(before.id, "s1")
        self.assertEqual(after.id, "s3")

    def test_rejects_same_survey_pair(self) -> None:
        with self.assertRaises(NearmapApiError):
            select_surveys([Survey("s1", date(2021, 2, 1), {})], before_date="2021-12-31")


class DetectionTests(unittest.TestCase):
    @staticmethod
    def _write_tif(path: Path, array: np.ndarray) -> None:
        with rasterio.open(path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1], count=array.shape[0], dtype=array.dtype, crs="EPSG:3857", transform=from_origin(0, 100, 1, 1)) as target:
            target.write(array)

    def test_extracts_building_candidate_and_height_gain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_rgb = np.full((3, 100, 100), 80, dtype=np.uint8)
            after_rgb = before_rgb.copy()
            after_rgb[:, 30:60, 35:70] = 200
            before_dsm = np.zeros((1, 100, 100), dtype=np.float32)
            after_dsm = before_dsm.copy()
            after_dsm[:, 30:60, 35:70] = 4.0
            before_path, after_path = root / "before.tif", root / "after.tif"
            before_dsm_path, after_dsm_path = root / "before_dsm.tif", root / "after_dsm.tif"
            self._write_tif(before_path, before_rgb)
            self._write_tif(after_path, after_rgb)
            self._write_tif(before_dsm_path, before_dsm)
            self._write_tif(after_dsm_path, after_dsm)
            report = run_detection(
                before_path, after_path, root / "output", before_dsm=before_dsm_path, after_dsm=after_dsm_path,
                config=DetectionConfig(change_threshold=0.05, morphology_m=0, min_area_m2=20, enable_registration=False),
            )
            self.assertGreaterEqual(report["candidate_count"], 1)
            candidates = json.loads(Path(report["outputs"]["construction_candidates"]).read_text())
            classifications = [feature["properties"]["classification"] for feature in candidates["features"]]
            self.assertIn("likely_new_building", classifications)

    def test_defers_shadow_only_change_to_uncertain_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_rgb = np.full((3, 100, 100), 180, dtype=np.uint8)
            # Stable low/high reference strips keep per-image normalisation
            # unchanged, emulating a real image with tonal variation.
            before_rgb[:, :5, :] = 0
            before_rgb[:, 5:10, :] = 255
            after_rgb = before_rgb.copy()
            # A dark, locally isolated patch in the older image emulates a cast shadow.
            before_rgb[:, 35:55, 35:55] = 50
            before_path, after_path = root / "before.tif", root / "after.tif"
            self._write_tif(before_path, before_rgb)
            self._write_tif(after_path, after_rgb)
            report = run_detection(
                before_path, after_path, root / "output",
                config=DetectionConfig(change_threshold=0.05, morphology_m=0, min_area_m2=20, enable_registration=False),
            )
            self.assertEqual(report["candidate_count"], 0)
            self.assertGreaterEqual(report["uncertain_shadow_candidate_count"], 1)
            uncertain = json.loads(Path(report["outputs"]["uncertain_shadow_candidates"]).read_text())
            self.assertEqual(uncertain["features"][0]["properties"]["classification"], "uncertain_shadow")


if __name__ == "__main__":
    unittest.main()
