from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from building_change.fusion import fuse_candidates, load_candidate_input, write_fusion_outputs


def _feature(candidate_id: int, west: float, south: float, east: float, north: float, classification: str) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "properties": {"candidate_id": candidate_id, "classification": classification},
    }


class CandidateFusionTests(unittest.TestCase):
    def test_fuses_only_agreeing_sources_and_retains_unconfirmed_candidates(self) -> None:
        local = {"type": "FeatureCollection", "features": [_feature(1, 145.4000, -36.3400, 145.4005, -36.3395, "likely_new_building"), _feature(2, 145.4100, -36.3400, 145.4105, -36.3395, "likely_building_change")]}
        remote = {"type": "FeatureCollection", "features": [_feature(8, 145.4001, -36.3401, 145.4006, -36.3396, "new_building")]}

        fused = fuse_candidates({"pixel": local, "segformer": remote}, merge_distance_m=2)

        self.assertEqual(len(fused["features"]), 2)
        agreed, unconfirmed = fused["features"]
        self.assertEqual(agreed["properties"]["source_count"], 2)
        self.assertEqual(agreed["properties"]["agreement"], "multi_source_agreement")
        self.assertEqual(set(agreed["properties"]["candidate_sources"]), {"pixel", "segformer"})
        self.assertEqual(unconfirmed["properties"]["source_count"], 1)
        self.assertEqual(unconfirmed["properties"]["agreement"], "single_source_candidate")

    def test_load_and_write_outputs_are_auditable(self) -> None:
        collection = {"type": "FeatureCollection", "features": [_feature(1, 145.4, -36.34, 145.401, -36.339, "new_building")]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.geojson"
            input_path.write_text(json.dumps(collection), encoding="utf-8")
            source, loaded = load_candidate_input(f"remote={input_path}")
            report = write_fusion_outputs(root / "output", {source: loaded})
            output = json.loads(Path(report["outputs"]["fused_candidates"]).read_text(encoding="utf-8"))

        self.assertEqual(output["metadata"]["input_sources"], ["remote"])
        self.assertTrue(output["metadata"]["single_source_candidates_are_retained"])


if __name__ == "__main__":
    unittest.main()
