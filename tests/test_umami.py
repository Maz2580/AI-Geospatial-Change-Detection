from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from building_change.umami import UmamiAnalysisRequest, UmamiBoundingBox, UmamiClient, write_analysis_outputs


class _Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class _Session:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict, dict]] = []
        self.status_calls = 0

    def post(self, url: str, *, json: dict, headers: dict, timeout: float) -> _Response:
        self.posts.append((url, json, headers))
        if url.endswith("/api/auth/token"):
            return _Response(200, {"access_token": "not-a-real-token"})
        return _Response(202, {"job_id": "job-123"})

    def get(self, url: str, *, headers: dict, timeout: float) -> _Response:
        self.status_calls += 1
        return _Response(
            200,
            {
                "status": "done",
                "result": {
                    "type": "FeatureCollection",
                    "summary": {"model": "dfine"},
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[145.4, -36.34], [145.401, -36.34], [145.401, -36.339], [145.4, -36.34]]],
                            },
                            "properties": {"vlm_type": "new_building", "relevant": True},
                        },
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[145.41, -36.34], [145.411, -36.34], [145.411, -36.339], [145.41, -36.34]]],
                            },
                            "properties": {"clip_type": "vehicle", "relevant": False},
                        },
                    ],
                },
            },
        )


class UmamiClientTests(unittest.TestCase):
    def test_submits_short_bbox_keys_and_writes_secret_free_outputs(self) -> None:
        session = _Session()
        request = UmamiAnalysisRequest(
            bbox=UmamiBoundingBox(145.4, -36.34, 145.42, -36.32),
            before_date="2022-07-30",
            after_date="2026-04-18",
            detector="segformer",
        )
        client = UmamiClient("https://example.test", "user", "password", session=session)
        job_id = client.start_analysis(request)
        completed = client.wait_for_analysis(job_id, poll_seconds=1, timeout_seconds=5)

        self.assertEqual(job_id, "job-123")
        self.assertEqual(session.posts[1][1]["bbox"], {"w": 145.4, "s": -36.34, "e": 145.42, "n": -36.32})
        self.assertEqual(session.posts[1][1]["detector"], "segformer")

        with tempfile.TemporaryDirectory() as directory:
            report = write_analysis_outputs(directory, request, job_id, completed)
            all_candidates = json.loads(Path(report["all_candidates"]).read_text(encoding="utf-8"))
            relevant_candidates = json.loads(Path(report["relevant_candidates"]).read_text(encoding="utf-8"))
            written_report = Path(report["report"]).read_text(encoding="utf-8")

        self.assertEqual(len(all_candidates["features"]), 2)
        self.assertEqual(len(relevant_candidates["features"]), 1)
        self.assertEqual(all_candidates["features"][0]["properties"]["candidate_source"], "umami")
        self.assertEqual(all_candidates["features"][0]["properties"]["classification"], "new_building")
        self.assertNotIn("password", written_report)
        self.assertNotIn("not-a-real-token", written_report)


if __name__ == "__main__":
    unittest.main()
