"""Optional UMAMI Encroachment API client.

The remote service is treated as another candidate source, never as ground
truth.  Credentials are read by the CLI from the local environment and are
never written to reports or GeoJSON files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import requests


class UmamiApiError(RuntimeError):
    """Raised when UMAMI cannot accept or complete an analysis request."""


@dataclass(frozen=True)
class UmamiBoundingBox:
    """WGS84 bounds, converted to the service's required short-key payload."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not self.west < self.east:
            raise ValueError("west must be less than east.")
        if not self.south < self.north:
            raise ValueError("south must be less than north.")

    def as_service_payload(self) -> dict[str, float]:
        # The worker expects w/s/e/n even though the published OpenAPI schema
        # currently describes bbox only as a generic object.
        return {"w": self.west, "s": self.south, "e": self.east, "n": self.north}


@dataclass(frozen=True)
class UmamiAnalysisRequest:
    bbox: UmamiBoundingBox
    before_date: str
    after_date: str
    detector: str = "detector_v5"
    mode: str = "change"
    long_side: int = 768
    percentile: float = 90.0
    sam_refine: bool = True
    regularize: bool = True
    regularize_method: str = "hybrid"

    def __post_init__(self) -> None:
        valid_detectors = {"detector_v5", "v5", "dfine_v5", "dfine", "segformer", "maskrcnn"}
        if self.detector not in valid_detectors:
            raise ValueError(f"detector must be one of: {', '.join(sorted(valid_detectors))}.")
        if self.mode not in {"change", "new_buildings", "object_change", "extract_footprints"}:
            raise ValueError("Unsupported UMAMI analysis mode.")
        if self.long_side < 128:
            raise ValueError("long_side must be at least 128 pixels.")
        if not 0 < self.percentile < 100:
            raise ValueError("percentile must be between 0 and 100.")

    def as_payload(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.as_service_payload(),
            "date_t1": self.before_date,
            "date_t2": self.after_date,
            "detector": self.detector,
            "mode": self.mode,
            "long_side": self.long_side,
            "percentile": self.percentile,
            "sam_refine": self.sam_refine,
            "regularize": self.regularize,
            "regularize_method": self.regularize_method,
        }


class UmamiClient:
    """Small authenticated client for the Encroachment analysis endpoints."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("UMAMI base URL is required.")
        if not username or not password:
            raise ValueError("UMAMI_USERNAME and UMAMI_PASSWORD are required.")
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self._token: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _payload(response: Any, context: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise UmamiApiError(f"{context} returned non-JSON HTTP {response.status_code}.") from exc
        if response.status_code >= 400:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            raise UmamiApiError(f"{context} failed with HTTP {response.status_code}: {detail}")
        if not isinstance(payload, dict):
            raise UmamiApiError(f"{context} returned an unexpected JSON payload.")
        return payload

    def authenticate(self) -> None:
        response = self.session.post(
            self._url("/api/auth/token"),
            json={"username": self.username, "password": self.password},
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
        )
        payload = self._payload(response, "UMAMI login")
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise UmamiApiError("UMAMI login succeeded but did not return an access token.")
        self._token = token

    def _headers(self, *, force_refresh: bool = False) -> dict[str, str]:
        if self._token is None or force_refresh:
            self.authenticate()
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    def _request(self, method: str, path: str, **kwargs) -> "requests.Response":
        """Make an authenticated request, retrying once on 401 (expired token)."""
        kwargs.setdefault("timeout", self.timeout_seconds)
        response = self.session.request(
            method, self._url(path),
            headers={**self._headers(), **kwargs.pop("headers", {})},
            **kwargs,
        )
        if response.status_code == 401:
            response = self.session.request(
                method, self._url(path),
                headers={**self._headers(force_refresh=True), **kwargs.pop("headers", {})},
                **kwargs,
            )
        return response

    def start_analysis(self, request: UmamiAnalysisRequest) -> str:
        response = self._request(
            "POST", "/api/encroachment/analyze",
            json=request.as_payload(),
            headers={"Content-Type": "application/json"},
        )
        payload = self._payload(response, "UMAMI analysis submission")
        job_id = payload.get("job_id")
        if response.status_code != 202 or not isinstance(job_id, str) or not job_id:
            raise UmamiApiError("UMAMI analysis was not accepted as an asynchronous job.")
        return job_id

    def get_analysis_status(self, job_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/api/encroachment/analyze/{job_id}")
        return self._payload(response, "UMAMI analysis status")

    def wait_for_analysis(
        self,
        job_id: str,
        *,
        poll_seconds: float = 5.0,
        timeout_seconds: float = 300.0,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> dict[str, Any]:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive.")
        deadline = monotonic() + timeout_seconds
        while True:
            status = self.get_analysis_status(job_id)
            state = status.get("status")
            if state == "done":
                if not isinstance(status.get("result"), dict):
                    raise UmamiApiError("UMAMI marked the job done without a result.")
                return status
            if state == "error":
                raise UmamiApiError(f"UMAMI analysis failed: {status.get('error') or 'unknown error'}")
            if state != "running":
                raise UmamiApiError(f"UMAMI returned unknown job state: {state!r}")
            if monotonic() >= deadline:
                raise UmamiApiError(f"UMAMI analysis {job_id} is still running after {timeout_seconds:g} seconds.")
            sleep(min(poll_seconds, max(0.0, deadline - monotonic())))


def normalise_result(result: dict[str, Any], request: UmamiAnalysisRequest) -> dict[str, Any]:
    """Return an analysis FeatureCollection with stable local provenance fields."""
    if result.get("type") != "FeatureCollection" or not isinstance(result.get("features"), list):
        raise UmamiApiError("UMAMI result is not a GeoJSON FeatureCollection.")
    features: list[dict[str, Any]] = []
    for index, source_feature in enumerate(result["features"], start=1):
        if not isinstance(source_feature, dict) or source_feature.get("type") != "Feature":
            continue
        feature = json.loads(json.dumps(source_feature))
        properties = feature.setdefault("properties", {})
        if not isinstance(properties, dict):
            properties = feature["properties"] = {}
        properties.setdefault("candidate_id", index)
        properties.setdefault(
            "classification",
            properties.get("vlm_type") or properties.get("clip_type") or properties.get("label") or "remote_change_candidate",
        )
        properties["candidate_source"] = "umami"
        properties["requested_detector"] = request.detector
        properties["analysis_mode"] = request.mode
        features.append(feature)
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "candidate_source": "umami",
            "requested_detector": request.detector,
            "analysis_mode": request.mode,
            "pipeline_summary": result.get("summary", {}),
        },
    }


def write_analysis_outputs(
    output_dir: str | Path,
    request: UmamiAnalysisRequest,
    job_id: str,
    completed_status: dict[str, Any],
) -> dict[str, str | int]:
    """Write raw and UI-equivalent relevant candidates without storing credentials."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    collection = normalise_result(completed_status["result"], request)
    relevant = [feature for feature in collection["features"] if feature.get("properties", {}).get("relevant") is True]
    stem = f"umami_{request.detector}_{request.mode}"
    all_path = directory / f"{stem}_all_candidates.geojson"
    relevant_path = directory / f"{stem}_relevant_candidates.geojson"
    report_path = directory / f"{stem}_report.json"
    all_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    relevant_path.write_text(json.dumps({**collection, "features": relevant}, indent=2), encoding="utf-8")
    report = {
        "job_id": job_id,
        "request": {**asdict(request), "bbox": asdict(request.bbox)},
        "candidate_count": len(collection["features"]),
        "relevant_candidate_count": len(relevant),
        "pipeline_summary": collection["metadata"]["pipeline_summary"],
        "outputs": {"all_candidates": str(all_path), "relevant_candidates": str(relevant_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"job_id": job_id, "candidate_count": len(collection["features"]), "relevant_candidate_count": len(relevant), "all_candidates": str(all_path), "relevant_candidates": str(relevant_path), "report": str(report_path)}
