"""Vision-model review of change candidates.

The geometry pipeline is good at finding *where* something changed and bad at
saying *what* changed: a solar array on an existing roof scores like a new
building. This sends the before/after crop pair for each candidate to a vision
model and records a category alongside the existing evidence.

The verdict is advisory review evidence. It is stored under a ``visual_review``
property and never overwrites ``classification``.

Providers are pluggable so a hosted test endpoint can be swapped for a
production one without touching the review logic.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)

# A model reply is untrusted input, so anything outside this set is rejected.
REVIEW_LABELS: tuple[str, ...] = (
    "new_building",
    "building_extension",
    "solar_panels",
    "hardscape",
    "vegetation",
    "no_visible_change",
    "unclear",
)

_PROMPT = """You are reviewing aerial imagery of the same location at two dates.
Image 1 is BEFORE ({before_date}). Image 2 is AFTER ({after_date}).
A candidate change region is outlined in magenta in both images.

Decide what changed INSIDE the outlined region. Choose exactly one category:
- new_building: a structure that did not exist before now stands there
- building_extension: an existing structure grew
- solar_panels: panels added to an existing roof
- hardscape: driveway, path, slab, pool, fence or other ground surface work
- vegetation: planting, clearing, growth or seasonal difference only
- no_visible_change: nothing meaningful differs
- unclear: too small, blurred or obscured to judge

Reply with ONLY a JSON object, no prose and no code fence:
{{"label": "<category>", "confidence": <0.0-1.0>, "reason": "<max 20 words>"}}"""


class VisualReviewError(ValueError):
    """Raised when candidate visual review cannot run."""


@dataclass(frozen=True)
class VisualReviewConfig:
    """Settings for rendering crops and querying the review provider."""

    crop_padding_m: float = 12.0
    max_crop_px: int = 448
    jpeg_quality: int = 82
    # Two base64 images must fit the endpoint's inline limit (NVIDIA: 180 kB).
    max_image_bytes: int = 62_000
    max_candidates: int | None = None
    min_area_m2: float = 0.0

    def validate(self) -> None:
        if self.crop_padding_m < 0:
            raise VisualReviewError("crop_padding_m cannot be negative.")
        if self.max_crop_px < 64:
            raise VisualReviewError("max_crop_px must be at least 64.")
        if not 1 <= self.jpeg_quality <= 100:
            raise VisualReviewError("jpeg_quality must be between 1 and 100.")
        if self.max_image_bytes < 4_000:
            raise VisualReviewError("max_image_bytes must be at least 4000.")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise VisualReviewError("max_candidates must be positive when set.")


# Verified against docs.api.nvidia.com Multimodal APIs. The smaller two are the
# sensible free-tier choices; the 90b is the same API but much heavier.
VISION_MODELS: tuple[str, ...] = (
    "meta/llama-3.2-11b-vision-instruct",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "meta/llama-3.2-90b-vision-instruct",
)


@dataclass(frozen=True)
class ProviderConfig:
    """Connection settings for an NVIDIA NIM or OpenAI-compatible vision endpoint."""

    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "meta/llama-3.2-11b-vision-instruct"
    api_key_env: str = "NVIDIA_API_KEY"
    # NIM multimodal models are invoked at /{model}; OpenAI-style routes use
    # /chat/completions with the model named in the body.
    use_model_path: bool = True
    # NIM answers 202 plus an NVCF-REQID header when a result is not ready yet.
    poll_base_url: str = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status"
    poll_interval_s: float = 2.0
    poll_timeout_s: float = 180.0
    timeout_s: float = 90.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 200

    def validate(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise VisualReviewError("temperature must be between 0 and 2.")
        if not 1 <= self.max_tokens <= 8192:
            raise VisualReviewError("max_tokens must be between 1 and 8192.")
        if self.poll_interval_s <= 0 or self.poll_timeout_s <= 0:
            raise VisualReviewError("Polling interval and timeout must be positive.")

    def invoke_url(self) -> str:
        base = self.base_url.rstrip("/")
        return f"{base}/{self.model}" if self.use_model_path else f"{base}/chat/completions"

    def resolve_key(self) -> str:
        key = os.getenv(self.api_key_env, "").strip()
        if not key:
            raise VisualReviewError(
                f"No API key found in ${self.api_key_env}. Add it to .env; it is never written to outputs."
            )
        return key


class ReviewProvider(Protocol):
    """Returns the raw model reply for one before/after image pair."""

    def classify(self, before_jpeg: bytes, after_jpeg: bytes, prompt: str) -> str: ...


class OpenAICompatibleProvider:
    """Vision review via an NVIDIA NIM or OpenAI-compatible chat endpoint."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()
        self.config.validate()

    @staticmethod
    def _content(reply: dict[str, Any]) -> str:
        return reply["choices"][0]["message"]["content"]

    def _poll(self, session: Any, request_id: str) -> str:
        """Follow a NIM 202 until the result is ready."""
        import time

        deadline = time.monotonic() + self.config.poll_timeout_s
        url = f"{self.config.poll_base_url.rstrip('/')}/{request_id}"
        while time.monotonic() < deadline:
            time.sleep(self.config.poll_interval_s)
            response = session.get(
                url,
                headers={"Authorization": f"Bearer {self.config.resolve_key()}", "Accept": "application/json"},
                timeout=self.config.timeout_s,
            )
            if response.status_code == 202:
                continue
            if response.status_code >= 400:
                raise VisualReviewError(f"Polling request {request_id} returned HTTP {response.status_code}.")
            return self._content(response.json())
        raise VisualReviewError(f"Request {request_id} did not complete within {self.config.poll_timeout_s:.0f}s.")

    def classify(self, before_jpeg: bytes, after_jpeg: bytes, prompt: str) -> str:
        import requests

        def part(payload: bytes) -> dict[str, Any]:
            encoded = base64.b64encode(payload).decode("ascii")
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}

        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        part(before_jpeg),
                        part(after_jpeg),
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.config.resolve_key()}",
            "Accept": "application/json",
        }
        last_error: Exception | None = None
        with requests.Session() as session:
            for _ in range(self.config.max_retries + 1):
                try:
                    response = session.post(
                        self.config.invoke_url(),
                        json=body,
                        headers=headers,
                        timeout=self.config.timeout_s,
                    )
                except Exception as exc:
                    last_error = exc
                    continue

                if response.status_code == 202:
                    request_id = response.headers.get("NVCF-REQID")
                    if not request_id:
                        last_error = VisualReviewError("Endpoint returned 202 without an NVCF-REQID header.")
                        continue
                    return self._poll(session, request_id)

                if response.status_code >= 400:
                    # Status only: the body can echo back the submitted request content.
                    last_error = VisualReviewError(
                        f"Review endpoint returned HTTP {response.status_code} for model {self.config.model!r}."
                    )
                    if response.status_code in (400, 401, 403, 404, 422):
                        raise last_error
                    continue
                try:
                    return self._content(response.json())
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    last_error = exc
        raise VisualReviewError(f"Review request failed after {self.config.max_retries + 1} attempts: {last_error}")


def parse_verdict(reply: str) -> dict[str, Any]:
    """Validate an untrusted model reply into a known label, or mark it unusable."""
    text = (reply or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"label": "unclear", "confidence": 0.0, "reason": "unparsable reply", "valid": False}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"label": "unclear", "confidence": 0.0, "reason": "unparsable reply", "valid": False}
    if not isinstance(payload, dict):
        return {"label": "unclear", "confidence": 0.0, "reason": "unparsable reply", "valid": False}

    label = payload.get("label")
    if not isinstance(label, str) or label not in REVIEW_LABELS:
        return {"label": "unclear", "confidence": 0.0, "reason": f"unknown label {label!r}", "valid": False}
    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = payload.get("reason")
    reason = reason[:200] if isinstance(reason, str) else ""
    return {"label": label, "confidence": round(confidence, 3), "reason": reason, "valid": True}


def _stretch(window_data: np.ndarray) -> np.ndarray:
    result = np.zeros(window_data.shape[1:] + (3,), dtype=np.uint8)
    for band in range(3):
        values = window_data[band][window_data[band] > 0]
        if values.size:
            low, high = np.percentile(values, (2, 98))
            result[:, :, band] = np.clip(
                (window_data[band] - low) * 255.0 / max(high - low, 1.0), 0, 255
            ).astype(np.uint8)
    return result


def render_crop(
    dataset: Any,
    geometry: Any,
    config: VisualReviewConfig,
) -> bytes | None:
    """Render a JPEG crop around a candidate with its outline drawn on top."""
    west, south, east, north = geometry.bounds
    pad = config.crop_padding_m
    window = dataset.window(west - pad, south - pad, east + pad, north + pad)
    data = dataset.read((1, 2, 3), window=window, boundless=True, fill_value=0)
    if data.size == 0 or data.shape[1] < 8 or data.shape[2] < 8:
        return None

    image = _stretch(data)
    transform = dataset.window_transform(window)
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        xs, ys = polygon.exterior.xy
        rows, cols = rasterio.transform.rowcol(transform, list(xs), list(ys))
        points = np.column_stack([np.asarray(cols), np.asarray(rows)]).astype(np.int32)
        cv2.polylines(image, [points.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 255), thickness=2)

    longest = max(image.shape[0], image.shape[1])
    if longest > config.max_crop_px:
        scale = config.max_crop_px / longest
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    return _encode_within_budget(image, config)


def _encode_within_budget(image: np.ndarray, config: VisualReviewConfig) -> bytes | None:
    """Encode to JPEG, stepping quality down until the byte budget is met."""
    quality = config.jpeg_quality
    while quality >= 40:
        ok, buffer = cv2.imencode(".jpg", image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        if buffer.nbytes <= config.max_image_bytes:
            return buffer.tobytes()
        quality -= 10
    # Still too large at the quality floor, so shrink the crop instead.
    smaller = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
    if smaller.shape[0] < 64 or smaller.shape[1] < 64:
        return None
    return _encode_within_budget(smaller, config)


def review_candidates(
    candidates: dict[str, Any],
    before_image: str | Path,
    after_image: str | Path,
    provider: ReviewProvider,
    *,
    before_date: str = "earlier date",
    after_date: str = "later date",
    config: VisualReviewConfig | None = None,
) -> dict[str, Any]:
    """Attach a vision-model verdict to each candidate feature."""
    active = config or VisualReviewConfig()
    active.validate()

    features = candidates.get("features")
    if candidates.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise VisualReviewError("Candidates must be a GeoJSON FeatureCollection.")

    selected = [f for f in features if float(f.get("properties", {}).get("area_m2", 0)) >= active.min_area_m2]
    selected.sort(key=lambda f: -float(f.get("properties", {}).get("area_m2", 0)))
    if active.max_candidates is not None:
        selected = selected[: active.max_candidates]

    prompt = _PROMPT.format(before_date=before_date, after_date=after_date)
    counts: dict[str, int] = {}
    reviewed = 0
    failed = 0

    with rasterio.open(before_image) as before_ds, rasterio.open(after_image) as after_ds:
        target_crs = after_ds.crs
        for index, feature in enumerate(selected, start=1):
            geometry = shape(transform_geom("EPSG:4326", target_crs, feature["geometry"]))
            before_jpeg = render_crop(before_ds, geometry, active)
            after_jpeg = render_crop(after_ds, geometry, active)
            if before_jpeg is None or after_jpeg is None:
                failed += 1
                continue
            try:
                reply = provider.classify(before_jpeg, after_jpeg, prompt)
            except VisualReviewError:
                raise
            except Exception as exc:
                logger.warning("Review failed for candidate %s: %s", feature.get("properties", {}).get("candidate_id"), exc)
                failed += 1
                continue

            verdict = parse_verdict(reply)
            feature.setdefault("properties", {})["visual_review"] = {
                "label": verdict["label"],
                "confidence": verdict["confidence"],
                "reason": verdict["reason"],
                "parsed": verdict["valid"],
                "evidence_role": "advisory_visual_review",
            }
            counts[verdict["label"]] = counts.get(verdict["label"], 0) + 1
            reviewed += 1
            if index % 10 == 0:
                logger.info("Reviewed %d/%d candidates", index, len(selected))

    return {
        "candidates_total": len(features),
        "candidates_reviewed": reviewed,
        "candidates_failed": failed,
        "label_counts": dict(sorted(counts.items(), key=lambda item: -item[1])),
        "evidence_role": "advisory_visual_review",
        "warning": "Vision-model labels are advisory review evidence and do not change candidate classification.",
    }
