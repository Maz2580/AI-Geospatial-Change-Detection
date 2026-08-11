"""Review construction candidates with NVIDIA NIM using one comparison image.

NVIDIA's current hosted multimodal endpoint accepts one data-URI image per
request reliably, while a pair of image parts returns HTTP 400. This runner
places the dated crops side-by-side in one JPEG and keeps the model verdict
strictly advisory alongside the geometry pipeline's existing classification.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any


def _prefer_rasterio_projection_data() -> None:
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    bundled_data = Path(spec.origin).parent / "proj_data"
    if bundled_data.is_dir():
        os.environ["PROJ_DATA"] = str(bundled_data)
        os.environ.pop("PROJ_LIB", None)


_prefer_rasterio_projection_data()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from building_change.visual_review import (  # noqa: E402
    VISION_MODELS,
    ProviderConfig,
    VisualReviewConfig,
    VisualReviewError,
    review_candidates,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _comparison_jpeg(before_jpeg: bytes, after_jpeg: bytes, max_bytes: int = 110_000) -> bytes:
    before = cv2.imdecode(np.frombuffer(before_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    after = cv2.imdecode(np.frombuffer(after_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if before is None or after is None:
        raise VisualReviewError("Could not decode a rendered candidate crop.")
    height = max(before.shape[0], after.shape[0])

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    before, after = resize_to_height(before), resize_to_height(after)
    header, gap = 28, 4
    composite = np.full((height + header, before.shape[1] + after.shape[1] + gap, 3), 255, dtype=np.uint8)
    composite[header:, : before.shape[1]] = before
    composite[header:, before.shape[1] + gap :] = after
    cv2.putText(composite, "BEFORE", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(composite, "AFTER", (before.shape[1] + gap + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    quality = 82
    image = composite
    while True:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise VisualReviewError("Could not encode the side-by-side comparison image.")
        if encoded.nbytes <= max_bytes:
            return encoded.tobytes()
        if quality > 40:
            quality -= 10
            continue
        if min(image.shape[:2]) < 96:
            raise VisualReviewError("Comparison image could not be made small enough for NVIDIA NIM.")
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
        quality = 82


class NvidiaCompositeProvider:
    """NVIDIA NIM provider that sends one labelled before/after comparison image."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.config.validate()

    def classify(self, before_jpeg: bytes, after_jpeg: bytes, prompt: str) -> str:
        import requests

        comparison = _comparison_jpeg(before_jpeg, after_jpeg)
        prompt = prompt.replace("Image 1 is BEFORE", "The LEFT half is BEFORE").replace("Image 2 is AFTER", "The RIGHT half is AFTER")
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(comparison).decode("ascii")}},
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.config.resolve_key()}", "Accept": "application/json"}
        last_error: Exception | None = None
        with requests.Session() as session:
            for _ in range(self.config.max_retries + 1):
                try:
                    response = session.post(self.config.invoke_url(), json=body, headers=headers, timeout=self.config.timeout_s)
                except Exception as exc:
                    last_error = exc
                    continue
                if response.status_code >= 400:
                    last_error = VisualReviewError(f"NVIDIA review returned HTTP {response.status_code} for model {self.config.model!r}.")
                    if response.status_code in (400, 401, 403, 404, 422):
                        raise last_error
                    continue
                try:
                    return response.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    last_error = exc
        raise VisualReviewError(f"NVIDIA review failed after {self.config.max_retries + 1} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-date", default="earlier date")
    parser.add_argument("--after-date", default="later date")
    parser.add_argument("--model", default=VISION_MODELS[0], choices=VISION_MODELS)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--min-area-m2", type=float, default=0.0)
    parser.add_argument("--crop-padding-m", type=float, default=12.0)
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    collection = json.loads(args.candidates.read_text(encoding="utf-8"))
    provider = NvidiaCompositeProvider(ProviderConfig(model=args.model, use_model_path=False))
    report = review_candidates(
        collection,
        args.before,
        args.after,
        provider,
        before_date=args.before_date,
        after_date=args.after_date,
        config=VisualReviewConfig(
            crop_padding_m=args.crop_padding_m,
            max_candidates=args.max_candidates,
            min_area_m2=args.min_area_m2,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    report.update({"output": str(args.output), "model": args.model, "transport": "single_side_by_side_comparison_image"})
    (args.output.parent / "visual_review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
