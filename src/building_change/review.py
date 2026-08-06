"""Create portable, human-reviewable before/after candidate reports."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rasterio.transform import rowcol
from rasterio.warp import transform
from shapely.geometry import shape


def _candidate_window(feature: dict[str, Any], profile: dict[str, Any], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return a padded, clamped pixel window around one WGS84 GeoJSON feature."""

    geometry = shape(feature["geometry"])
    coordinates = list(geometry.exterior.coords)
    longitudes, latitudes = zip(*coordinates)
    projected_x, projected_y = transform("EPSG:4326", profile["crs"], longitudes, latitudes)
    rows, columns = rowcol(profile["transform"], projected_x, projected_y)
    height, width = image_shape
    padding = 70
    left = max(0, min(columns) - padding)
    right = min(width, max(columns) + padding + 1)
    top = max(0, min(rows) - padding)
    bottom = min(height, max(rows) + padding + 1)
    return left, top, right, bottom


def _draw_outline(image: np.ndarray, feature: dict[str, Any], profile: dict[str, Any], *, left: int, top: int) -> np.ndarray:
    result = np.ascontiguousarray(image.copy())
    geometry = shape(feature["geometry"])
    coordinates = list(geometry.exterior.coords)
    longitudes, latitudes = zip(*coordinates)
    projected_x, projected_y = transform("EPSG:4326", profile["crs"], longitudes, latitudes)
    rows, columns = rowcol(profile["transform"], projected_x, projected_y)
    outline = np.array([(column - left, row - top) for row, column in zip(rows, columns)], dtype=np.int32)
    cv2.polylines(result, [outline], isClosed=True, color=(255, 48, 48), thickness=2)
    return result


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write review image: {path}")


def write_candidate_review(
    before: np.ndarray,
    after: np.ndarray,
    profile: dict[str, Any],
    features: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Write HTML plus annotated PNG crops for manual quality assurance.

    ``before`` and ``after`` are matching H×W×RGB arrays on ``profile``'s
    raster grid. This means the review images show exactly what the detector
    evaluated, including any registration correction.
    """

    review_dir = Path(output_dir) / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for feature in features:
        properties = feature["properties"]
        candidate_id = int(properties["candidate_id"])
        left, top, right, bottom = _candidate_window(feature, profile, before.shape[:2])
        before_crop = _draw_outline(before[top:bottom, left:right], feature, profile, left=left, top=top)
        after_crop = _draw_outline(after[top:bottom, left:right], feature, profile, left=left, top=top)
        before_name = f"candidate_{candidate_id:03d}_before.png"
        after_name = f"candidate_{candidate_id:03d}_after.png"
        _write_image(review_dir / before_name, before_crop)
        _write_image(review_dir / after_name, after_crop)
        rows = "".join(
            f"<tr><th>{escape(str(key).replace('_', ' '))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in properties.items()
            if key != "candidate_id"
        )
        cards.append(
            "<article>"
            f"<h2>Candidate {candidate_id}: {escape(str(properties['classification']))}</h2>"
            "<div class='images'>"
            f"<figure><figcaption>Before</figcaption><img src='{before_name}' alt='Before candidate {candidate_id}'></figure>"
            f"<figure><figcaption>After</figcaption><img src='{after_name}' alt='After candidate {candidate_id}'></figure>"
            "</div>"
            f"<table>{rows}</table>"
            "</article>"
        )
    if not cards:
        cards.append("<p>No candidates met the configured threshold and minimum area.</p>")
    html = """<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Construction change review</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;background:#f6f8fa;color:#172033;max-width:1200px}
article{background:white;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin:1rem 0}
.images{display:flex;gap:1rem;flex-wrap:wrap}.images figure{margin:0;min-width:280px;flex:1}
img{width:100%;max-height:520px;object-fit:contain;background:#20252f;border-radius:4px}figcaption{font-weight:650;margin-bottom:.45rem}
table{border-collapse:collapse;margin-top:1rem}th,td{padding:.35rem .6rem;border:1px solid #d8dee4;text-align:left}th{text-transform:capitalize;background:#f6f8fa}
</style></head><body><h1>Construction change review</h1>
<p>Red outlines are candidates extracted by the change detector. Review the before/after pair before treating an RGB-only result as a confirmed new building.</p>
""" + "\n".join(cards) + "</body></html>"
    index_path = review_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    (review_dir / "candidates.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8"
    )
    return index_path
