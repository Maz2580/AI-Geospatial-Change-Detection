"""Create before/after review chips for GeoJSON already on an imagery raster grid.

Unlike ``generate_change_review.py``, this helper does not assume WGS84
coordinates. It is for model vectors written in the exact CRS and transform of
the supplied after-date GeoTIFF, such as the ChangeStar baseline outputs.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import MultiPolygon, Polygon, shape


def _read_rgb(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as source:
        if source.count < 3:
            raise ValueError(f"{path} does not contain three RGB bands.")
        image = np.moveaxis(source.read(indexes=(1, 2, 3)), 0, -1)
        profile = source.profile.copy()
    if image.dtype != np.uint8:
        raise ValueError(f"{path} is not uint8 RGB imagery.")
    return image, profile


def _matching_grid(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return (
        before["width"] == after["width"]
        and before["height"] == after["height"]
        and before["crs"] == after["crs"]
        and before["transform"] == after["transform"]
    )


def _outlines(geometry: Polygon | MultiPolygon) -> list[list[tuple[float, float]]]:
    if isinstance(geometry, Polygon):
        return [list(geometry.exterior.coords)]
    if isinstance(geometry, MultiPolygon):
        return [list(part.exterior.coords) for part in geometry.geoms]
    raise ValueError("Candidate geometry must be a Polygon or MultiPolygon.")


def _window(feature: dict[str, Any], profile: dict[str, Any], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = shape(feature["geometry"]).bounds
    rows, columns = rowcol(profile["transform"], [min_x, min_x, max_x, max_x], [min_y, max_y, min_y, max_y])
    height, width = image_shape
    padding = 70
    return (
        max(0, min(columns) - padding),
        max(0, min(rows) - padding),
        min(width, max(columns) + padding + 1),
        min(height, max(rows) + padding + 1),
    )


def _outline_image(image: np.ndarray, feature: dict[str, Any], profile: dict[str, Any], left: int, top: int) -> np.ndarray:
    result = np.ascontiguousarray(image.copy())
    for coordinates in _outlines(shape(feature["geometry"])):
        xs, ys = zip(*coordinates)
        rows, columns = rowcol(profile["transform"], xs, ys)
        pixels = np.array([(column - left, row - top) for row, column in zip(rows, columns)], dtype=np.int32)
        cv2.polylines(result, [pixels], isClosed=True, color=(255, 48, 48), thickness=2)
    return result


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write {path}")


def _write_review(before: np.ndarray, after: np.ndarray, profile: dict[str, Any], features: list[dict[str, Any]], output: Path, source: str) -> Path:
    review_dir = output / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for feature in features:
        properties = feature.get("properties", {})
        candidate_id = int(properties.get("candidate_id", len(cards) + 1))
        left, top, right, bottom = _window(feature, profile, before.shape[:2])
        _write_png(review_dir / f"candidate_{candidate_id:03d}_before.png", _outline_image(before[top:bottom, left:right], feature, profile, left, top))
        _write_png(review_dir / f"candidate_{candidate_id:03d}_after.png", _outline_image(after[top:bottom, left:right], feature, profile, left, top))
        rows = "".join(
            f"<tr><th>{escape(str(key).replace('_', ' '))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in properties.items()
            if key != "candidate_id"
        )
        label = escape(str(properties.get("classification", properties.get("candidate_kind", "candidate"))))
        cards.append(
            "<article>"
            f"<h2>Candidate {candidate_id}: {label}</h2>"
            "<div class='images'>"
            f"<figure><figcaption>Before</figcaption><img src='candidate_{candidate_id:03d}_before.png' alt='Before candidate {candidate_id}'></figure>"
            f"<figure><figcaption>After</figcaption><img src='candidate_{candidate_id:03d}_after.png' alt='After candidate {candidate_id}'></figure>"
            "</div>"
            f"<table>{rows}</table>"
            "</article>"
        )
    if not cards:
        cards.append("<p>No candidates met the fixed threshold and minimum area.</p>")
    html = """<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Projected-grid model review</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;background:#f6f8fa;color:#172033;max-width:1200px}article{background:white;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin:1rem 0}.images{display:flex;gap:1rem;flex-wrap:wrap}.images figure{margin:0;min-width:280px;flex:1}img{width:100%;max-height:520px;object-fit:contain;background:#20252f;border-radius:4px}figcaption{font-weight:650;margin-bottom:.45rem}table{border-collapse:collapse;margin-top:1rem}th,td{padding:.35rem .6rem;border:1px solid #d8dee4;text-align:left}th{text-transform:capitalize;background:#f6f8fa}</style></head>
<body><h1>Projected-grid model review</h1>""" + f"<p>Red outlines are model proposals from {escape(source)}. The vectors are on the same projected grid as the displayed imagery.</p>" + "\n".join(cards) + "</body></html>"
    index = review_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    (review_dir / "candidates.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-source", default="the supplied model")
    args = parser.parse_args()
    before, before_profile = _read_rgb(args.before)
    after, after_profile = _read_rgb(args.after)
    if not _matching_grid(before_profile, after_profile):
        raise ValueError("Projected-grid review requires matching before/after GeoTIFF grids.")
    features = json.loads(args.candidates.read_text(encoding="utf-8")).get("features", [])
    print(_write_review(before, after, after_profile, features, args.output, args.candidate_source))


if __name__ == "__main__":
    main()
