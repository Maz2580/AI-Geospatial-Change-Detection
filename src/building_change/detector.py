"""Geospatially-aware construction-change candidate extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.warp import Resampling, reproject, transform_geom
from shapely.geometry import mapping, shape


@dataclass(frozen=True)
class DetectionConfig:
    change_percentile: float = 98.5
    change_threshold: float | None = None
    morphology_m: float = 0.6
    min_area_m2: float = 20.0
    building_rectangularity: float = 0.35
    min_height_rise_m: float = 1.5
    max_registration_shift_px: float = 10.0
    enable_registration: bool = True


def _read_new_image(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as dataset:
        if dataset.count < 3:
            raise ValueError(f"{path} has {dataset.count} band(s); an RGB GeoTIFF is required.")
        rgb = dataset.read((1, 2, 3), out_dtype="float32")
        valid = np.all(dataset.read_masks((1, 2, 3)) > 0, axis=0)
        profile = dataset.profile.copy()
        profile.update(height=dataset.height, width=dataset.width, transform=dataset.transform, crs=dataset.crs)
    return rgb, valid, profile


def _reproject_rgb(source_path: Path, target_profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    height, width = target_profile["height"], target_profile["width"]
    destination = np.full((3, height, width), np.nan, dtype=np.float32)
    valid_source = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(source_path) as source:
        if source.count < 3:
            raise ValueError(f"{source_path} has {source.count} band(s); an RGB GeoTIFF is required.")
        for band in range(1, 4):
            reproject(
                source=rasterio.band(source, band), destination=destination[band - 1],
                src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
                dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
                dst_nodata=np.nan, resampling=Resampling.bilinear,
            )
        reproject(
            source=source.read_masks(1), destination=valid_source,
            src_transform=source.transform, src_crs=source.crs,
            dst_transform=target_profile["transform"], dst_crs=target_profile["crs"], resampling=Resampling.nearest,
        )
    return destination, np.all(np.isfinite(destination), axis=0) & (valid_source > 0)


def _reproject_single_band(source_path: Path, target_profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    height, width = target_profile["height"], target_profile["width"]
    destination = np.full((height, width), np.nan, dtype=np.float32)
    valid_source = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(source_path) as source:
        reproject(
            source=rasterio.band(source, 1), destination=destination,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
            dst_nodata=np.nan, resampling=Resampling.bilinear,
        )
        reproject(
            source=source.read_masks(1), destination=valid_source,
            src_transform=source.transform, src_crs=source.crs,
            dst_transform=target_profile["transform"], dst_crs=target_profile["crs"], resampling=Resampling.nearest,
        )
    return destination, np.isfinite(destination) & (valid_source > 0)


def _normalise_rgb(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Robust per-image scaling reduces false positives from exposure changes."""
    result = np.zeros(rgb.shape, dtype=np.uint8)
    for band in range(3):
        values = rgb[band][valid & np.isfinite(rgb[band])]
        if values.size:
            low, high = np.percentile(values, (2, 98))
            result[band] = np.clip((rgb[band] - low) * 255.0 / max(high - low, 1.0), 0, 255).astype(np.uint8)
    return np.moveaxis(result, 0, -1)


def _register_before(
    before: np.ndarray, after: np.ndarray, valid: np.ndarray, *, enabled: bool, max_shift_px: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool | str]]:
    """Perform conservative translation-only ECC registration."""
    details: dict[str, float | bool | str] = {"applied": False, "dx_px": 0.0, "dy_px": 0.0}
    height, width = before.shape[:2]
    if not enabled:
        details["reason"] = "disabled"
        return before, valid, details
    if min(height, width) < 32 or valid.sum() < 1024:
        details["reason"] = "insufficient_valid_pixels"
        return before, valid, details
    scale = min(1.0, 1200.0 / max(height, width))
    size = (max(32, round(width * scale)), max(32, round(height * scale)))
    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    before_small = cv2.GaussianBlur(cv2.resize(before_gray, size, interpolation=cv2.INTER_AREA), (5, 5), 0)
    after_small = cv2.GaussianBlur(cv2.resize(after_gray, size, interpolation=cv2.INTER_AREA), (5, 5), 0)
    mask_small = cv2.resize(valid.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    try:
        correlation, warp = cv2.findTransformECC(after_small, before_small, warp, cv2.MOTION_TRANSLATION, criteria, inputMask=mask_small)
        warp[:, 2] /= scale
        dx, dy = float(warp[0, 2]), float(warp[1, 2])
        if math.hypot(dx, dy) > max_shift_px:
            details.update({"reason": "shift_exceeds_limit", "dx_px": dx, "dy_px": dy, "correlation": float(correlation)})
            return before, valid, details
        aligned = cv2.warpAffine(before, warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        aligned_valid = cv2.warpAffine(valid.astype(np.uint8), warp, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT).astype(bool)
        details.update({"applied": True, "dx_px": dx, "dy_px": dy, "correlation": float(correlation)})
        return aligned, aligned_valid, details
    except cv2.error as exc:
        details["reason"] = f"ecc_failed: {str(exc).splitlines()[0]}"
        return before, valid, details


def _pixel_size_m(profile: dict[str, Any]) -> float:
    transform = profile["transform"]
    determinant = abs(transform.a * transform.e - transform.b * transform.d)
    if profile["crs"] and profile["crs"].is_projected and determinant > 0:
        return math.sqrt(determinant)
    return 1.0


def _change_score(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_smooth = cv2.GaussianBlur(before, (5, 5), 0)
    after_smooth = cv2.GaussianBlur(after, (5, 5), 0)
    before_lab = cv2.cvtColor(before_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    after_lab = cv2.cvtColor(after_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    colour = np.linalg.norm(after_lab - before_lab, axis=2) / (255.0 * math.sqrt(3.0))
    before_gray, after_gray = cv2.cvtColor(before_smooth, cv2.COLOR_RGB2GRAY), cv2.cvtColor(after_smooth, cv2.COLOR_RGB2GRAY)
    before_edges = cv2.Sobel(before_gray, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(before_gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    after_edges = cv2.Sobel(after_gray, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(after_gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    edges = np.abs(np.sqrt(after_edges) - np.sqrt(before_edges)) / (255.0 * math.sqrt(2.0))
    return np.clip(0.8 * colour + 0.2 * edges, 0, 1).astype(np.float32)


def _clean_mask(score: np.ndarray, valid: np.ndarray, profile: dict[str, Any], config: DetectionConfig) -> tuple[np.ndarray, float]:
    values = score[valid]
    if not values.size:
        raise ValueError("The before and after images do not overlap on valid pixels.")
    threshold = config.change_threshold
    if threshold is None:
        if not 0 < config.change_percentile < 100:
            raise ValueError("change_percentile must be between 0 and 100.")
        threshold = float(np.percentile(values, config.change_percentile))
    if not 0 <= threshold <= 1:
        raise ValueError("change_threshold must be between 0 and 1.")
    mask = ((score >= threshold) & valid).astype(np.uint8)
    kernel_size = max(1, round(config.morphology_m / _pixel_size_m(profile)))
    if kernel_size > 1:
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, float(threshold)


def _area_m2(geometry, source_crs) -> float:
    if source_crs and source_crs.is_projected:
        return float(geometry.area)
    return float(shape(transform_geom(source_crs, "EPSG:3857", mapping(geometry), precision=12)).area)


def _write_raster(path: Path, array: np.ndarray, profile: dict[str, Any], *, dtype: str, nodata: float | int) -> None:
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(array.astype(dtype), 1)


def _write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _vectorise(
    mask: np.ndarray, score: np.ndarray, profile: dict[str, Any], config: DetectionConfig, *,
    height_delta: np.ndarray | None, valid_height: np.ndarray | None, metadata: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    likely_buildings: list[dict[str, Any]] = []
    source_crs = profile["crs"]
    if source_crs is None:
        raise ValueError("The after image has no CRS; a GeoTIFF with georeferencing is required.")
    # rasterio.features.shapes returns (geometry, pixel_value), in that order.
    for geometry_json, value in shapes(mask, mask=mask.astype(bool), transform=profile["transform"]):
        if value != 1:
            continue
        geometry = shape(geometry_json)
        area = _area_m2(geometry, source_crs)
        if area < config.min_area_m2:
            continue
        footprint = geometry_mask([geometry_json], out_shape=mask.shape, transform=profile["transform"], invert=True)
        rectangularity = 0.0 if geometry.minimum_rotated_rectangle.area <= 0 else min(1.0, float(geometry.area / geometry.minimum_rotated_rectangle.area))
        mean_height_change: float | None = None
        if height_delta is not None and valid_height is not None:
            values = height_delta[footprint & valid_height]
            if values.size:
                mean_height_change = float(np.mean(values))
        classification = "construction_change_candidate"
        if mean_height_change is not None:
            if mean_height_change >= config.min_height_rise_m:
                classification = "likely_new_building"
            elif mean_height_change <= -config.min_height_rise_m:
                classification = "likely_demolition"
        elif rectangularity >= config.building_rectangularity:
            classification = "likely_building_change"
        feature = {
            "type": "Feature",
            "geometry": transform_geom(source_crs, "EPSG:4326", mapping(geometry), precision=7),
            "properties": {
                "candidate_id": len(candidates) + 1, "classification": classification,
                "area_m2": round(area, 2), "rectangularity": round(rectangularity, 3),
                "mean_change_score": round(float(np.mean(score[footprint])), 4),
                "mean_height_change_m": round(mean_height_change, 3) if mean_height_change is not None else None,
                **metadata,
            },
        }
        candidates.append(feature)
        if classification == "likely_new_building" or (height_delta is None and classification == "likely_building_change"):
            likely_buildings.append(feature)
    return candidates, likely_buildings


def run_detection(
    before_image: str | Path, after_image: str | Path, output_dir: str | Path, *,
    before_dsm: str | Path | None = None, after_dsm: str | Path | None = None,
    config: DetectionConfig | None = None, before_capture_date: str | None = None, after_capture_date: str | None = None,
) -> dict[str, Any]:
    """Run the full local detection stage and return a JSON-serialisable report."""
    if (before_dsm is None) != (after_dsm is None):
        raise ValueError("Provide both before_dsm and after_dsm, or neither.")
    config = config or DetectionConfig()
    before_path, after_path, output_path = Path(before_image), Path(after_image), Path(output_dir)
    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError("Both before_image and after_image must exist.")
    output_path.mkdir(parents=True, exist_ok=True)
    after_rgb, after_valid, profile = _read_new_image(after_path)
    before_rgb, before_valid = _reproject_rgb(before_path, profile)
    before_normal, after_normal = _normalise_rgb(before_rgb, before_valid), _normalise_rgb(after_rgb, after_valid)
    before_aligned, aligned_valid, registration = _register_before(
        before_normal, after_normal, before_valid & after_valid,
        enabled=config.enable_registration, max_shift_px=config.max_registration_shift_px,
    )
    valid = aligned_valid & after_valid
    score = _change_score(before_aligned, after_normal)
    score[~valid] = 0
    mask, threshold = _clean_mask(score, valid, profile, config)
    height_delta: np.ndarray | None = None
    valid_height: np.ndarray | None = None
    if before_dsm is not None and after_dsm is not None:
        before_elevation, before_elevation_valid = _reproject_single_band(Path(before_dsm), profile)
        after_elevation, after_elevation_valid = _reproject_single_band(Path(after_dsm), profile)
        height_delta = after_elevation - before_elevation
        valid_height = before_elevation_valid & after_elevation_valid & valid
        height_delta[~valid_height] = np.nan
    candidates, likely_buildings = _vectorise(
        mask, score, profile, config, height_delta=height_delta, valid_height=valid_height,
        metadata={"before_capture_date": before_capture_date or "unknown", "after_capture_date": after_capture_date or "unknown"},
    )
    score_path, mask_path = output_path / "change_score.tif", output_path / "change_mask.tif"
    candidates_path, buildings_path = output_path / "construction_change_candidates.geojson", output_path / "likely_new_buildings.geojson"
    _write_raster(score_path, score, profile, dtype="float32", nodata=-9999.0)
    _write_raster(mask_path, mask * 255, profile, dtype="uint8", nodata=0)
    _write_geojson(candidates_path, candidates)
    _write_geojson(buildings_path, likely_buildings)
    report = {
        "before_image": str(before_path), "after_image": str(after_path),
        "outputs": {"change_score": str(score_path), "change_mask": str(mask_path), "construction_candidates": str(candidates_path), "likely_new_buildings": str(buildings_path)},
        "candidate_count": len(candidates), "likely_new_building_count": len(likely_buildings),
        "threshold": threshold, "registration": registration, "used_dsm": height_delta is not None, "config": asdict(config),
    }
    (output_path / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
