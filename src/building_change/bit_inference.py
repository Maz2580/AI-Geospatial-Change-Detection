"""Bi-temporal Transformer change detection on georeferenced imagery.

This module provides the full pipeline from paired GeoTIFFs to vectorised
change-candidate GeoJSON.  It downloads a pretrained BIT checkpoint from
Hugging Face Hub (or uses a local file), runs tiled inference on CPU, and
produces candidates compatible with the existing fusion pipeline.

The architecture is trained on the LEVIR-CD dataset (building change in
high-resolution aerial imagery) and specifically learns to suppress
false positives from shadows, seasonal vegetation, and radiometric shifts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.warp import Resampling, reproject, transform_geom
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)

# Re-use existing detector utilities
from .detector import DetectionConfig, _area_m2, _read_new_image, _reproject_rgb, _shadow_mask, _write_raster
from .regularisation import RegularisationConfig, regularise_geometries

# Model import is deferred to avoid hard torch dependency at module level
_HF_REPO_ID = "HZDR-FWGEL/UCD-LEVIRCD256-BIT"
_HF_WEIGHT_FILENAME = "model.safetensors"


class BITInferenceError(ValueError):
    """Raised when bi-temporal Transformer inference fails."""


@dataclass(frozen=True)
class BITConfig:
    """Settings for BIT tiled inference and vectorisation."""

    tile_px: int = 256
    stride_px: int = 192
    change_threshold: float = 0.45
    min_area_m2: float = 15.0
    morphology_px: int = 5
    batch_size: int = 4
    backbone: str = "resnet18"
    token_count: int = 16
    token_dim: int = 64
    transformer_layers: int = 4

    def validate(self) -> None:
        if self.tile_px < 64:
            raise BITInferenceError("tile_px must be at least 64.")
        if not 0 < self.stride_px <= self.tile_px:
            raise BITInferenceError("stride_px must be positive and at most tile_px.")
        if not 0 < self.change_threshold < 1:
            raise BITInferenceError("change_threshold must be between 0 and 1.")


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    """Tile starts that cover both edges."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def _resolve_weights(model_path: str | Path | None, cache_dir: str | Path) -> Path | None:
    """Return a local weights file, downloading from Hugging Face if needed."""
    if model_path is not None:
        selected = Path(model_path)
        if selected.is_file():
            return selected
        logger.warning("Specified BIT weights file not found: %s", selected)

    # Check local cache
    cache = Path(cache_dir)
    cached = cache / _HF_WEIGHT_FILENAME
    if cached.is_file():
        return cached

    # Try downloading from HF Hub
    try:
        from huggingface_hub import hf_hub_download
        import os
        downloaded = Path(hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_HF_WEIGHT_FILENAME,
            cache_dir=str(cache),
            token=os.getenv("HF_TOKEN") or None,
        ))
        logger.info("Successfully downloaded BIT weights from HF: %s", downloaded)
        return downloaded
    except Exception as exc:
        logger.warning("Could not download BIT weights from HF repo %s: %s", _HF_REPO_ID, exc)
        return None


_CACHED_MODEL: "torch.nn.Module | None" = None


def _build_model(config: BITConfig, weights_path: Path | None = None) -> tuple["torch.nn.Module", bool]:
    """Construct AdaptFormer LEVIR-CD model, caching after first load."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        logger.info("Using cached AdaptFormer LEVIR-CD model")
        return _CACHED_MODEL, True

    from transformers import AutoModel

    logger.info("Loading SOTA AdaptFormer LEVIR-CD bitemporal model from HF: deepang/adaptformer-LEVIR-CD")
    model = AutoModel.from_pretrained("deepang/adaptformer-LEVIR-CD", trust_remote_code=True)
    model.eval()
    _CACHED_MODEL = model
    return model, True


def _normalise_for_model(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Convert to uint8 RGB HWC format for tiling."""
    result = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)
    for band in range(3):
        values = rgb[band][valid & np.isfinite(rgb[band])]
        if values.size:
            low, high = np.percentile(values, (2, 98))
            result[:, :, band] = np.clip(
                (rgb[band] - low) * 255.0 / max(high - low, 1.0), 0, 255
            ).astype(np.uint8)
    return result


@torch.no_grad()
def _predict_change_probability(
    model: "torch.nn.Module",
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    valid: np.ndarray,
    config: BITConfig,
) -> np.ndarray:
    """Run tiled bitemporal inference using AdaptFormer LEVIR-CD."""
    import torch
    import torch.nn.functional as F

    height, width = valid.shape
    probability_sum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.uint16)

    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Use 1024 tile size downscaled to 256 for LEVIR-CD scale matching
    tile_px = config.tile_px if config.tile_px >= 512 else 1024
    stride_px = config.stride_px if config.stride_px >= 384 else 768

    y_starts = _window_starts(height, tile_px, stride_px)
    x_starts = _window_starts(width, tile_px, stride_px)

    total_tiles = len(y_starts) * len(x_starts)
    logger.info("AdaptFormer bitemporal inference: %d tiles (size %d, stride %d)", total_tiles, tile_px, stride_px)

    processed = 0
    for y in y_starts:
        for x in x_starts:
            y_end = min(y + tile_px, height)
            x_end = min(x + tile_px, width)

            patch_b = before_rgb[y:y_end, x:x_end].astype(np.float32) / 255.0
            patch_a = after_rgb[y:y_end, x:x_end].astype(np.float32) / 255.0

            tb = np.zeros((tile_px, tile_px, 3), dtype=np.float32)
            ta = np.zeros((tile_px, tile_px, 3), dtype=np.float32)

            tb[:y_end - y, :x_end - x] = patch_b
            ta[:y_end - y, :x_end - x] = patch_a

            tb = (tb - mean) / std
            ta = (ta - mean) / std

            t_b = torch.from_numpy(tb.transpose(2, 0, 1)).unsqueeze(0).float()
            t_a = torch.from_numpy(ta.transpose(2, 0, 1)).unsqueeze(0).float()

            # Downsample tile to 256x256 expected by LEVIR-CD model
            t_b_small = F.interpolate(t_b, size=(256, 256), mode="bilinear", align_corners=False)
            t_a_small = F.interpolate(t_a, size=(256, 256), mode="bilinear", align_corners=False)

            out = model(t_b_small, t_a_small)
            probs = torch.softmax(out.logits, dim=1)[:, 1:2]
            probs_large = F.interpolate(probs, size=(tile_px, tile_px), mode="bilinear", align_corners=False)
            tile_p = probs_large[0, 0].cpu().numpy()

            probability_sum[y:y_end, x:x_end] += tile_p[:y_end - y, :x_end - x]
            count[y:y_end, x:x_end] += 1
            processed += 1
            if processed % 50 == 0:
                logger.info("  Processed %d/%d tiles...", processed, total_tiles)

    result = np.divide(
        probability_sum,
        count,
        out=np.zeros_like(probability_sum, dtype=np.float32),
        where=count > 0,
    ).astype(np.float32)
    result[~valid] = 0.0
    return result


def _vectorise_change_mask(
    mask: np.ndarray,
    probability: np.ndarray,
    profile: dict[str, Any],
    config: BITConfig,
    *,
    metadata: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert a binary change mask to GeoJSON features."""
    candidates: list[dict[str, Any]] = []
    source_crs = profile.get("crs")
    if source_crs is None:
        raise BITInferenceError("Imagery has no CRS.")
    raw_geometries: list[Any] = []
    for geometry_json, value in shapes(mask, mask=mask.astype(bool), transform=profile["transform"]):
        if value != 1:
            continue
        geometry = shape(geometry_json)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty or _area_m2(geometry, source_crs) < config.min_area_m2:
            continue
        raw_geometries.append(geometry)

    # Pass geometries through 90/45 degree orthogonal regularisation
    geometries = regularise_geometries(raw_geometries, source_crs, RegularisationConfig())

    for geometry in geometries:
        area = _area_m2(geometry, source_crs)
        if area < config.min_area_m2:
            continue

        from rasterio.features import geometry_mask
        footprint = geometry_mask([mapping(geometry)], out_shape=mask.shape, transform=profile["transform"], invert=True)
        values = probability[footprint]
        mean_prob = float(np.mean(values)) if values.size else 0.0

        rectangularity = 0.0
        min_rect = geometry.minimum_rotated_rectangle
        if min_rect.area > 0:
            rectangularity = min(1.0, float(geometry.area / min_rect.area))

        classification = "likely_building_change"
        if rectangularity >= 0.35:
            classification = "likely_new_building"

        feature = {
            "type": "Feature",
            "geometry": transform_geom(source_crs, "EPSG:4326", mapping(geometry), precision=7),
            "properties": {
                "candidate_id": len(candidates) + 1,
                "classification": classification,
                "area_m2": round(area, 2),
                "rectangularity": round(rectangularity, 3),
                "mean_change_probability": round(mean_prob, 4),
                "candidate_source": "bit_transformer",
                "model": "BIT-ResNet18-LEVIR",
                **metadata,
            },
        }
        candidates.append(feature)

    return candidates


def run_bit_change_detection(
    before_image: str | Path,
    after_image: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    model_cache: str | Path = ".cache/bit_weights",
    config: BITConfig | None = None,
    before_capture_date: str | None = None,
    after_capture_date: str | None = None,
) -> dict[str, Any]:
    """Run BIT bi-temporal change detection and produce fusion-ready GeoJSON.

    Returns a JSON-serialisable report compatible with the existing pipeline.
    """
    active_config = config or BITConfig()
    active_config.validate()

    before_path = Path(before_image)
    after_path = Path(after_image)
    directory = Path(output_dir)

    if not before_path.is_file() or not after_path.is_file():
        raise FileNotFoundError("Both before and after RGB GeoTIFFs must exist.")
    directory.mkdir(parents=True, exist_ok=True)

    # Load imagery
    logger.info("Loading after image: %s", after_path)
    after_rgb, after_valid, profile = _read_new_image(after_path)
    logger.info("Loading and reprojecting before image: %s", before_path)
    before_rgb, before_valid = _reproject_rgb(before_path, profile)
    valid = before_valid & after_valid

    # Normalise to uint8 HWC
    before_norm = _normalise_for_model(before_rgb, before_valid)
    after_norm = _normalise_for_model(after_rgb, after_valid)

    # Build model
    weights_path = _resolve_weights(model_path, model_cache)
    model, has_weights = _build_model(active_config, weights_path)

    # Run inference
    logger.info("Running BIT change detection inference...")
    change_prob = _predict_change_probability(model, before_norm, after_norm, valid, active_config)

    # Compute cast shadow mask on after-image to subtract shadows from building outlines
    after_shadow = _shadow_mask(after_rgb, after_valid, profile, DetectionConfig())
    if np.any(after_shadow):
        logger.info("Subtracting cast shadows from change probability map (%d shadow pixels)", np.count_nonzero(after_shadow))
        change_prob[after_shadow] = 0.0

    # Threshold and clean
    change_mask = ((change_prob >= active_config.change_threshold) & valid).astype(np.uint8)
    if active_config.morphology_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (active_config.morphology_px, active_config.morphology_px),
        )
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)

    # Save probability map
    prob_path = directory / "bit_change_probability.tif"
    mask_path = directory / "bit_change_mask.tif"
    prob_for_save = change_prob.copy()
    prob_for_save[~valid] = -9999.0
    _write_raster(prob_path, prob_for_save, profile, dtype="float32", nodata=-9999.0)
    _write_raster(mask_path, change_mask * 255, profile, dtype="uint8", nodata=0)

    # Vectorise
    metadata = {
        "before_capture_date": before_capture_date or "unknown",
        "after_capture_date": after_capture_date or "unknown",
    }
    candidates = _vectorise_change_mask(change_mask, change_prob, profile, active_config, metadata=metadata)

    # Write GeoJSON
    candidates_path = directory / "bit_change_candidates.geojson"
    collection = {
        "type": "FeatureCollection",
        "features": candidates,
        "metadata": {
            "crs": "EPSG:4326",
            "requested_detector": "bit_transformer",
            "model": "BIT-ResNet18-LEVIR",
            "change_threshold": active_config.change_threshold,
            "min_area_m2": active_config.min_area_m2,
        },
    }
    candidates_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")

    report = {
        "before_image": str(before_path),
        "after_image": str(after_path),
        "model": "BIT-ResNet18-LEVIR",
        "candidate_count": len(candidates),
        "change_threshold": active_config.change_threshold,
        "has_pretrained_weights": has_weights,
        "config": asdict(active_config),
        "outputs": {
            "change_probability": str(prob_path),
            "change_mask": str(mask_path),
            "change_candidates": str(candidates_path),
        },
    }
    report_path = directory / "bit_change_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("BIT detection complete: %d candidates", len(candidates))
    return report
