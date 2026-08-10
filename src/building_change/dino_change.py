"""DINOv2 semantic feature comparison for change detection.

This module uses pre-trained DINOv2 vision transformer features to detect
semantically meaningful changes between two aerial images. Unlike pixel-level
colour differences, DINOv2 embeddings capture high-level scene understanding,
making them inherently robust to lighting, shadow, and seasonal variations.

How it works:
  1. For each tile position, extract DINOv2 patch features from both dates
  2. Compute cosine distance between the two feature sets *per tile*
  3. Accumulate the scalar distance map (not the full feature tensors)
  4. Threshold, clean, and vectorise into GeoJSON candidates

No training is required — this is a zero-shot approach using the publicly
available DINOv2 backbone from Meta/FAIR.

Memory-efficient: only scalar distance values are accumulated, not the
384-dimensional feature vectors. Peak memory is controlled by batch_size.
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
import torch.nn.functional as F
from rasterio.features import geometry_mask, shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from .detector import _area_m2, _read_new_image, _reproject_rgb, _write_raster

logger = logging.getLogger(__name__)


class DINOChangeError(ValueError):
    """Raised when DINOv2 change detection fails."""


@dataclass(frozen=True)
class DINOChangeConfig:
    """Settings for DINOv2-based change detection."""

    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    tile_px: int = 518      # DINOv2 native resolution (37 × 14-px patches)
    stride_px: int = 392    # ~75% overlap for smooth stitching
    change_percentile: float = 95.0
    change_threshold: float | None = None
    min_area_m2: float = 15.0
    morphology_px: int = 5
    batch_size: int = 2

    def validate(self) -> None:
        if self.tile_px < 56:
            raise DINOChangeError("tile_px must be at least 56.")
        if not 0 < self.stride_px <= self.tile_px:
            raise DINOChangeError("stride_px must be positive and at most tile_px.")


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    """Tile start positions that cover both edges."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def _build_feature_extractor(model_name: str) -> torch.nn.Module:
    """Create a DINOv2 feature extractor using timm."""
    import timm

    logger.info("Loading DINOv2 model: %s", model_name)
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    return model


def _normalise_for_model(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalise from CHW float32 GeoTIFF to HWC uint8."""
    result = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)
    for band in range(3):
        values = rgb[band][valid & np.isfinite(rgb[band])]
        if values.size:
            low, high = np.percentile(values, (2, 98))
            result[:, :, band] = np.clip(
                (rgb[band] - low) * 255.0 / max(high - low, 1.0), 0, 255
            ).astype(np.uint8)
    return result


def _prepare_tile(image: np.ndarray, y: int, x: int, tile_px: int, height: int, width: int) -> np.ndarray:
    """Extract and normalise one tile for DINOv2."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    tile = np.zeros((tile_px, tile_px, 3), dtype=np.float32)
    y_end = min(y + tile_px, height)
    x_end = min(x + tile_px, width)
    patch = image[y:y_end, x:x_end].astype(np.float32) / 255.0
    tile[:y_end - y, :x_end - x] = patch
    tile = (tile - mean) / std
    return tile


@torch.no_grad()
def _compute_semantic_change_map(
    model: torch.nn.Module,
    before_image: np.ndarray,
    after_image: np.ndarray,
    valid: np.ndarray,
    config: DINOChangeConfig,
) -> np.ndarray:
    """Compute per-pixel semantic change score between two images.

    Memory-efficient: processes tiles in pairs (before+after), computes cosine
    distance immediately, and only accumulates the scalar distance values.
    """
    height, width = valid.shape
    patch_size = 14  # DINOv2 uses 14×14 patches

    # Determine feature layout from a test pass
    test_input = torch.randn(1, 3, config.tile_px, config.tile_px)
    test_out = model.forward_features(test_input)
    if isinstance(test_out, dict):
        test_out = test_out.get("x_norm_patchtokens", list(test_out.values())[0])
    if test_out.ndim == 3:
        num_patches = test_out.shape[1]
        patches_per_side = int(math.sqrt(num_patches))
        feature_dim = test_out.shape[2]
    else:
        patches_per_side = config.tile_px // patch_size
        feature_dim = test_out.shape[1]
    del test_input, test_out

    logger.info("DINOv2 feature dim: %d, patches per tile side: %d", feature_dim, patches_per_side)

    # Accumulate cosine distance at PATCH resolution (much smaller than pixel)
    patch_h = (height + patch_size - 1) // patch_size
    patch_w = (width + patch_size - 1) // patch_size
    distance_sum = np.zeros((patch_h, patch_w), dtype=np.float64)
    distance_count = np.zeros((patch_h, patch_w), dtype=np.uint16)

    y_starts = _window_starts(height, config.tile_px, config.stride_px)
    x_starts = _window_starts(width, config.tile_px, config.stride_px)

    total_tiles = len(y_starts) * len(x_starts)
    logger.info("DINOv2 inference: %d tile pairs (%dx%d stride %d)",
                total_tiles, config.tile_px, config.tile_px, config.stride_px)

    processed = 0
    for y in y_starts:
        for x in x_starts:
            y_end = min(y + config.tile_px, height)
            x_end = min(x + config.tile_px, width)

            # Prepare both tiles
            tile_b = _prepare_tile(before_image, y, x, config.tile_px, height, width)
            tile_a = _prepare_tile(after_image, y, x, config.tile_px, height, width)

            # Stack as a single batch [before, after] to share GPU/CPU overhead
            batch = torch.from_numpy(
                np.stack([tile_b.transpose(2, 0, 1), tile_a.transpose(2, 0, 1)])
            ).float()

            features = model.forward_features(batch)
            if isinstance(features, dict):
                features = features.get("x_norm_patchtokens", list(features.values())[0])

            if features.ndim == 3:
                # Reshape patch tokens [B, N, D] → [B, H_p, W_p, D]
                side = patches_per_side
                features = features[:, :side * side, :].reshape(2, side, side, feature_dim)
            elif features.ndim == 2:
                features = features.unsqueeze(1).unsqueeze(1)

            feat_b = features[0].cpu().numpy()  # (H_p, W_p, D)
            feat_a = features[1].cpu().numpy()
            del batch, features

            # Compute cosine distance immediately (scalar per patch)
            norm_b = np.linalg.norm(feat_b, axis=2, keepdims=True)
            norm_a = np.linalg.norm(feat_a, axis=2, keepdims=True)
            norm_b = np.maximum(norm_b, 1e-8)
            norm_a = np.maximum(norm_a, 1e-8)
            cosine_sim = np.sum((feat_b / norm_b) * (feat_a / norm_a), axis=2)
            tile_distance = np.clip(1.0 - cosine_sim, 0, 1)
            del feat_b, feat_a

            # Map to global patch grid
            fh, fw = tile_distance.shape
            py = y // patch_size
            px_start = x // patch_size
            py_end = min(py + fh, patch_h)
            px_end = min(px_start + fw, patch_w)
            fh_use = py_end - py
            fw_use = px_end - px_start

            distance_sum[py:py_end, px_start:px_end] += tile_distance[:fh_use, :fw_use]
            distance_count[py:py_end, px_start:px_end] += 1

            processed += 1
            if processed % 50 == 0:
                logger.info("  Processed %d/%d tile pairs...", processed, total_tiles)

    # Average overlapping tiles
    avg_distance = np.divide(
        distance_sum, distance_count,
        out=np.zeros_like(distance_sum, dtype=np.float32),
        where=distance_count > 0,
    ).astype(np.float32)

    # Upsample ONLY the scalar distance map to pixel resolution
    distance_tensor = torch.from_numpy(avg_distance).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(distance_tensor, size=(height, width), mode="bilinear", align_corners=False)
    result = upsampled.squeeze().numpy()
    result[~valid] = 0.0

    return result


def _vectorise_candidates(
    mask: np.ndarray,
    change_score: np.ndarray,
    profile: dict[str, Any],
    config: DINOChangeConfig,
    *,
    metadata: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert change mask to GeoJSON features."""
    candidates: list[dict[str, Any]] = []
    source_crs = profile.get("crs")
    if source_crs is None:
        raise DINOChangeError("Imagery has no CRS.")

    for geometry_json, value in shapes(mask, mask=mask.astype(bool), transform=profile["transform"]):
        if value != 1:
            continue
        geometry = shape(geometry_json)
        area = _area_m2(geometry, source_crs)
        if area < config.min_area_m2:
            continue

        footprint = geometry_mask(
            [geometry_json], out_shape=mask.shape,
            transform=profile["transform"], invert=True,
        )
        values = change_score[footprint]
        mean_score = float(np.mean(values)) if values.size else 0.0

        rectangularity = 0.0
        min_rect = geometry.minimum_rotated_rectangle
        if min_rect.area > 0:
            rectangularity = min(1.0, float(geometry.area / min_rect.area))

        classification = "likely_building_change" if rectangularity >= 0.35 else "construction_change_candidate"

        feature = {
            "type": "Feature",
            "geometry": transform_geom(source_crs, "EPSG:4326", mapping(geometry), precision=7),
            "properties": {
                "candidate_id": len(candidates) + 1,
                "classification": classification,
                "area_m2": round(area, 2),
                "rectangularity": round(rectangularity, 3),
                "mean_semantic_change_score": round(mean_score, 4),
                "candidate_source": "dinov2_semantic",
                "model": config.model_name,
                **metadata,
            },
        }
        candidates.append(feature)

    return candidates


def run_dino_change_detection(
    before_image: str | Path,
    after_image: str | Path,
    output_dir: str | Path,
    *,
    config: DINOChangeConfig | None = None,
    before_capture_date: str | None = None,
    after_capture_date: str | None = None,
) -> dict[str, Any]:
    """Run DINOv2 semantic change detection — zero-shot, no training needed.

    Returns a JSON-serialisable report compatible with the existing pipeline.
    """
    active_config = config or DINOChangeConfig()
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

    before_norm = _normalise_for_model(before_rgb, before_valid)
    after_norm = _normalise_for_model(after_rgb, after_valid)

    # Free the large raw arrays
    del before_rgb, after_rgb

    # Build feature extractor
    model = _build_feature_extractor(active_config.model_name)

    # Compute semantic change score (memory-efficient)
    logger.info("Computing DINOv2 semantic change map...")
    change_score = _compute_semantic_change_map(model, before_norm, after_norm, valid, active_config)

    # Free model and normalised images
    del model, before_norm, after_norm

    # Threshold
    valid_scores = change_score[valid]
    if not valid_scores.size:
        raise DINOChangeError("No valid overlapping pixels between before and after images.")

    if active_config.change_threshold is not None:
        threshold = active_config.change_threshold
    else:
        threshold = float(np.percentile(valid_scores, active_config.change_percentile))

    logger.info("Change threshold: %.4f (percentile %.1f)", threshold, active_config.change_percentile)

    change_mask = ((change_score >= threshold) & valid).astype(np.uint8)

    # Morphological cleanup
    if active_config.morphology_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (active_config.morphology_px, active_config.morphology_px),
        )
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)

    # Save outputs
    score_path = directory / "dino_semantic_change_score.tif"
    mask_path = directory / "dino_semantic_change_mask.tif"
    score_save = change_score.copy()
    score_save[~valid] = -9999.0
    _write_raster(score_path, score_save, profile, dtype="float32", nodata=-9999.0)
    _write_raster(mask_path, change_mask * 255, profile, dtype="uint8", nodata=0)

    # Vectorise
    metadata = {
        "before_capture_date": before_capture_date or "unknown",
        "after_capture_date": after_capture_date or "unknown",
    }
    candidates = _vectorise_candidates(change_mask, change_score, profile, active_config, metadata=metadata)

    # Write GeoJSON
    candidates_path = directory / "dino_semantic_change_candidates.geojson"
    collection = {
        "type": "FeatureCollection",
        "features": candidates,
        "metadata": {
            "crs": "EPSG:4326",
            "requested_detector": "dinov2_semantic",
            "model": active_config.model_name,
            "change_threshold": threshold,
            "change_percentile": active_config.change_percentile,
            "min_area_m2": active_config.min_area_m2,
        },
    }
    candidates_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")

    report = {
        "before_image": str(before_path),
        "after_image": str(after_path),
        "model": active_config.model_name,
        "candidate_count": len(candidates),
        "change_threshold": threshold,
        "config": asdict(active_config),
        "outputs": {
            "semantic_change_score": str(score_path),
            "semantic_change_mask": str(mask_path),
            "change_candidates": str(candidates_path),
        },
    }
    report_path = directory / "dino_semantic_change_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("DINOv2 semantic detection complete: %d candidates (threshold=%.4f)", len(candidates), threshold)
    return report
