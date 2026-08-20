"""WHU Building-footprint segmentation.

This adapter runs `giswqs/whu-building-unetplusplus-efficientnet-b4` to extract
highly precise building footprints that ignore roads, paths, and vegetation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
import torch

from .detector import DetectionConfig, _area_m2, _read_new_image, _reproject_rgb, _shadow_mask, _write_raster
from .footprints import write_comparison_outputs
from .regularisation import RegularisationConfig, regularise_geometries

WHU_BUILDINGS_REPOSITORY = "giswqs/whu-building-unetplusplus-efficientnet-b4"
WHU_BUILDINGS_FILENAME = "model.pth"
WHU_BUILDINGS_THRESHOLD = 0.50

class WhuBuildingsError(ValueError):
    """Raised when the WHU building-footprint model cannot run."""

@dataclass(frozen=True)
class WhuBuildingsConfig:
    threshold: float = WHU_BUILDINGS_THRESHOLD
    window_px: int = 2048
    stride_px: int = 1536
    downscale_factor: float = 4.0
    min_area_m2: float = 10.0
    simplify_m: float = 0.25
    regularisation: RegularisationConfig = field(default_factory=RegularisationConfig)

    def validate(self) -> None:
        if not 0 < self.threshold < 1:
            raise WhuBuildingsError("WHU building threshold must be between zero and one.")
        if not 0 < self.stride_px <= self.window_px:
            raise WhuBuildingsError("WHU stride must be greater than zero and no larger than the window.")
        if self.downscale_factor <= 0:
            raise WhuBuildingsError("Downscale factor must be positive.")
        if self.min_area_m2 <= 0 or self.simplify_m < 0:
            raise WhuBuildingsError("Minimum area must be positive and simplify distance cannot be negative.")
        self.regularisation.validate()

def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= 0:
        raise WhuBuildingsError("Imagery dimensions must be positive.")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts

def resolve_model_path(
    model_path: str | Path | None = None,
    *,
    cache_dir: str | Path = ".cache/huggingface",
) -> Path:
    if model_path is not None:
        selected = Path(model_path)
        if not selected.is_file():
            raise WhuBuildingsError(f"WHU model was not found: {selected}")
        return selected
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise WhuBuildingsError("Install huggingface_hub.") from exc
    try:
        return Path(
            hf_hub_download(
                repo_id=WHU_BUILDINGS_REPOSITORY,
                filename=WHU_BUILDINGS_FILENAME,
                cache_dir=str(cache_dir),
                token=os.getenv("HF_TOKEN") or None,
            )
        )
    except Exception as exc:
        raise WhuBuildingsError(f"Could not obtain {WHU_BUILDINGS_REPOSITORY}: {exc}") from exc

def _load_model(model_path: Path) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise WhuBuildingsError("Install segmentation-models-pytorch.") from exc
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=2,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def _segment_probability(
    model: torch.nn.Module,
    rgb: np.ndarray,
    valid: np.ndarray,
    config: WhuBuildingsConfig,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise WhuBuildingsError("WHU building inference requires an RGB array shaped [3, height, width].")
    if valid.shape != rgb.shape[1:]:
        raise WhuBuildingsError("RGB and valid-pixel masks have incompatible dimensions.")
    config.validate()
    height, width = valid.shape
    probability_total = np.zeros((height, width), dtype=np.float32)
    contribution_count = np.zeros((height, width), dtype=np.uint16)
    
    image = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0).clip(0.0, 255.0) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = (image - mean) / std

    import torch.nn.functional as F

    for top in _window_starts(height, config.window_px, config.stride_px):
        for left in _window_starts(width, config.window_px, config.stride_px):
            bottom = min(top + config.window_px, height)
            right = min(left + config.window_px, width)
            tile = np.zeros((3, config.window_px, config.window_px), dtype=np.float32)
            tile[:, : bottom - top, : right - left] = image[:, top:bottom, left:right]
            
            with torch.no_grad():
                tensor = torch.from_numpy(tile).unsqueeze(0)
                
                # Downsample to expected model scale
                model_size = int(config.window_px / config.downscale_factor)
                tensor_small = F.interpolate(tensor, size=(model_size, model_size), mode="bilinear", align_corners=False)
                
                outputs = model(tensor_small)
                probs = torch.softmax(outputs, dim=1)
                
                # Upsample back to original scale
                probs_large = F.interpolate(probs[:, 1:2, :, :], size=(config.window_px, config.window_px), mode="bilinear", align_corners=False)
                tile_probability = probs_large[0, 0].numpy()
                
            probability_total[top:bottom, left:right] += tile_probability[: bottom - top, : right - left]
            contribution_count[top:bottom, left:right] += 1
            
    result = np.divide(
        probability_total,
        contribution_count,
        out=np.zeros_like(probability_total),
        where=contribution_count > 0,
    )
    result[~valid] = 0.0
    return result

def _footprint_collection(
    probability: np.ndarray,
    valid: np.ndarray,
    profile: dict[str, Any],
    config: WhuBuildingsConfig,
    *,
    capture_date: str | None,
) -> dict[str, Any]:
    source_crs = profile.get("crs")
    if source_crs is None:
        raise WhuBuildingsError("The imagery GeoTIFF has no CRS; footprints cannot be georeferenced.")
    mask = (probability >= config.threshold) & valid
    raw_geometries: list[Any] = []
    for geometry_json, value in shapes(mask.astype(np.uint8), mask=mask, transform=profile["transform"]):
        if value != 1:
            continue
        geometry = shape(geometry_json)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty:
            continue
        if config.simplify_m and source_crs.is_projected:
            geometry = geometry.simplify(config.simplify_m, preserve_topology=True)
        if geometry.is_empty or _area_m2(geometry, source_crs) < config.min_area_m2:
            continue
        raw_geometries.append(geometry)

    geometries = regularise_geometries(raw_geometries, source_crs, config.regularisation)
    features: list[dict[str, Any]] = []
    for geometry in geometries:
        area_m2 = _area_m2(geometry, source_crs)
        if area_m2 < config.min_area_m2:
            continue
        footprint = geometry_mask([mapping(geometry)], out_shape=mask.shape, transform=profile["transform"], invert=True)
        values = probability[footprint & valid]
        if not values.size:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": transform_geom(source_crs, "EPSG:4326", mapping(geometry), precision=8),
                "properties": {
                    "candidate_id": len(features) + 1,
                    "classification": "building_footprint",
                    "area_m2": round(float(area_m2), 1),
                    "mean_building_probability": round(float(np.mean(values)), 4),
                    "max_building_probability": round(float(np.max(values)), 4),
                    "capture_date": capture_date or "unknown",
                    "source_model": WHU_BUILDINGS_REPOSITORY,
                    "evidence_role": "dated_object_footprint",
                },
            }
        )
    features.sort(key=lambda feature: -float(feature["properties"]["area_m2"]))
    for candidate_id, feature in enumerate(features, start=1):
        feature["properties"]["candidate_id"] = candidate_id
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "crs": "EPSG:4326",
            "requested_detector": "whu_buildings",
            "source_model": WHU_BUILDINGS_REPOSITORY,
            "probability_threshold": config.threshold,
            "min_area_m2": config.min_area_m2,
            "regularisation": asdict(config.regularisation),
            "capture_date": capture_date or "unknown",
            "evidence_role": "dated_object_footprint",
        },
    }

def _write_geojson(path: Path, collection: dict[str, Any]) -> None:
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")

def run_whu_building_comparison(
    before_image: str | Path,
    after_image: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    model_cache: str | Path = ".cache/huggingface",
    config: WhuBuildingsConfig | None = None,
    before_capture_date: str | None = None,
    after_capture_date: str | None = None,
    match_distance_m: float = 6.0,
    match_iou: float = 0.10,
    extension_outside_fraction: float = 0.25,
) -> dict[str, Any]:
    active_config = config or WhuBuildingsConfig()
    active_config.validate()
    before_path, after_path, directory = Path(before_image), Path(after_image), Path(output_dir)
    if not before_path.is_file() or not after_path.is_file():
        raise FileNotFoundError("Both before and after RGB GeoTIFFs must exist.")
    directory.mkdir(parents=True, exist_ok=True)
    after_rgb, after_valid, profile = _read_new_image(after_path)
    before_rgb, before_valid = _reproject_rgb(before_path, profile)
    
    model = _load_model(resolve_model_path(model_path, cache_dir=model_cache))
    before_probability = _segment_probability(model, before_rgb, before_valid, active_config)
    after_probability = _segment_probability(model, after_rgb, after_valid, active_config)
    
    # Subtract cast shadows from building probabilities
    before_shadow = _shadow_mask(before_rgb, before_valid, profile, DetectionConfig())
    after_shadow = _shadow_mask(after_rgb, after_valid, profile, DetectionConfig())
    if np.any(before_shadow):
        before_probability[before_shadow] = 0.0
    if np.any(after_shadow):
        after_probability[after_shadow] = 0.0
    
    before_collection = _footprint_collection(
        before_probability, before_valid, profile, active_config, capture_date=before_capture_date
    )
    after_collection = _footprint_collection(
        after_probability, after_valid, profile, active_config, capture_date=after_capture_date
    )
    
    before_probability_path = directory / "before_whu_building_probability.tif"
    after_probability_path = directory / "after_whu_building_probability.tif"
    before_footprints_path = directory / "before_whu_building_footprints.geojson"
    after_footprints_path = directory / "after_whu_building_footprints.geojson"
    
    before_for_write = before_probability.copy()
    after_for_write = after_probability.copy()
    before_for_write[~before_valid] = -9999.0
    after_for_write[~after_valid] = -9999.0
    
    _write_raster(before_probability_path, before_for_write, profile, dtype="float32", nodata=-9999.0)
    _write_raster(after_probability_path, after_for_write, profile, dtype="float32", nodata=-9999.0)
    _write_geojson(before_footprints_path, before_collection)
    _write_geojson(after_footprints_path, after_collection)
    
    comparison = write_comparison_outputs(
        directory,
        before_collection,
        after_collection,
        match_distance_m=match_distance_m,
        match_iou=match_iou,
        extension_outside_fraction=extension_outside_fraction,
        min_area_m2=active_config.min_area_m2,
    )
    report = {
        "before_image": str(before_path),
        "after_image": str(after_path),
        "model": WHU_BUILDINGS_REPOSITORY,
        "before_footprint_count": len(before_collection["features"]),
        "after_footprint_count": len(after_collection["features"]),
        "config": asdict(active_config),
        "comparison": comparison,
        "outputs": {
            "before_probability": str(before_probability_path),
            "after_probability": str(after_probability_path),
            "before_footprints": str(before_footprints_path),
            "after_footprints": str(after_footprints_path),
            **comparison["outputs"],
        },
    }
    (directory / "whu_buildings_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
