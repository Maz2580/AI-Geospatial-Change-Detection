"""DINOv3 building-footprint segmentation as independent dated evidence.

This adapter runs ``hotosm/dinov3s-buildings`` on each date separately and
compares the two building-footprint sets.  It deliberately does *not* label a
polygon as confirmed construction: a roof segmentation can be wrong, and a
building has to be absent from the earlier date before it is a new-building
candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from .detector import _area_m2, _read_new_image, _reproject_rgb, _write_raster
from .footprints import write_comparison_outputs
from .regularisation import RegularisationConfig, regularise_geometries


DINO_BUILDINGS_REPOSITORY = "hotosm/dinov3s-buildings"
DINO_BUILDINGS_FILENAME = "model.onnx"
DINO_BUILDINGS_THRESHOLD = 0.4371

# The published graph wraps a frozen DINOv3 ViT-S/16 backbone, which is trained
# on ImageNet-normalised input. Feeding it raw 0-255 saturates the logits and
# collapses building coverage to ~2% of a fully built-out suburban scene.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _normalise_tile(tile: np.ndarray) -> np.ndarray:
    """Convert a 0-255 RGB tile to the backbone's expected input statistics."""
    return (tile / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD


class DinoBuildingsError(ValueError):
    """Raised when the optional DINO building-footprint model cannot run."""


class InferenceSession(Protocol):
    def run(self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class DinoBuildingsConfig:
    """Publisher-aligned inference and vectorisation settings."""

    threshold: float = DINO_BUILDINGS_THRESHOLD
    window_px: int = 256
    stride_px: int = 192
    min_area_m2: float = 10.0
    simplify_m: float = 0.25
    regularisation: RegularisationConfig = field(default_factory=RegularisationConfig)

    def validate(self) -> None:
        if not 0 < self.threshold < 1:
            raise DinoBuildingsError("DINO building threshold must be between zero and one.")
        if self.window_px != 256:
            raise DinoBuildingsError("This published ONNX model requires a 256-pixel input window.")
        if not 0 < self.stride_px <= self.window_px:
            raise DinoBuildingsError("DINO stride must be greater than zero and no larger than the window.")
        if self.min_area_m2 <= 0 or self.simplify_m < 0:
            raise DinoBuildingsError("Minimum area must be positive and simplify distance cannot be negative.")
        self.regularisation.validate()


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    """Return starts that cover both edges without leaving a seam."""
    if length <= 0:
        raise DinoBuildingsError("Imagery dimensions must be positive.")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _building_probability(logits: np.ndarray) -> np.ndarray:
    """Convert the ONNX three-class logits to the model's building channel.

    The published graph exposes three logits although its model card describes
    a building probability.  Class zero is its building channel; classes one
    and two are auxiliary/background outputs.  This is kept explicit rather
    than silently applying a sigmoid to an arbitrary channel.
    """
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 3:
        raise DinoBuildingsError(
            "Expected DINO model logits shaped [1, 3, 256, 256]; the selected ONNX file is not supported."
        )
    shifted = logits.astype(np.float32) - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probability = exponent / np.sum(exponent, axis=1, keepdims=True)
    return probability[0, 0]


def _segment_probability(
    session: InferenceSession,
    rgb: np.ndarray,
    valid: np.ndarray,
    config: DinoBuildingsConfig,
) -> np.ndarray:
    """Run full-resolution overlapping-window inference on RGB in 0--255 space.

    Tiles are assembled in 0-255 space and normalised immediately before
    inference, so edge padding stays black rather than becoming a live value.
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise DinoBuildingsError("DINO building inference requires an RGB array shaped [3, height, width].")
    if valid.shape != rgb.shape[1:]:
        raise DinoBuildingsError("RGB and valid-pixel masks have incompatible dimensions.")
    config.validate()
    height, width = valid.shape
    probability_total = np.zeros((height, width), dtype=np.float32)
    contribution_count = np.zeros((height, width), dtype=np.uint16)
    image = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0).clip(0.0, 255.0)
    for top in _window_starts(height, config.window_px, config.stride_px):
        for left in _window_starts(width, config.window_px, config.stride_px):
            bottom = min(top + config.window_px, height)
            right = min(left + config.window_px, width)
            tile = np.zeros((3, config.window_px, config.window_px), dtype=np.float32)
            tile[:, : bottom - top, : right - left] = image[:, top:bottom, left:right]
            outputs = session.run(None, {"image": _normalise_tile(tile)[None]})
            if not outputs:
                raise DinoBuildingsError("The DINO ONNX session returned no outputs.")
            tile_probability = _building_probability(np.asarray(outputs[0]))
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
    config: DinoBuildingsConfig,
    *,
    capture_date: str | None,
) -> dict[str, Any]:
    """Threshold the building probability and preserve dated source evidence."""
    source_crs = profile.get("crs")
    if source_crs is None:
        raise DinoBuildingsError("The imagery GeoTIFF has no CRS; DINO footprints cannot be georeferenced.")
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
                    "source_model": DINO_BUILDINGS_REPOSITORY,
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
            "requested_detector": "dino_buildings",
            "source_model": DINO_BUILDINGS_REPOSITORY,
            "probability_threshold": config.threshold,
            "min_area_m2": config.min_area_m2,
            "regularisation": asdict(config.regularisation),
            "capture_date": capture_date or "unknown",
            "evidence_role": "dated_object_footprint",
        },
    }


def _write_geojson(path: Path, collection: dict[str, Any]) -> None:
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def resolve_model_path(
    model_path: str | Path | None = None,
    *,
    cache_dir: str | Path = ".cache/huggingface",
) -> Path:
    """Return a local model path, downloading only to the project's cache when needed."""
    if model_path is not None:
        selected = Path(model_path)
        if not selected.is_file():
            raise DinoBuildingsError(f"DINO ONNX model was not found: {selected}")
        return selected
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DinoBuildingsError(
            "Install optional DINO dependencies with `pip install -r requirements-models.txt`, "
            "or pass --model-path to an existing model.onnx file."
        ) from exc
    try:
        return Path(
            hf_hub_download(
                repo_id=DINO_BUILDINGS_REPOSITORY,
                filename=DINO_BUILDINGS_FILENAME,
                cache_dir=str(cache_dir),
                token=os.getenv("HF_TOKEN") or None,
            )
        )
    except Exception as exc:  # pragma: no cover - provider/network dependent
        raise DinoBuildingsError(f"Could not obtain {DINO_BUILDINGS_REPOSITORY}: {exc}") from exc


def _load_session(model_path: Path) -> InferenceSession:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DinoBuildingsError("Install optional DINO dependencies with `pip install -r requirements-models.txt`.") from exc
    try:
        # CPU is intentional: it is dependable on a fresh Windows venv.  A
        # CUDA provider can later be enabled when its matching CUDA runtime is
        # installed; the method and outputs remain identical.
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover - host runtime dependent
        raise DinoBuildingsError(f"Could not load DINO ONNX model {model_path}: {exc}") from exc


def run_dino_building_comparison(
    before_image: str | Path,
    after_image: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    model_cache: str | Path = ".cache/huggingface",
    config: DinoBuildingsConfig | None = None,
    before_capture_date: str | None = None,
    after_capture_date: str | None = None,
    match_distance_m: float = 6.0,
    match_iou: float = 0.10,
    extension_outside_fraction: float = 0.25,
) -> dict[str, Any]:
    """Extract DINO building footprints on both dates and compare them.

    The returned candidates are independent *object evidence* for review.  A
    shadow, car, pool, driveway, or garden is outside the scope of this model.
    """
    active_config = config or DinoBuildingsConfig()
    active_config.validate()
    before_path, after_path, directory = Path(before_image), Path(after_image), Path(output_dir)
    if not before_path.is_file() or not after_path.is_file():
        raise FileNotFoundError("Both before and after RGB GeoTIFFs must exist.")
    directory.mkdir(parents=True, exist_ok=True)
    after_rgb, after_valid, profile = _read_new_image(after_path)
    before_rgb, before_valid = _reproject_rgb(before_path, profile)
    session = _load_session(resolve_model_path(model_path, cache_dir=model_cache))
    before_probability = _segment_probability(session, before_rgb, before_valid, active_config)
    after_probability = _segment_probability(session, after_rgb, after_valid, active_config)
    before_collection = _footprint_collection(
        before_probability, before_valid, profile, active_config, capture_date=before_capture_date
    )
    after_collection = _footprint_collection(
        after_probability, after_valid, profile, active_config, capture_date=after_capture_date
    )
    before_probability_path = directory / "before_dino_building_probability.tif"
    after_probability_path = directory / "after_dino_building_probability.tif"
    before_footprints_path = directory / "before_dino_building_footprints.geojson"
    after_footprints_path = directory / "after_dino_building_footprints.geojson"
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
        "model": DINO_BUILDINGS_REPOSITORY,
        "runtime_provider": "CPUExecutionProvider",
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
        "warning": "DINO footprints are review evidence, not confirmed construction. It does not classify pools, gardens, driveways, vehicles, or shadows.",
    }
    (directory / "dino_buildings_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
