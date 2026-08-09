"""Run one fixed, zero-shot ChangeStar building-change baseline.

This runner deliberately keeps the three model outputs separate:

* binary change probability;
* before-date building probability; and
* after-date building probability.

It is intended for a bounded validation run, not as a claim that the
after-date segmentation is automatically a survey-grade roof footprint.
The model environment is intentionally external to the project's ordinary
``venv`` because ChangeStar needs PyTorch and its own model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ID = "Changen2-ChangeStar1x256:s1c1_cstar_vitb"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path, help="Fixed before-date RGB GeoTIFF.")
    parser.add_argument("--after", required=True, type=Path, help="Fixed after-date RGB GeoTIFF on the same grid.")
    parser.add_argument("--weights", required=True, type=Path, help="Local published s1c1_cstar_vitb_1x256.pth weight.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for untracked runtime outputs.")
    parser.add_argument("--tile-size", type=int, default=256, help="Model patch size. Fixed default matches the published 1x256 model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed probability threshold used only for inspection vectors.")
    parser.add_argument("--min-area-m2", type=float, default=20.0, help="Minimum vector area for review candidates.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Inference device. The baseline uses CPU unless explicitly changed.")
    args = parser.parse_args()
    if args.tile_size <= 0:
        parser.error("--tile-size must be positive.")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1.")
    if args.min_area_m2 < 0:
        parser.error("--min-area-m2 must be non-negative.")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pair(before_path: Path, after_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import rasterio

    with rasterio.open(before_path) as before_source, rasterio.open(after_path) as after_source:
        if before_source.count < 3 or after_source.count < 3:
            raise ValueError("ChangeStar baseline requires at least three RGB bands in each GeoTIFF.")
        same_grid = (
            before_source.width == after_source.width
            and before_source.height == after_source.height
            and before_source.crs == after_source.crs
            and before_source.transform == after_source.transform
        )
        if not same_grid:
            raise ValueError(
                "The fixed baseline does not resample or register imagery. "
                "Supply matching before/after GeoTIFF grids."
            )
        before = np.moveaxis(before_source.read(indexes=(1, 2, 3)), 0, -1)
        after = np.moveaxis(after_source.read(indexes=(1, 2, 3)), 0, -1)
        profile = after_source.profile.copy()
    if before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError("The fixed baseline expects uint8 RGB imagery.")
    return before, after, profile


def _load_model(weights_path: Path, device_name: str):
    import torch
    from torchange.models import changen2

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this model environment.")
    model = changen2.changestar_1x256("vitb", "s1c1", changen2_pretrained=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    return model, device


def _probability_array(value: Any) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 1:
        raise RuntimeError(f"Unexpected ChangeStar output shape: {array.shape}")
    return array[0, 0]


def _infer(
    before: np.ndarray,
    after: np.ndarray,
    model: Any,
    device: Any,
    *,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import albumentations as transforms
    from albumentations.pytorch import ToTensorV2
    import torch

    height, width = before.shape[:2]
    change = np.zeros((height, width), dtype=np.float32)
    before_building = np.zeros((height, width), dtype=np.float32)
    after_building = np.zeros((height, width), dtype=np.float32)
    preprocess = transforms.Compose(
        [transforms.Normalize(), ToTensorV2()],
        additional_targets={"image2": "image"},
    )
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            before_tile = before[top:bottom, left:right]
            after_tile = after[top:bottom, left:right]
            pad_bottom = tile_size - before_tile.shape[0]
            pad_right = tile_size - before_tile.shape[1]
            if pad_bottom or pad_right:
                pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))
                before_tile = np.pad(before_tile, pad_width, mode="edge")
                after_tile = np.pad(after_tile, pad_width, mode="edge")
            data = preprocess(image=before_tile, image2=after_tile)
            batch = torch.cat([data["image"], data["image2"]], dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(batch)
            tile_height, tile_width = bottom - top, right - left
            change[top:bottom, left:right] = _probability_array(output.change_prediction)[:tile_height, :tile_width]
            before_building[top:bottom, left:right] = _probability_array(output.t1_semantic_prediction)[:tile_height, :tile_width]
            after_building[top:bottom, left:right] = _probability_array(output.t2_semantic_prediction)[:tile_height, :tile_width]
    return change, before_building, after_building


def _write_raster(path: Path, values: np.ndarray, profile: dict[str, Any], *, dtype: str, nodata: float | int) -> None:
    import rasterio

    output_profile = profile.copy()
    output_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(values.astype(dtype), 1)


def _vectorize(mask: np.ndarray, profile: dict[str, Any], *, min_area_m2: float, candidate_kind: str) -> dict[str, Any]:
    from rasterio.features import shapes
    from shapely.geometry import shape

    features: list[dict[str, Any]] = []
    for geometry, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=profile["transform"]):
        if not value:
            continue
        polygon = shape(geometry)
        if polygon.area < min_area_m2:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "candidate_id": len(features) + 1,
                    "candidate_kind": candidate_kind,
                    "area_m2": round(float(polygon.area), 2),
                },
                "geometry": geometry,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _write_geojson(path: Path, collection: dict[str, Any]) -> None:
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if not args.before.is_file() or not args.after.is_file() or not args.weights.is_file():
        raise FileNotFoundError("--before, --after, and --weights must all exist.")
    args.output.mkdir(parents=True, exist_ok=True)
    before, after, profile = _load_pair(args.before, args.after)
    model, device = _load_model(args.weights, args.device)
    change, before_building, after_building = _infer(
        before,
        after,
        model,
        device,
        tile_size=args.tile_size,
    )
    threshold = args.threshold
    change_mask = change >= threshold
    before_mask = before_building >= threshold
    after_mask = after_building >= threshold
    after_only_mask = after_mask & ~before_mask
    agreement_mask = change_mask & after_only_mask

    _write_raster(args.output / "change_probability.tif", change, profile, dtype="float32", nodata=-9999.0)
    _write_raster(args.output / "before_building_probability.tif", before_building, profile, dtype="float32", nodata=-9999.0)
    _write_raster(args.output / "after_building_probability.tif", after_building, profile, dtype="float32", nodata=-9999.0)
    _write_raster(args.output / "change_mask.tif", change_mask, profile, dtype="uint8", nodata=0)
    _write_raster(args.output / "before_building_mask.tif", before_mask, profile, dtype="uint8", nodata=0)
    _write_raster(args.output / "after_building_mask.tif", after_mask, profile, dtype="uint8", nodata=0)
    _write_raster(args.output / "after_only_building_mask.tif", after_only_mask, profile, dtype="uint8", nodata=0)
    _write_raster(args.output / "building_change_agreement_mask.tif", agreement_mask, profile, dtype="uint8", nodata=0)

    candidates = {
        "change_candidates.geojson": _vectorize(change_mask, profile, min_area_m2=args.min_area_m2, candidate_kind="change"),
        "after_building_footprints.geojson": _vectorize(after_mask, profile, min_area_m2=args.min_area_m2, candidate_kind="after_building"),
        "after_only_building_candidates.geojson": _vectorize(after_only_mask, profile, min_area_m2=args.min_area_m2, candidate_kind="after_only_building"),
        "building_change_agreement_candidates.geojson": _vectorize(agreement_mask, profile, min_area_m2=args.min_area_m2, candidate_kind="change_and_after_only_building"),
    }
    for name, collection in candidates.items():
        _write_geojson(args.output / name, collection)
    report = {
        "model": MODEL_ID,
        "model_packages": {"torchange": version("torchange"), "torch": version("torch"), "albumentations": version("albumentations")},
        "device": str(device),
        "parameters": {"tile_size": args.tile_size, "threshold": threshold, "min_area_m2": args.min_area_m2},
        "inputs": {"before": str(args.before), "after": str(args.after), "before_sha256": _sha256(args.before), "after_sha256": _sha256(args.after), "weights": str(args.weights), "weights_sha256": _sha256(args.weights)},
        "output_summary": {
            "change_pixels": int(change_mask.sum()),
            "before_building_pixels": int(before_mask.sum()),
            "after_building_pixels": int(after_mask.sum()),
            "after_only_building_pixels": int(after_only_mask.sum()),
            "building_change_agreement_pixels": int(agreement_mask.sum()),
            "candidate_counts": {name.removesuffix(".geojson"): len(collection["features"]) for name, collection in candidates.items()},
        },
        "interpretation": {
            "change": "Observed semantic change evidence, not a roof outline.",
            "after_building": "After-date building segmentation proposal, not automatically a survey-grade footprint.",
            "after_only_building": "After-date building pixels absent from the before-date segmentation.",
            "building_change_agreement": "Intersection of change and after-only-building masks; use as a conservative review prompt, not automatic confirmation.",
        },
    }
    (args.output / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["output_summary"], indent=2))


if __name__ == "__main__":
    main()
