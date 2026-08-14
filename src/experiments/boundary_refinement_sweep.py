"""Does placing the boundary from the image beat thresholding the probability?

The baseline to beat, measured on the same chips: boundary F1 at 0.25 m of 0.511
at threshold 0.35 with regularisation off. Refinement is only worth adopting if
it improves that without losing detections.

Inference is cached, so each variant costs seconds plus the GrabCut passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import replace
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

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from building_change.boundary_refinement import (  # noqa: E402
    BoundaryRefinementConfig,
    refine_building_mask,
    refinement_report,
)
from building_change.detector import _read_new_image  # noqa: E402
from building_change.dino_buildings import (  # noqa: E402
    DinoBuildingsConfig,
    _load_session,
    _mask_collection,
    cached_building_probability,
    resolve_model_path,
)
from building_change.outline_metrics import (  # noqa: E402
    VICTORIAN_METRIC_CRS,
    match_instances,
    score_pair,
    summarise,
)

CLEAN_CHIPS = ("909", "912", "913")


def _predictions(collection: dict[str, Any]) -> gpd.GeoDataFrame:
    if not collection["features"]:
        return gpd.GeoDataFrame(geometry=[], crs=VICTORIAN_METRIC_CRS)
    return gpd.GeoDataFrame(
        geometry=[shape(feature["geometry"]) for feature in collection["features"]],
        crs="EPSG:4326",
    ).to_crs(VICTORIAN_METRIC_CRS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", type=Path, default=Path("data/benchmarks/uc5_goldset_v2"))
    parser.add_argument("--chips", nargs="*", default=list(CLEAN_CHIPS))
    parser.add_argument("--cache", type=Path, default=Path("data/output/goldset_probability_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/benchmarks/uc5_goldset_v2/boundary_refinement_sweep.json")
    )
    parser.add_argument("--model-cache", default="data/output/model_cache/huggingface")
    parser.add_argument("--threshold", type=float, default=0.35, help="The best threshold measured so far.")
    args = parser.parse_args()

    base = replace(
        DinoBuildingsConfig(),
        threshold=args.threshold,
        regularisation=replace(DinoBuildingsConfig().regularisation, enabled=False),
    )
    session = _load_session(resolve_model_path(None, cache_dir=args.model_cache))

    chips: dict[str, dict[str, Any]] = {}
    for chip_id in args.chips:
        chip = args.goldset / f"chip_{chip_id}.tif"
        probability, valid, profile = cached_building_probability(chip, args.cache, session, base)
        rgb, _, _ = _read_new_image(chip)
        labels = gpd.read_file(args.goldset / f"labels_{chip_id}.shp").to_crs(VICTORIAN_METRIC_CRS).reset_index(drop=True)
        chips[chip_id] = {
            "probability": probability,
            "valid": valid,
            "rgb": rgb,
            "profile": profile,
            "labels": labels,
            "pixel_size_m": abs(profile["transform"].a),
        }
        print(f"chip {chip_id}: ready, {len(labels)} labels, {chips[chip_id]['pixel_size_m']:.4f} CRS units/px")

    label_total = sum(len(chip["labels"]) for chip in chips.values())
    variants: dict[str, BoundaryRefinementConfig | None] = {
        "threshold only (baseline)": None,
        "refined, core 0.75 / bg 0.10": BoundaryRefinementConfig(enabled=True),
        "refined, core 0.85 / bg 0.05": BoundaryRefinementConfig(
            enabled=True, core_probability=0.85, background_probability=0.05
        ),
        "refined, core 0.60 / bg 0.20": BoundaryRefinementConfig(
            enabled=True, core_probability=0.60, background_probability=0.20
        ),
        "refined, wider context 8 m": BoundaryRefinementConfig(enabled=True, padding_m=8.0),
    }

    print()
    header = (
        f"{'variant':<30} {'pred':>5} {'matched':>8} {'missed':>7} {'merges':>7} "
        f"{'bF1@0.25':>9} {'bF1@0.5':>8} {'IoU':>6} {'H95 m':>7} {'verts':>6}"
    )
    print(header)
    print("-" * len(header))

    results: list[dict[str, Any]] = []
    for name, refinement in variants.items():
        pooled: list[Any] = []
        predicted = matched = missed = merges = 0
        movement: list[dict[str, Any]] = []
        for chip_id, chip in chips.items():
            threshold_mask = (chip["probability"] >= base.threshold) & chip["valid"]
            if refinement is None:
                mask = threshold_mask
            else:
                mask = refine_building_mask(
                    chip["probability"],
                    chip["valid"],
                    chip["rgb"],
                    threshold=base.threshold,
                    pixel_size_m=chip["pixel_size_m"],
                    config=refinement,
                )
                movement.append(refinement_report(threshold_mask, mask, chip["pixel_size_m"]))
            collection = _mask_collection(mask, chip["probability"], chip["valid"], chip["profile"], base, capture_date=None)
            predictions = _predictions(collection)
            result = match_instances(list(predictions.geometry), list(chip["labels"].geometry))
            pooled.extend(
                score_pair(predictions.geometry[prediction], chip["labels"].geometry[label])
                for prediction, label in result.matches
            )
            predicted += len(predictions)
            matched += len(result.matches)
            missed += len(result.unmatched_labels)
            merges += len(result.merged_predictions)

        summary = summarise(pooled)
        f1 = summary.get("median_boundary_f1", {})
        results.append(
            {
                "variant": name,
                "predicted": predicted,
                "matched": matched,
                "missed": missed,
                "merges": merges,
                "recall": round(1 - missed / label_total, 4),
                "outline": summary,
                "boundary_movement": movement,
            }
        )
        print(
            f"{name:<30} {predicted:>5} {matched:>8} {missed:>7} {merges:>7} "
            f"{f1.get('0.25', 0):>9.3f} {f1.get('0.5', 0):>8.3f} {summary.get('median_iou', 0):>6.3f} "
            f"{summary.get('median_hausdorff_95_m', 0):>7.2f} {summary.get('median_predicted_vertices', 0):>6}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"chips": list(args.chips), "label_total": label_total, "threshold": base.threshold, "variants": results},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
