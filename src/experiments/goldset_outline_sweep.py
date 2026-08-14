"""How much of the outline error is the threshold, and how much is the regulariser?

The baseline scores boundary F1 at 0.25 m around 0.27-0.41 against human roofs.
Two candidate causes sit in the same pipeline: the probability threshold decides
where the boundary is placed, and dominant-orientation regularisation then moves
it again. Separating them decides whether the fix is to replace the boundary
source or simply to stop polishing it.

Inference is the expensive part and does not depend on either setting, so the
probability raster is computed once per chip and cached. Every variant after
that costs seconds.
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

from building_change.dino_buildings import (  # noqa: E402
    DinoBuildingsConfig,
    _footprint_collection,
    _load_session,
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
THRESHOLDS = (0.25, 0.35, 0.4371, 0.55, 0.70)


def _score(
    probability: np.ndarray,
    valid: np.ndarray,
    profile: dict[str, Any],
    config: DinoBuildingsConfig,
    labels: gpd.GeoDataFrame,
) -> dict[str, Any]:
    collection = _footprint_collection(probability, valid, profile, config, capture_date=None)
    if not collection["features"]:
        return {"predicted": 0, "matched": 0, "missed": len(labels), "merges": 0, "scores": []}
    predictions = gpd.GeoDataFrame(
        geometry=[shape(feature["geometry"]) for feature in collection["features"]],
        crs="EPSG:4326",
    ).to_crs(VICTORIAN_METRIC_CRS)
    result = match_instances(list(predictions.geometry), list(labels.geometry))
    return {
        "predicted": len(predictions),
        "matched": len(result.matches),
        "missed": len(result.unmatched_labels),
        "false_alarms": len(result.unmatched_predictions),
        "merges": len(result.merged_predictions),
        "scores": [
            score_pair(predictions.geometry[prediction], labels.geometry[label])
            for prediction, label in result.matches
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", type=Path, default=Path("data/benchmarks/uc5_goldset_v2"))
    parser.add_argument("--chips", nargs="*", default=list(CLEAN_CHIPS))
    parser.add_argument("--cache", type=Path, default=Path("data/output/goldset_probability_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/uc5_goldset_v2/outline_sweep.json"))
    parser.add_argument("--model-cache", default="data/output/model_cache/huggingface")
    args = parser.parse_args()

    base = DinoBuildingsConfig()
    session = _load_session(resolve_model_path(None, cache_dir=args.model_cache))

    rasters: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    labels: dict[str, gpd.GeoDataFrame] = {}
    for chip_id in args.chips:
        rasters[chip_id] = cached_building_probability(
            args.goldset / f"chip_{chip_id}.tif", args.cache, session, base
        )
        labels[chip_id] = (
            gpd.read_file(args.goldset / f"labels_{chip_id}.shp").to_crs(VICTORIAN_METRIC_CRS).reset_index(drop=True)
        )
        print(f"chip {chip_id}: probability ready, {len(labels[chip_id])} labels")

    label_total = sum(len(frame) for frame in labels.values())
    print()
    header = (
        f"{'threshold':>9} {'regular':>8} {'pred':>5} {'matched':>8} {'missed':>7} "
        f"{'merges':>7} {'bF1@0.25':>9} {'bF1@0.5':>8} {'bF1@1.0':>8} {'IoU':>6} {'H95 m':>7} {'verts':>6}"
    )
    print(header)
    print("-" * len(header))

    results: list[dict[str, Any]] = []
    for regularise in (True, False):
        for threshold in THRESHOLDS:
            config = replace(
                base,
                threshold=threshold,
                regularisation=replace(base.regularisation, enabled=regularise),
            )
            config.validate()
            pooled: list[Any] = []
            predicted = matched = missed = merges = 0
            for chip_id in args.chips:
                probability, valid, profile = rasters[chip_id]
                outcome = _score(probability, valid, profile, config, labels[chip_id])
                pooled.extend(outcome["scores"])
                predicted += outcome["predicted"]
                matched += outcome["matched"]
                missed += outcome["missed"]
                merges += outcome["merges"]
            summary = summarise(pooled)
            row = {
                "threshold": threshold,
                "regularised": regularise,
                "predicted": predicted,
                "matched": matched,
                "missed": missed,
                "merges": merges,
                "recall": round(1 - missed / label_total, 4),
                "outline": summary,
            }
            results.append(row)
            f1 = summary.get("median_boundary_f1", {})
            print(
                f"{threshold:>9.4f} {'yes' if regularise else 'no':>8} {predicted:>5} {matched:>8} "
                f"{missed:>7} {merges:>7} {f1.get('0.25', 0):>9.3f} {f1.get('0.5', 0):>8.3f} "
                f"{f1.get('1.0', 0):>8.3f} {summary.get('median_iou', 0):>6.3f} "
                f"{summary.get('median_hausdorff_95_m', 0):>7.2f} {summary.get('median_predicted_vertices', 0):>6}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "chips": list(args.chips),
                "label_total": label_total,
                "metric_crs": VICTORIAN_METRIC_CRS,
                "median_label_vertices": results[0]["outline"].get("median_label_vertices"),
                "variants": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
