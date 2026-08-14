"""Score predicted building outlines against the UC5 human gold set.

This is the project's first measurement of outline *position*. Earlier figures
described whether a change was found, or how consistent its edge angles were;
neither says whether the polygon sits on the roof.

The gold labels are evaluation-only. Nothing here writes to them, and nothing
here may feed a detector: they are the only unbiased geometry available, and a
model tuned against them stops being measurable by them.
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
    """Avoid an incompatible external OSGeo PROJ_LIB in this process only."""
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

from building_change.detector import _read_new_image  # noqa: E402
from building_change.dino_buildings import (  # noqa: E402
    DinoBuildingsConfig,
    _footprint_collection,
    _load_session,
    cached_building_probability,
    resolve_model_path,
)
from building_change.outline_metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    VICTORIAN_METRIC_CRS,
    match_instances,
    score_pair,
    summarise,
)

# Chips whose labels carry no unresolved parent/part containment. A chip where one
# roof is present twice would score a correct single detection as a hit plus a
# miss, so it cannot be used until the human decision is made.
CLEAN_CHIPS = ("909", "912", "913")

# Reported alongside the strict figure. Buildings cut by the chip border cannot
# be outlined fairly, and shapes the labeller marked doubtful are not truth.
STRICT_EXCLUSIONS = ("edge", "unsure")

AREA_BANDS = ((0, 25), (25, 100), (100, 400), (400, float("inf")))


def _predicted_outlines(
    chip: Path, config: DinoBuildingsConfig, session: Any, cache_dir: Path
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Run the current footprint pipeline on one chip, unchanged."""
    probability, valid, profile = cached_building_probability(chip, cache_dir, session, config)
    # Boundary refinement reads the imagery, so the benchmark must supply it or
    # it would silently measure the un-refined pipeline.
    rgb, _, _ = _read_new_image(chip)
    collection = _footprint_collection(probability, valid, profile, config, capture_date=None, rgb=rgb)
    geometries = [shape(feature["geometry"]) for feature in collection["features"]]
    frame = gpd.GeoDataFrame(
        {"source_area_m2": [feature["properties"]["area_m2"] for feature in collection["features"]]},
        geometry=geometries,
        crs="EPSG:4326",
    )
    return frame.to_crs(VICTORIAN_METRIC_CRS), {
        "raster_crs": str(profile["crs"]),
        "footprint_count": len(geometries),
    }


def _labels(shapefile: Path) -> gpd.GeoDataFrame:
    frame = gpd.read_file(shapefile).to_crs(VICTORIAN_METRIC_CRS).reset_index(drop=True)
    for column in STRICT_EXCLUSIONS:
        if column not in frame:
            frame[column] = 0
        frame[column] = frame[column].fillna(0).astype(int)
    if "kind" not in frame:
        frame["kind"] = ""
    frame["kind"] = frame["kind"].fillna("").astype(str).str.strip()
    return frame


def _band(area: float) -> str:
    for low, high in AREA_BANDS:
        if low <= area < high:
            return f"{low:.0f}-{high:.0f} m2" if high != float("inf") else f"{low:.0f}+ m2"
    return "unknown"


def evaluate_chip(
    chip_id: str, goldset: Path, config: DinoBuildingsConfig, session: Any, cache_dir: Path
) -> dict[str, Any]:
    chip = goldset / f"chip_{chip_id}.tif"
    shapefile = goldset / f"labels_{chip_id}.shp"
    if not chip.is_file() or not shapefile.is_file():
        raise FileNotFoundError(f"Chip {chip_id} needs both {chip.name} and {shapefile.name}.")

    predictions, provenance = _predicted_outlines(chip, config, session, cache_dir)
    labels = _labels(shapefile)
    strict = labels[(labels["edge"] == 0) & (labels["unsure"] == 0)]

    result = match_instances(list(predictions.geometry), list(labels.geometry))
    scores = [
        score_pair(predictions.geometry[prediction], labels.geometry[label])
        for prediction, label in result.matches
    ]

    by_band: dict[str, list] = {}
    by_kind: dict[str, list] = {}
    for (_, label_index), score in zip(result.matches, scores):
        by_band.setdefault(_band(score.label_area_m2), []).append(score)
        by_kind.setdefault(labels["kind"][label_index] or "(blank)", []).append(score)

    detected = len(result.matches) + len(result.split_labels) + len(result.merged_predictions)
    return {
        "chip_id": chip_id,
        "provenance": provenance,
        "label_count": int(len(labels)),
        "label_count_strict": int(len(strict)),
        "predicted_count": int(len(predictions)),
        "detection": {
            "matched_labels": len(result.matches),
            "missed_labels": len(result.unmatched_labels),
            "false_alarms": len(result.unmatched_predictions),
            "split_labels": len(result.split_labels),
            "merged_predictions": len(result.merged_predictions),
            "recall": round(1 - len(result.unmatched_labels) / len(labels), 4) if len(labels) else None,
            "precision": round(
                1 - len(result.unmatched_predictions) / len(predictions), 4
            ) if len(predictions) else None,
        },
        "outline": summarise(scores),
        "outline_by_area_band": {band: summarise(items) for band, items in sorted(by_band.items())},
        "outline_by_kind": {kind: summarise(items) for kind, items in sorted(by_kind.items())},
        "detected_any_way": detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", type=Path, default=Path("data/benchmarks/uc5_goldset_v2"))
    parser.add_argument("--chips", nargs="*", default=list(CLEAN_CHIPS), help=f"Default: {' '.join(CLEAN_CHIPS)}")
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/uc5_goldset_v2/outline_baseline.json"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-cache", default="data/output/model_cache/huggingface")
    parser.add_argument("--threshold", type=float, help="Override the published building probability threshold.")
    parser.add_argument(
        "--min-area-m2",
        type=float,
        help="Defaults to the pipeline's own setting, so the benchmark measures what ships.",
    )
    parser.add_argument("--regularise", action="store_true", help="Apply dominant-orientation snapping (off by default).")
    parser.add_argument("--label", default="dino_v3_current_pipeline", help="Name for this configuration.")
    parser.add_argument(
        "--probability-cache",
        type=Path,
        default=Path("data/output/goldset_probability_cache"),
        help="Reuses model output across runs. Clear it if the model or window changes.",
    )
    args = parser.parse_args()

    base = DinoBuildingsConfig()
    config = replace(
        base,
        threshold=args.threshold if args.threshold is not None else base.threshold,
        min_area_m2=args.min_area_m2 if args.min_area_m2 is not None else base.min_area_m2,
        regularisation=replace(base.regularisation, enabled=args.regularise),
    )
    config.validate()
    session = _load_session(resolve_model_path(args.model_path, cache_dir=args.model_cache))

    chips = [
        evaluate_chip(chip_id, args.goldset, config, session, args.probability_cache)
        for chip_id in args.chips
    ]
    every_score_count = sum(chip["outline"].get("instance_count", 0) for chip in chips)

    report = {
        "configuration": args.label,
        "metric_crs": VICTORIAN_METRIC_CRS,
        "boundary_tolerances_m": list(DEFAULT_TOLERANCES_M),
        "model_config": {
            "threshold": config.threshold,
            "min_area_m2": config.min_area_m2,
            "regularisation_enabled": config.regularisation.enabled,
            "simplify_m": config.simplify_m,
        },
        "chips": chips,
        "totals": {
            "labels": sum(chip["label_count"] for chip in chips),
            "predictions": sum(chip["predicted_count"] for chip in chips),
            "matched": sum(chip["detection"]["matched_labels"] for chip in chips),
            "missed": sum(chip["detection"]["missed_labels"] for chip in chips),
            "false_alarms": sum(chip["detection"]["false_alarms"] for chip in chips),
            "splits": sum(chip["detection"]["split_labels"] for chip in chips),
            "merges": sum(chip["detection"]["merged_predictions"] for chip in chips),
            "scored_outlines": every_score_count,
        },
        "warning": (
            "Gold labels are evaluation-only. They must never be used to train, tune, or "
            "seed a detector, and outline scores here describe only the chips listed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["totals"], indent=2))
    for chip in chips:
        print(
            f"chip {chip['chip_id']}: {chip['detection']['matched_labels']}/{chip['label_count']} matched, "
            f"bF1@0.5 {chip['outline'].get('median_boundary_f1', {}).get('0.5')}, "
            f"IoU {chip['outline'].get('median_iou')}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
