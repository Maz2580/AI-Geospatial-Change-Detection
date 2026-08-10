"""Sweep the before-date footprint threshold to cut phantom new buildings.

A building the 2021 pass misses becomes a false "new building" in 2026, because
the comparison sees no prior footprint to match against. Recall on the before
date therefore matters more than precision: a spurious 2021 footprint only
suppresses one candidate, while a missed one invents one.

Re-thresholds the saved probability raster rather than re-running inference, so
the whole sweep costs seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from building_change.footprints import compare_footprints
from building_change.regularisation import RegularisationConfig, regularise_geometries

DINO_DIR = ROOT / "data" / "output" / "dino_fixed"
BEFORE_PROB = DINO_DIR / "before_dino_building_probability.tif"
AFTER_FOOTPRINTS = DINO_DIR / "after_dino_building_footprints.geojson"
OUT = ROOT / "data" / "output" / "before_threshold_sweep"

DEFAULT_THRESHOLD = 0.4371
SWEEP = [0.4371, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
MIN_AREA_M2 = 10.0


def vectorise(probability: np.ndarray, valid: np.ndarray, profile: dict[str, Any], threshold: float) -> dict[str, Any]:
    crs = profile["crs"]
    mask = (probability >= threshold) & valid
    raw = []
    for geometry_json, value in shapes(mask.astype(np.uint8), mask=mask, transform=profile["transform"]):
        if value != 1:
            continue
        geometry = shape(geometry_json)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty:
            continue
        geometry = geometry.simplify(0.25, preserve_topology=True)
        if geometry.is_empty or geometry.area < MIN_AREA_M2:
            continue
        raw.append(geometry)

    geometries = regularise_geometries(raw, crs, RegularisationConfig(simplify_tolerance_m=0.2))
    features = []
    for geometry in geometries:
        features.append(
            {
                "type": "Feature",
                "geometry": transform_geom(crs, "EPSG:4326", mapping(geometry), precision=8),
                "properties": {"candidate_id": len(features) + 1, "area_m2": round(float(geometry.area), 1)},
            }
        )
    return {"type": "FeatureCollection", "features": features, "metadata": {"probability_threshold": threshold}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    after = json.loads(AFTER_FOOTPRINTS.read_text(encoding="utf-8"))
    after_area = sum(f["properties"]["area_m2"] for f in after["features"])

    with rasterio.open(BEFORE_PROB) as src:
        probability = src.read(1)
        profile = {"crs": src.crs, "transform": src.transform}
        scene_m2 = src.width * src.res[0] * src.height * src.res[1]
    valid = probability >= 0.0

    print(f"scene {scene_m2:,.0f} m2   2026 footprints: {len(after['features'])}, {after_area:,.0f} m2 "
          f"({100*after_area/scene_m2:.1f}% coverage)\n")
    header = f"{'thresh':>7} {'2021 fp':>8} {'2021 m2':>10} {'cover':>7} {'median':>7} | {'new':>4} {'ext':>4} {'new m2':>9} {'new>=100':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for threshold in SWEEP:
        before = vectorise(probability, valid, profile, threshold)
        areas = np.array([f["properties"]["area_m2"] for f in before["features"]]) if before["features"] else np.zeros(0)
        comparison = compare_footprints(before, after, min_area_m2=MIN_AREA_M2)
        classes: dict[str, int] = {}
        new_areas = []
        for feature in comparison["features"]:
            name = feature["properties"]["classification"]
            classes[name] = classes.get(name, 0) + 1
            if name == "new_building_footprint_candidate":
                new_areas.append(feature["properties"]["area_m2"])
        new_areas = np.array(new_areas) if new_areas else np.zeros(0)

        row = {
            "threshold": threshold,
            "before_footprints": len(before["features"]),
            "before_area_m2": float(areas.sum()),
            "before_coverage_pct": round(100 * float(areas.sum()) / scene_m2, 2),
            "before_median_m2": float(np.median(areas)) if areas.size else 0.0,
            "new_building_count": classes.get("new_building_footprint_candidate", 0),
            "extension_count": classes.get("building_extension_footprint_candidate", 0),
            "new_building_area_m2": float(new_areas.sum()),
            "new_over_100_m2": int((new_areas >= 100).sum()),
        }
        rows.append(row)
        print(f"{threshold:>7.4f} {row['before_footprints']:>8d} {row['before_area_m2']:>10,.0f} "
              f"{row['before_coverage_pct']:>6.1f}% {row['before_median_m2']:>7.0f} | "
              f"{row['new_building_count']:>4d} {row['extension_count']:>4d} "
              f"{row['new_building_area_m2']:>9,.0f} {row['new_over_100_m2']:>9d}")

        if threshold in (DEFAULT_THRESHOLD, 0.25):
            tag = "default" if threshold == DEFAULT_THRESHOLD else "relaxed"
            (OUT / f"before_footprints_{tag}.geojson").write_text(json.dumps(before, indent=2), encoding="utf-8")

    print("\nMarginal effect of each threshold step:")
    for previous, current in zip(rows, rows[1:]):
        added_fp = current["before_footprints"] - previous["before_footprints"]
        added_area = current["before_area_m2"] - previous["before_area_m2"]
        new_delta = current["new_building_count"] - previous["new_building_count"]
        per_fp = added_area / added_fp if added_fp else 0.0
        print(f"  {previous['threshold']:.4f} -> {current['threshold']:.4f}: "
              f"2021 footprints {added_fp:+4d} ({added_area:+8,.0f} m2, {per_fp:5.0f} m2 each), "
              f"new-building candidates {new_delta:+3d}")

    after_coverage = 100 * after_area / scene_m2
    implausible = [f"{r['threshold']:.2f}" for r in rows if r["before_coverage_pct"] >= after_coverage]
    if implausible:
        print(f"\n2021 coverage reaches or exceeds the 2026 figure ({after_coverage:.1f}%) at threshold(s) "
              f"{', '.join(implausible)}. The estate cannot have held more building in 2021 than in 2026, "
              "so those settings are over-detecting on the before date.")

    (OUT / "sweep_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'sweep_report.json'}")


if __name__ == "__main__":
    main()
