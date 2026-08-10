"""Resolution ladder: does the building model work better on downsampled imagery?

The DINO footprint model is trained on ~0.3-0.5 m/px imagery but is being run at
0.075 m/px. This resamples the pair to coarser grids and re-runs footprint
extraction at each scale to find where the model actually performs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

ROOT = Path(r"c:\Users\maz.ghasemi\Downloads\Maz - 2 July 2025\python\change detection")
sys.path.insert(0, str(ROOT / "src"))

BEFORE = ROOT / r"data\input\EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000\EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
AFTER = ROOT / r"data\input\EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000\EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
WORK = ROOT / "data" / "output" / "resolution_ladder"
WORK.mkdir(parents=True, exist_ok=True)


def resample(source: Path, target_res: float, destination: Path) -> Path:
    if destination.exists():
        return destination
    with rasterio.open(source) as src:
        factor = src.res[0] / target_res
        height = int(src.height * factor)
        width = int(src.width * factor)
        data = src.read(
            out_shape=(src.count, height, width),
            resampling=Resampling.average,
        )
        transform = src.transform * src.transform.scale(src.width / width, src.height / height)
        profile = src.profile.copy()
        profile.update(height=height, width=width, transform=transform, compress="deflate")
        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(data)
    return destination


def scene_area_m2(path: Path) -> float:
    with rasterio.open(path) as src:
        return src.width * src.res[0] * src.height * src.res[1]


if __name__ == "__main__":
    from building_change.dino_buildings import DinoBuildingsConfig, run_dino_building_comparison
    from building_change.regularisation import RegularisationConfig, tolerance_for_pixel_size

    total_scene = scene_area_m2(AFTER)
    print(f"scene area {total_scene:,.0f} m2 ({total_scene/10000:.1f} ha)\n")

    rows = []
    for target_res in [0.075, 0.15, 0.30, 0.50]:
        tag = f"{int(target_res*1000):04d}mm"
        if target_res == 0.075:
            before_path, after_path = BEFORE, AFTER
        else:
            before_path = resample(BEFORE, target_res, WORK / f"before_{tag}.tif")
            after_path = resample(AFTER, target_res, WORK / f"after_{tag}.tif")

        out_dir = WORK / tag
        config = DinoBuildingsConfig(
            min_area_m2=10.0,
            regularisation=RegularisationConfig(
                simplify_tolerance_m=tolerance_for_pixel_size(target_res),
            ),
        )
        print(f"--- {target_res} m/px ---")
        try:
            report = run_dino_building_comparison(
                before_path, after_path, out_dir,
                config=config,
                before_capture_date="2021-12-01",
                after_capture_date="2026-04-18",
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        after_features = json.loads((out_dir / "after_dino_building_footprints.geojson").read_text(encoding="utf-8"))["features"]
        areas = np.array([f["properties"]["area_m2"] for f in after_features]) if after_features else np.zeros(0)
        house_sized = int((areas >= 60).sum())
        rows.append({
            "res_m": target_res,
            "before_footprints": report["before_footprint_count"],
            "after_footprints": report["after_footprint_count"],
            "after_total_m2": float(areas.sum()),
            "coverage_pct": 100.0 * float(areas.sum()) / total_scene,
            "median_m2": float(np.median(areas)) if areas.size else 0.0,
            "house_sized": house_sized,
            "change_candidates": report["comparison"]["candidate_count"],
        })
        print(f"  before={rows[-1]['before_footprints']} after={rows[-1]['after_footprints']} "
              f"total={rows[-1]['after_total_m2']:,.0f} m2 coverage={rows[-1]['coverage_pct']:.2f}% "
              f"median={rows[-1]['median_m2']:.0f} m2 house_sized={house_sized}")

    print("\n" + "=" * 104)
    print(f"{'res m/px':>9} {'before':>7} {'after':>7} {'built m2':>11} {'coverage':>9} {'median m2':>10} {'>=60m2':>7} {'chg cands':>10}")
    print("=" * 104)
    for row in rows:
        print(f"{row['res_m']:>9.3f} {row['before_footprints']:>7d} {row['after_footprints']:>7d} "
              f"{row['after_total_m2']:>11,.0f} {row['coverage_pct']:>8.2f}% {row['median_m2']:>10.0f} "
              f"{row['house_sized']:>7d} {row['change_candidates']:>10d}")
    (WORK / "ladder_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
