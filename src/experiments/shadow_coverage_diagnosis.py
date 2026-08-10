"""Diagnose partial building detections, especially the shadow-to-roof case.

The pixel change score is a radiometric delta, so a dark shadow replaced by a
bright roof scores far higher than bare ground replaced by the same roof. The
thresholded mask can therefore capture only the part of a building that used to
be in shadow and drop the rest.

This measures completeness per new building by asking what fraction of each new
footprint the pixel-change candidates actually cover, splits that by how dark
the before imagery was, and renders worked examples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BEFORE_IMG = ROOT / r"data\input\EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000\EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
AFTER_IMG = ROOT / r"data\input\EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000\EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"

NEW_FOOTPRINTS = ROOT / r"data\output\dino_fixed\after_dino_building_footprints.geojson"
OLD_FOOTPRINTS = ROOT / r"data\output\dino_fixed\before_dino_building_footprints.geojson"
PIXEL_NEW = ROOT / r"data\output\preset_high_recall_reg\construction_change_candidates.geojson"
FUSED_OLD = ROOT / r"data\output\enhanced_test_v4\enhanced_fused_candidates.geojson"
FUSED_NEW = ROOT / r"data\output\fixed_pipeline\fused_candidates.geojson"

OUT = ROOT / "data" / "output" / "shadow_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

MIN_BUILDING_M2 = 60.0
SHADOW_PERCENTILE = 25.0


def load(path: Path, crs) -> list:
    if not path.exists():
        return []
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    out = []
    for feature in features:
        geometry = shape(transform_geom("EPSG:4326", crs, feature["geometry"]))
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if not geometry.is_empty:
            out.append(geometry)
    return out


def stretch(data: np.ndarray) -> np.ndarray:
    result = np.zeros(data.shape[1:] + (3,), dtype=np.uint8)
    for band in range(3):
        values = data[band][data[band] > 0]
        if values.size:
            low, high = np.percentile(values, (2, 98))
            result[:, :, band] = np.clip((data[band] - low) * 255.0 / max(high - low, 1.0), 0, 255)
    return result


def mean_luma(dataset, geometry) -> float:
    """Mean brightness of the source imagery inside a geometry, 0-255."""
    west, south, east, north = geometry.bounds
    window = dataset.window(west, south, east, north)
    data = dataset.read((1, 2, 3), window=window, boundless=True, fill_value=0).astype(np.float32)
    if data.size == 0:
        return float("nan")
    luma = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
    valid = luma[luma > 0]
    return float(valid.mean()) if valid.size else float("nan")


def draw(ax, geometries, transform, window, colour, width=1.8):
    for geometry in geometries:
        polys = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            rows, cols = rasterio.transform.rowcol(transform, list(xs), list(ys))
            ax.plot(np.asarray(cols) - window.col_off, np.asarray(rows) - window.row_off,
                    color=colour, linewidth=width)


def main() -> None:
    with rasterio.open(AFTER_IMG) as after_ds, rasterio.open(BEFORE_IMG) as before_ds:
        crs = after_ds.crs
        after_fp = load(NEW_FOOTPRINTS, crs)
        before_fp = load(OLD_FOOTPRINTS, crs)
        pixel_new = load(PIXEL_NEW, crs)
        fused_old = load(FUSED_OLD, crs)
        fused_new = load(FUSED_NEW, crs)

        before_union = unary_union(before_fp) if before_fp else None
        pixel_union = unary_union(pixel_new) if pixel_new else None

        # A new building is a 2026 footprint with little overlap from any 2021 footprint.
        rows = []
        for geometry in after_fp:
            if geometry.area < MIN_BUILDING_M2:
                continue
            prior = geometry.intersection(before_union).area / geometry.area if before_union else 0.0
            if prior > 0.25:
                continue
            covered = geometry.intersection(pixel_union).area / geometry.area if pixel_union else 0.0
            rows.append({
                "geometry": geometry,
                "area_m2": geometry.area,
                "coverage": covered,
                "before_luma": mean_luma(before_ds, geometry),
            })

        if not rows:
            print("no new buildings found")
            return

        coverage = np.array([r["coverage"] for r in rows])
        luma = np.array([r["before_luma"] for r in rows])
        dark_cut = np.nanpercentile(luma, SHADOW_PERCENTILE)
        dark = luma <= dark_cut

        print(f"new buildings >= {MIN_BUILDING_M2:.0f} m2: {len(rows)}")
        print(f"\npixel-detector coverage of each new building footprint:")
        for lo, hi in [(0.0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
            count = int(((coverage >= lo) & (coverage < hi)).sum())
            label = "missed entirely" if hi == 0.01 else f"{lo:.0%}-{hi:.0%} covered"
            print(f"  {label:22} {count:4d}  {'#' * int(40 * count / len(rows))}")
        print(f"\n  median coverage {np.median(coverage):.1%}")
        print(f"  fully missed    {int((coverage < 0.01).sum())} of {len(rows)}")
        print(f"  partial (<50%)  {int((coverage < 0.5).sum())} of {len(rows)}")

        print(f"\nsplit by how dark the 2021 imagery was inside the footprint")
        print(f"  (dark = mean luma <= {dark_cut:.0f} of 255, the darkest {SHADOW_PERCENTILE:.0f}%)")
        print(f"  dark  before: n={int(dark.sum()):3d}  median coverage {np.median(coverage[dark]):.1%}")
        print(f"  light before: n={int((~dark).sum()):3d}  median coverage {np.median(coverage[~dark]):.1%}")

        summary = {
            "new_buildings": len(rows),
            "median_coverage": round(float(np.median(coverage)), 4),
            "missed_entirely": int((coverage < 0.01).sum()),
            "partial_under_half": int((coverage < 0.5).sum()),
            "dark_before_median_coverage": round(float(np.median(coverage[dark])), 4),
            "light_before_median_coverage": round(float(np.median(coverage[~dark])), 4),
            "dark_luma_threshold": round(float(dark_cut), 1),
        }
        (OUT / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Worked examples spanning the darkest 2021 sites, where the shadow-to-roof
        # delta is strongest and the partial-detection effect should be worst.
        ranked = sorted(rows, key=lambda r: r["before_luma"])
        examples = ranked[:6]

        for index, record in enumerate(examples, start=1):
            geometry = record["geometry"]
            west, south, east, north = geometry.bounds
            pad = max(18.0, 0.5 * max(east - west, north - south))
            box = (west - pad, south - pad, east + pad, north + pad)
            win_a = after_ds.window(*box)
            win_b = before_ds.window(*box)
            img_a = stretch(after_ds.read((1, 2, 3), window=win_a, boundless=True, fill_value=0))
            img_b = stretch(before_ds.read((1, 2, 3), window=win_b, boundless=True, fill_value=0))

            near = lambda geoms: [g for g in geoms if g.intersects(geometry.buffer(pad))]

            fig, axes = plt.subplots(1, 4, figsize=(21, 5.6))
            axes[0].imshow(img_b)
            axes[0].set_title(f"BEFORE 2021-12-01\nmean brightness {record['before_luma']:.0f}/255", fontsize=10)

            axes[1].imshow(img_a)
            axes[1].set_title("AFTER 2026-04-18", fontsize=10)

            axes[2].imshow(img_a)
            draw(axes[2], near(fused_old), after_ds.transform, win_a, "#ff2d2d")
            axes[2].set_title(f"OLD pipeline\n{len(near(fused_old))} candidates here", fontsize=10)

            axes[3].imshow(img_a)
            draw(axes[3], near(fused_new), after_ds.transform, win_a, "#00e56a")
            draw(axes[3], [geometry], after_ds.transform, win_a, "#ffd400", width=2.4)
            axes[3].set_title(
                f"NEW pipeline\nfootprint {record['area_m2']:.0f} m2, pixel coverage {record['coverage']:.0%}",
                fontsize=10,
            )
            for ax in axes:
                ax.set_xlim(0, img_a.shape[1])
                ax.set_ylim(img_a.shape[0], 0)
                ax.axis("off")

            fig.suptitle(
                "yellow = building footprint (2026)   green = new-pipeline candidates   red = old-pipeline candidates",
                fontsize=11,
            )
            fig.tight_layout()
            path = OUT / f"example_{index:02d}_coverage_{int(record['coverage']*100):03d}pct.png"
            fig.savefig(path, dpi=125, bbox_inches="tight")
            plt.close(fig)
            print("wrote", path.name)

        # Coverage histogram split by before-brightness.
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(0, 1, 11)
        ax.hist([coverage[dark], coverage[~dark]], bins=bins, stacked=True,
                label=[f"dark in 2021 (luma <= {dark_cut:.0f})", "lighter in 2021"],
                color=["#2b3a67", "#8fb8de"], edgecolor="white")
        ax.set_xlabel("fraction of the new building covered by pixel-change candidates")
        ax.set_ylabel("number of new buildings")
        ax.set_title("How completely does the pixel detector cover each new building?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "coverage_histogram.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote coverage_histogram.png")


if __name__ == "__main__":
    main()
