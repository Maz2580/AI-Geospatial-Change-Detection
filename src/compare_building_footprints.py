"""Compare building-footprint proposals extracted at two aerial dates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _prefer_rasterio_projection_data() -> None:
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    bundled_data = Path(spec.origin).parent / "proj_data"
    if bundled_data.is_dir():
        os.environ["PROJ_DATA"] = str(bundled_data)
        os.environ.pop("PROJ_LIB", None)


_prefer_rasterio_projection_data()

from building_change.footprints import FootprintComparisonError, load_feature_collection, write_comparison_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WGS84 building-footprint proposals from two dates.")
    parser.add_argument("--before", required=True, type=Path, help="Before-date footprint FeatureCollection.")
    parser.add_argument("--after", required=True, type=Path, help="After-date footprint FeatureCollection.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for footprint change candidates and report.")
    parser.add_argument("--match-distance-m", type=float, default=6, help="Absorb small footprint offsets up to this distance (default: 6).")
    parser.add_argument("--match-iou", type=float, default=0.10, help="Minimum overlap IoU indicating the same footprint (default: 0.10).")
    parser.add_argument("--extension-outside-fraction", type=float, default=0.25, help="Outside-area fraction needed to flag an extension (default: 0.25).")
    parser.add_argument("--min-area-m2", type=float, default=20, help="Discard smaller footprint candidates (default: 20).")
    args = parser.parse_args()
    try:
        report = write_comparison_outputs(
            args.output,
            load_feature_collection(args.before),
            load_feature_collection(args.after),
            match_distance_m=args.match_distance_m,
            match_iou=args.match_iou,
            extension_outside_fraction=args.extension_outside_fraction,
            min_area_m2=args.min_area_m2,
        )
    except (FootprintComparisonError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
