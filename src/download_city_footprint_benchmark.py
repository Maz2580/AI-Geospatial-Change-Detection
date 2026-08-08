"""Download dated City of Melbourne building footprints for an auditable benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


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

from building_change.city_footprints import (  # noqa: E402  (must follow PROJ setup)
    CityFootprintError,
    FootprintBoundingBox,
    write_city_footprint_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download dated official City of Melbourne building footprints and produce review-only reference changes."
    )
    parser.add_argument("--west", required=True, type=float, help="AOI western longitude (WGS84).")
    parser.add_argument("--south", required=True, type=float, help="AOI southern latitude (WGS84).")
    parser.add_argument("--east", required=True, type=float, help="AOI eastern longitude (WGS84).")
    parser.add_argument("--north", required=True, type=float, help="AOI northern latitude (WGS84).")
    parser.add_argument("--before-year", required=True, type=int, help="Earlier City footprint snapshot year.")
    parser.add_argument("--after-year", required=True, type=int, help="Later City footprint snapshot year.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for versionable reference GeoJSON and review candidates.")
    parser.add_argument("--match-distance-m", type=float, default=6.0, help="Footprint offset tolerance in metres (default: 6).")
    parser.add_argument("--match-iou", type=float, default=0.10, help="Footprint overlap IoU indicating the same structure (default: 0.10).")
    parser.add_argument("--extension-outside-fraction", type=float, default=0.25, help="Outside-area fraction required to flag an extension (default: 0.25).")
    parser.add_argument("--min-area-m2", type=float, default=20.0, help="Discard smaller reference polygons (default: 20).")
    args = parser.parse_args()
    try:
        report = write_city_footprint_benchmark(
            args.output,
            before_year=args.before_year,
            after_year=args.after_year,
            bbox=FootprintBoundingBox(args.west, args.south, args.east, args.north),
            match_distance_m=args.match_distance_m,
            match_iou=args.match_iou,
            extension_outside_fraction=args.extension_outside_fraction,
            min_area_m2=args.min_area_m2,
        )
    except (CityFootprintError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
