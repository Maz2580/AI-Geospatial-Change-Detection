"""Build construction-site review candidates from change and footprint evidence."""

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

from building_change.sites import SiteGroupingError, load_feature_collection, write_site_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Group change and footprint evidence into construction-site review candidates.")
    parser.add_argument("--changes", required=True, type=Path, help="Fused WGS84 change-candidate FeatureCollection.")
    parser.add_argument("--footprints", required=True, type=Path, help="WGS84 footprint-comparison FeatureCollection.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for site GeoJSON and report.")
    parser.add_argument("--anchor-merge-distance-m", type=float, default=10, help="Merge nearby change anchors within this distance (default: 10).")
    parser.add_argument("--site-radius-m", type=float, default=25, help="Attach nearby footprint evidence within this distance (default: 25).")
    args = parser.parse_args()
    try:
        report = write_site_outputs(
            args.output,
            load_feature_collection(args.changes),
            load_feature_collection(args.footprints),
            anchor_merge_distance_m=args.anchor_merge_distance_m,
            site_radius_m=args.site_radius_m,
        )
    except (SiteGroupingError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
