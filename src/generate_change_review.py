"""Create an annotated HTML review report for an existing change-detection run."""

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

from building_change.detector import _normalise_rgb, _read_new_image, _register_before, _reproject_rgb  # noqa: E402
from building_change.review import write_candidate_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create before/after candidate chips for manual review.")
    parser.add_argument("--before", required=True, type=Path, help="Older RGB GeoTIFF.")
    parser.add_argument("--after", required=True, type=Path, help="Newer RGB GeoTIFF; defines the output grid.")
    parser.add_argument("--candidates", required=True, type=Path, help="Construction-candidate GeoJSON.")
    parser.add_argument("--output", required=True, type=Path, help="Existing run output directory.")
    parser.add_argument("--candidate-source", default="the change detector", help="Human-readable source shown in the review page.")
    parser.add_argument("--no-register", action="store_true", help="Skip the same conservative registration used by the detector.")
    args = parser.parse_args()

    features = json.loads(args.candidates.read_text(encoding="utf-8")).get("features", [])
    after_rgb, after_valid, profile = _read_new_image(args.after)
    before_rgb, before_valid = _reproject_rgb(args.before, profile)
    before_normal = _normalise_rgb(before_rgb, before_valid)
    after_normal = _normalise_rgb(after_rgb, after_valid)
    before_aligned, _, _ = _register_before(
        before_normal,
        after_normal,
        before_valid & after_valid,
        enabled=not args.no_register,
        max_shift_px=10.0,
    )
    index = write_candidate_review(
        before_aligned, after_normal, profile, features, args.output, candidate_source=args.candidate_source
    )
    print(index)


if __name__ == "__main__":
    main()
