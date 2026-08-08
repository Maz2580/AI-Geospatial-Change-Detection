"""Add before/after shadow-risk fields to an existing candidate GeoJSON."""

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

from building_change.shadow_risk import ShadowRiskError, annotate_shadow_risk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate existing change candidates with shadow-review risk.")
    parser.add_argument("--before", required=True, type=Path, help="Older RGB GeoTIFF.")
    parser.add_argument("--after", required=True, type=Path, help="Newer RGB GeoTIFF; defines the candidate grid.")
    parser.add_argument("--candidates", required=True, type=Path, help="Candidate GeoJSON to annotate.")
    parser.add_argument("--output", required=True, type=Path, help="Annotated GeoJSON output.")
    args = parser.parse_args()
    try:
        result = annotate_shadow_risk(args.before, args.after, args.candidates, args.output)
    except (ShadowRiskError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
