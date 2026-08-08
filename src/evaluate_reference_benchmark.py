"""Evaluate one candidate GeoJSON against human-confirmed reference changes."""

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

from building_change.reference_evaluation import ReferenceEvaluationError, load_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a detector candidate set against human-confirmed reference changes.")
    parser.add_argument("--reference", required=True, type=Path, help="Reference footprint-change GeoJSON.")
    parser.add_argument("--labels", required=True, type=Path, help="Human labels for the reference candidates.")
    parser.add_argument("--predictions", required=True, type=Path, help="Detector candidate GeoJSON to score.")
    parser.add_argument("--match-distance-m", type=float, default=8.0, help="Reference-to-candidate matching tolerance (default: 8).")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    try:
        report = load_and_evaluate(args.reference, args.labels, args.predictions, match_distance_m=args.match_distance_m)
    except (ReferenceEvaluationError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
