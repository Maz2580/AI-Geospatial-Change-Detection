"""Fuse independent construction-change candidate GeoJSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from building_change.fusion import FusionError, load_candidate_input, write_fusion_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse WGS84 candidate GeoJSON from independent sources; retain single-source candidates for review."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="SOURCE=PATH",
        help="Candidate FeatureCollection; repeat for each independent source.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Directory for fused GeoJSON and report.")
    parser.add_argument("--merge-distance-m", type=float, default=2, help="Merge distance in metres (default: 2).")
    args = parser.parse_args()
    try:
        inputs: dict[str, dict] = {}
        for value in args.candidate:
            source, collection = load_candidate_input(value)
            if source in inputs:
                raise FusionError(f"Candidate source {source!r} was supplied more than once.")
            inputs[source] = collection
        report = write_fusion_outputs(args.output, inputs, merge_distance_m=args.merge_distance_m)
    except (FusionError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
