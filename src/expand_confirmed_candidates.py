"""Expand selected candidate polygons into nearby missed change sections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from building_change.detector import DetectionConfig, _vectorise, _write_geojson, _write_raster
from building_change.expansion import expand_existing_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Grow only selected candidate polygons into nearby missing sections.")
    parser.add_argument("--score", required=True, type=Path, help="Existing change_score.tif.")
    parser.add_argument("--seed-candidates", required=True, type=Path, help="Candidate GeoJSON; these are the only allowed seeds.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grow-percentile", type=float, default=96.5)
    parser.add_argument("--max-distance-m", type=float, default=6.0)
    parser.add_argument("--closing-m", type=float, default=1.0)
    parser.add_argument("--min-area-m2", type=float, default=20.0)
    args = parser.parse_args()

    with rasterio.open(args.score) as source:
        score = source.read(1).astype(np.float32)
        profile = source.profile.copy()
        nodata = source.nodata
    source_features = json.loads(args.seed_candidates.read_text(encoding="utf-8")).get("features", [])
    if not source_features:
        raise ValueError("seed-candidates does not contain any features.")
    geometries = [transform_geom("EPSG:4326", profile["crs"], feature["geometry"]) for feature in source_features]
    seeds = rasterize([(geometry, 1) for geometry in geometries], out_shape=score.shape, transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
    valid = np.isfinite(score) & ((score != nodata) if nodata is not None else True)
    expanded, grow_threshold = expand_existing_candidates(
        seeds, score, valid, profile,
        grow_percentile=args.grow_percentile, max_distance_m=args.max_distance_m, closing_m=args.closing_m,
    )
    config = DetectionConfig(min_area_m2=args.min_area_m2, morphology_m=args.closing_m)
    candidates, likely_buildings = _vectorise(
        expanded, score, profile, config, height_delta=None, valid_height=None,
        metadata={"grow_percentile": str(args.grow_percentile), "max_distance_m": str(args.max_distance_m), "seed_source": "selected_candidates"},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_raster(args.output / "change_mask_expanded.tif", expanded * 255, profile, dtype="uint8", nodata=0)
    _write_geojson(args.output / "construction_change_candidates.geojson", candidates)
    _write_geojson(args.output / "likely_new_buildings.geojson", likely_buildings)
    report = {"candidate_count": len(candidates), "grow_threshold": grow_threshold, "grow_percentile": args.grow_percentile, "max_distance_m": args.max_distance_m, "closing_m": args.closing_m, "seed_candidate_count": len(source_features)}
    (args.output / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
