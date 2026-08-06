"""Command-line interface for the repeatable building-change workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .detector import DetectionConfig, run_detection
from .nearmap import NearmapApiError, NearmapClient, select_surveys


def _add_detection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--change-percentile", type=float, default=98.5, help="Adaptive score percentile (default: 98.5).")
    parser.add_argument("--change-threshold", type=float, help="Fixed 0-1 score threshold; overrides --change-percentile.")
    parser.add_argument("--min-area-m2", type=float, default=20, help="Discard smaller candidate polygons (default: 20).")
    parser.add_argument("--morphology-m", type=float, default=0.6, help="Noise-removal size in metres (default: 0.6).")
    parser.add_argument("--keep-shadow-changes", action="store_true", help="Do not defer potential shadow changes to uncertain review output.")
    parser.add_argument("--no-register", action="store_true", help="Disable sub-pixel translation registration.")


def _config(args: argparse.Namespace) -> DetectionConfig:
    return DetectionConfig(
        change_percentile=args.change_percentile,
        change_threshold=args.change_threshold,
        min_area_m2=args.min_area_m2,
        morphology_m=args.morphology_m,
        enable_registration=not args.no_register,
        enable_shadow_filter=not args.keep_shadow_changes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect construction change from a local GeoTIFF pair or fixed-date Nearmap surveys."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("local", help="Compare two already-downloaded RGB GeoTIFFs.")
    local.add_argument("--before", required=True, type=Path, help="Older RGB GeoTIFF.")
    local.add_argument("--after", required=True, type=Path, help="Newer RGB GeoTIFF; sets the output grid.")
    local.add_argument("--output", required=True, type=Path, help="Directory for rasters, GeoJSON, and report.")
    local.add_argument("--before-dsm", type=Path, help="Older DSM GeoTIFF (optional; provide both DSMs).")
    local.add_argument("--after-dsm", type=Path, help="Newer DSM GeoTIFF (optional; provide both DSMs).")
    local.add_argument("--before-capture-date", help="ISO capture date retained in GeoJSON properties.")
    local.add_argument("--after-capture-date", help="ISO capture date retained in GeoJSON properties.")
    _add_detection_options(local)

    nearmap = commands.add_parser("nearmap", help="Download two fixed-date Nearmap surveys, then compare them.")
    nearmap.add_argument("--longitude", required=True, type=float)
    nearmap.add_argument("--latitude", required=True, type=float)
    nearmap.add_argument("--radius-m", required=True, type=int, help="Ground radius in metres. Keep at or below 100 m.")
    nearmap.add_argument("--before-date", required=True, help="Uses the latest survey captured on/before this ISO date.")
    nearmap.add_argument("--after-date", help="Uses the earliest survey on/after this ISO date; default is latest.")
    nearmap.add_argument("--output", required=True, type=Path)
    nearmap.add_argument("--source", choices=("tiles", "staticmap"), default="tiles", help="Use standard Tile API (default) or Staticmap subscription.")
    nearmap.add_argument("--tile-zoom", type=int, help="Optional Tile API zoom; default uses the survey's native maximum.")
    nearmap.add_argument("--content-type", choices=("Vert", "TrueOrtho"), default="Vert")
    nearmap.add_argument("--size-px", type=int, default=5000, help="Staticmap image width/height, maximum 5000.")
    nearmap.add_argument("--with-dsm", action="store_true", help="Staticmap only: add DSM height evidence.")
    _add_detection_options(nearmap)
    return parser


def _run_local(args: argparse.Namespace) -> dict:
    return run_detection(
        args.before, args.after, args.output,
        before_dsm=args.before_dsm, after_dsm=args.after_dsm, config=_config(args),
        before_capture_date=args.before_capture_date, after_capture_date=args.after_capture_date,
    )


def _run_tile_nearmap(client: NearmapClient, args: argparse.Namespace) -> dict:
    if args.content_type != "Vert":
        raise ValueError("The standard Tile API supports Vert imagery here. Use --source staticmap for TrueOrtho.")
    if args.with_dsm:
        raise ValueError("--with-dsm requires --source staticmap and the relevant Nearmap subscription.")
    surveys = client.tile_coverage(longitude=args.longitude, latitude=args.latitude, content_type="Vert")
    before_survey, after_survey = select_surveys(surveys, before_date=args.before_date, after_date=args.after_date)
    imagery_dir = args.output / "imagery"
    before_path = client.download_tile_mosaic(
        before_survey, longitude=args.longitude, latitude=args.latitude, radius_m=args.radius_m, zoom=args.tile_zoom,
        output_path=imagery_dir / f"before_{before_survey.capture_date}_Vert_tiles.tif",
    )
    after_path = client.download_tile_mosaic(
        after_survey, longitude=args.longitude, latitude=args.latitude, radius_m=args.radius_m, zoom=args.tile_zoom,
        output_path=imagery_dir / f"after_{after_survey.capture_date}_Vert_tiles.tif",
    )
    return run_detection(
        before_path, after_path, args.output, config=_config(args),
        before_capture_date=before_survey.capture_date.isoformat(), after_capture_date=after_survey.capture_date.isoformat(),
    )


def _run_staticmap_nearmap(client: NearmapClient, args: argparse.Namespace) -> dict:
    resources = [args.content_type] + (["DetailDsm"] if args.with_dsm else [])
    surveys, token = client.coverage(longitude=args.longitude, latitude=args.latitude, radius_m=args.radius_m, resources=resources)
    before_survey, after_survey = select_surveys(surveys, before_date=args.before_date, after_date=args.after_date)
    imagery_dir = args.output / "imagery"
    common = {"transaction_token": token, "longitude": args.longitude, "latitude": args.latitude, "radius_m": args.radius_m, "size_px": args.size_px}
    before_path = client.download(before_survey, content_type=args.content_type, **common, output_path=imagery_dir / f"before_{before_survey.capture_date}_{args.content_type}.tif")
    after_path = client.download(after_survey, content_type=args.content_type, **common, output_path=imagery_dir / f"after_{after_survey.capture_date}_{args.content_type}.tif")
    before_dsm = after_dsm = None
    if args.with_dsm:
        before_dsm = client.download(before_survey, content_type="DetailDsm", **common, output_path=imagery_dir / f"before_{before_survey.capture_date}_DetailDsm.tif")
        after_dsm = client.download(after_survey, content_type="DetailDsm", **common, output_path=imagery_dir / f"after_{after_survey.capture_date}_DetailDsm.tif")
    return run_detection(
        before_path, after_path, args.output, before_dsm=before_dsm, after_dsm=after_dsm, config=_config(args),
        before_capture_date=before_survey.capture_date.isoformat(), after_capture_date=after_survey.capture_date.isoformat(),
    )


def _run_nearmap(args: argparse.Namespace) -> dict:
    load_dotenv(Path.cwd() / ".env")
    client = NearmapClient(os.getenv("NEARMAP_API_KEY", ""))
    if args.source == "tiles":
        return _run_tile_nearmap(client, args)
    return _run_staticmap_nearmap(client, args)


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = _run_local(args) if args.command == "local" else _run_nearmap(args)
    except (NearmapApiError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
