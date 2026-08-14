"""Run optional DINOv3 building-footprint change evidence on two GeoTIFF dates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from dotenv import load_dotenv


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

from building_change.dino_buildings import (  # noqa: E402  (must follow PROJ setup)
    DINO_BUILDINGS_THRESHOLD,
    DinoBuildingsConfig,
    DinoBuildingsError,
    run_dino_building_comparison,
)
from building_change.regularisation import RegularisationConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dedicated DINOv3 building footprints at two aerial dates. Results require review."
    )
    parser.add_argument("--before", required=True, type=Path, help="Older RGB GeoTIFF.")
    parser.add_argument("--after", required=True, type=Path, help="Newer RGB GeoTIFF; defines the output grid.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for probability rasters and candidate GeoJSON.")
    parser.add_argument("--before-capture-date", help="ISO date recorded with the before footprints.")
    parser.add_argument("--after-capture-date", help="ISO date recorded with the after footprints.")
    parser.add_argument("--model-path", type=Path, help="Existing local model.onnx; skips Hugging Face download.")
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("data/output/model_cache/huggingface"),
        help="Project-local cache used only when --model-path is omitted.",
    )
    parser.add_argument("--threshold", type=float, default=DINO_BUILDINGS_THRESHOLD, help="Building probability threshold (default: publisher value 0.4371).")
    parser.add_argument("--stride-px", type=int, default=192, help="Sliding-window stride in pixels (default: publisher value 192).")
    parser.add_argument("--min-area-m2", type=float, default=10.0, help="Discard smaller footprint proposals (default: 10).")
    parser.add_argument("--simplify-m", type=float, default=0.25, help="Topology-preserving footprint simplification distance (default: 0.25).")
    parser.add_argument("--match-distance-m", type=float, default=6.0, help="Absorb date-to-date footprint offsets up to this distance.")
    parser.add_argument("--match-iou", type=float, default=0.10, help="Minimum overlap IoU indicating the same footprint.")
    parser.add_argument("--extension-outside-fraction", type=float, default=0.25, help="Outside-area fraction required to flag an extension.")
    parser.add_argument(
        "--regularise",
        action="store_true",
        help="Snap outlines to their dominant orientation. Off by default: measured against "
        "human-drawn roofs it lowers boundary accuracy. See docs/outline_accuracy.md.",
    )
    parser.add_argument("--regularise-tolerance-m", type=float, default=0.2, help="Regularisation simplify tolerance; use 2-3x the imagery pixel size (default: 0.2).")
    parser.add_argument("--no-neighbour-alignment", action="store_true", help="Do not align footprints onto a shared estate grid.")
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    config = DinoBuildingsConfig(
        threshold=args.threshold,
        stride_px=args.stride_px,
        min_area_m2=args.min_area_m2,
        simplify_m=args.simplify_m,
        regularisation=RegularisationConfig(
            enabled=args.regularise,
            simplify_tolerance_m=args.regularise_tolerance_m,
            align_neighbours=not args.no_neighbour_alignment,
        ),
    )
    try:
        report = run_dino_building_comparison(
            args.before,
            args.after,
            args.output,
            model_path=args.model_path,
            model_cache=args.model_cache,
            config=config,
            before_capture_date=args.before_capture_date,
            after_capture_date=args.after_capture_date,
            match_distance_m=args.match_distance_m,
            match_iou=args.match_iou,
            extension_outside_fraction=args.extension_outside_fraction,
        )
    except (DinoBuildingsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
