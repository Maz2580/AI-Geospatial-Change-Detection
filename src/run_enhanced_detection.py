"""Run all available change detection channels and fuse the results.

This script demonstrates the combined pipeline:
  1. Pixel-level change detection (existing detector)
  2. DINOv2 semantic change detection (zero-shot, no training)
  3. DINOv3s building footprint comparison
  4. Fusion of all sources

Usage:
    python src/run_enhanced_detection.py \
        --before data/input/.../before.tif \
        --after data/input/.../after.tif \
        --output data/output/enhanced_test \
        --before-date 2021-12-01 \
        --after-date 2026-04-18
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from building_change.detector import DetectionConfig, run_detection
from building_change.dino_change import DINOChangeConfig, DINOChangeError, run_dino_change_detection
from building_change.dino_buildings import DinoBuildingsConfig, run_dino_building_comparison, DinoBuildingsError
from building_change.fusion import fuse_candidates, FusionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enhanced_detection")


def run_enhanced_pipeline(
    before_image: Path,
    after_image: Path,
    output_dir: Path,
    *,
    before_date: str,
    after_date: str,
    skip_dino: bool = False,
    skip_dino_buildings: bool = False,
) -> dict:
    """Run all detection channels and fuse results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    candidate_inputs: dict[str, dict] = {}

    # ---- Channel 1: Existing pixel-level detector ----
    logger.info("=" * 60)
    logger.info("CHANNEL 1: Pixel-level change detection")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        pixel_dir = output_dir / "pixel_change"
        pixel_report = run_detection(
            before_image, after_image, pixel_dir,
            config=DetectionConfig(),
            before_capture_date=before_date,
            after_capture_date=after_date,
        )
        results["pixel_change"] = pixel_report
        logger.info("Pixel detection: %d candidates, %d uncertain (%.1fs)",
                     pixel_report["candidate_count"],
                     pixel_report.get("uncertain_shadow_candidate_count", 0),
                     time.time() - t0)

        # Load candidates for fusion (include uncertain shadow candidates too)
        for name, filename in [
            ("pixel_change", "construction_change_candidates.geojson"),
            ("pixel_uncertain", "uncertain_shadow_candidates.geojson"),
        ]:
            candidates_path = pixel_dir / filename
            if candidates_path.exists():
                collection = json.loads(candidates_path.read_text(encoding="utf-8"))
                if collection.get("features"):
                    candidate_inputs[name] = collection
    except Exception as exc:
        logger.error("Pixel detection failed: %s", exc)
        results["pixel_change"] = {"error": str(exc)}

    # ---- Channel 2: DINOv2 Semantic Change Detection ----
    if not skip_dino:
        logger.info("=" * 60)
        logger.info("CHANNEL 2: DINOv2 semantic change detection (zero-shot)")
        logger.info("=" * 60)
        t0 = time.time()
        try:
            dino_dir = output_dir / "dino_semantic"
            dino_report = run_dino_change_detection(
                before_image, after_image, dino_dir,
                config=DINOChangeConfig(),
                before_capture_date=before_date,
                after_capture_date=after_date,
            )
            results["dino_semantic"] = dino_report
            logger.info("DINOv2 detection: %d candidates (%.1fs)",
                         dino_report["candidate_count"], time.time() - t0)

            candidates_path = dino_dir / "dino_semantic_change_candidates.geojson"
            if candidates_path.exists():
                collection = json.loads(candidates_path.read_text(encoding="utf-8"))
                if collection.get("features"):
                    candidate_inputs["dinov2_semantic"] = collection
        except DINOChangeError as exc:
            logger.error("DINOv2 detection failed: %s", exc)
            results["dino_semantic"] = {"error": str(exc)}
        except Exception as exc:
            logger.error("DINOv2 detection failed unexpectedly: %s", exc)
            results["dino_semantic"] = {"error": str(exc)}

    # ---- Channel 3: DINOv3s Building Footprints ----
    if not skip_dino_buildings:
        logger.info("=" * 60)
        logger.info("CHANNEL 3: DINOv3s Building Footprint Extraction")
        logger.info("=" * 60)
        t0 = time.time()
        try:
            dino_bldg_dir = output_dir / "dino_buildings"
            dino_bldg_report = run_dino_building_comparison(
                before_image, after_image, dino_bldg_dir,
                config=DinoBuildingsConfig(),
                before_capture_date=before_date,
                after_capture_date=after_date,
            )
            results["dino_buildings"] = dino_bldg_report
            cand_count = dino_bldg_report.get("comparison", {}).get("candidate_count", 0)
            logger.info("DINOv3s footprints: %d candidates (%.1fs)", cand_count, time.time() - t0)

            candidates_path = dino_bldg_dir / "footprint_change_candidates.geojson"
            if candidates_path.exists():
                collection = json.loads(candidates_path.read_text(encoding="utf-8"))
                if collection.get("features"):
                    candidate_inputs["dino_footprints"] = collection
        except DinoBuildingsError as exc:
            logger.error("DINOv3s footprints failed: %s", exc)
            results["dino_buildings"] = {"error": str(exc)}
        except Exception as exc:
            logger.error("DINOv3s footprints failed unexpectedly: %s", exc)
            results["dino_buildings"] = {"error": str(exc)}

    # ---- Fusion ----
    logger.info("=" * 60)
    logger.info("FUSION: Combining %d sources: %s", len(candidate_inputs), list(candidate_inputs.keys()))
    logger.info("=" * 60)

    if len(candidate_inputs) >= 1:
        try:
            fused = fuse_candidates(candidate_inputs, merge_distance_m=3.0)
            fused_path = output_dir / "enhanced_fused_candidates.geojson"
            fused_path.write_text(json.dumps(fused, indent=2), encoding="utf-8")

            multi_source = sum(
                1 for f in fused["features"]
                if f["properties"]["source_count"] > 1
            )
            single_source = len(fused["features"]) - multi_source

            results["fusion"] = {
                "total_fused_candidates": len(fused["features"]),
                "multi_source_agreement": multi_source,
                "single_source_only": single_source,
                "sources_used": list(candidate_inputs.keys()),
                "output": str(fused_path),
            }
            logger.info(
                "Fusion results: %d total (%d multi-source, %d single-source)",
                len(fused["features"]), multi_source, single_source,
            )

            # Report per-candidate details
            for feature in fused["features"][:20]:  # Cap log output
                props = feature["properties"]
                logger.info(
                    "  Candidate %d: %s | sources=%s | area=%.0f m² | agreement=%s",
                    props["candidate_id"],
                    props["classification"],
                    props.get("candidate_sources", []),
                    props["area_m2"],
                    props["agreement"],
                )
            if len(fused["features"]) > 20:
                logger.info("  ... and %d more candidates", len(fused["features"]) - 20)
        except FusionError as exc:
            logger.error("Fusion failed: %s", exc)
            results["fusion"] = {"error": str(exc)}
    else:
        logger.warning("No candidates from any source — nothing to fuse.")
        results["fusion"] = {"total_fused_candidates": 0, "sources_used": []}

    # Write overall report
    overall_report_path = output_dir / "enhanced_detection_report.json"
    overall_report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Enhanced detection report: %s", overall_report_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run enhanced multi-channel change detection with fusion."
    )
    parser.add_argument("--before", required=True, type=Path, help="Older RGB GeoTIFF.")
    parser.add_argument("--after", required=True, type=Path, help="Newer RGB GeoTIFF.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--before-date", required=True, help="Before capture date (ISO).")
    parser.add_argument("--after-date", required=True, help="After capture date (ISO).")
    parser.add_argument("--skip-dino", action="store_true", help="Skip DINOv2 semantic channel.")
    parser.add_argument("--skip-dino-buildings", action="store_true", help="Skip DINOv3s footprint channel.")

    args = parser.parse_args()

    report = run_enhanced_pipeline(
        args.before, args.after, args.output,
        before_date=args.before_date,
        after_date=args.after_date,
        skip_dino=args.skip_dino,
        skip_dino_buildings=args.skip_dino_buildings,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
