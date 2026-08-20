"""Run all available change detection channels and fuse the results.

This script demonstrates the combined pipeline:
  1. Pixel-level change detection (existing detector)
  2. DINOv2 semantic change detection (zero-shot, no training)
  3. DINOv3s building footprint comparison
  4. Fusion of all sources
  5. Post-fusion architectural filter (calibrated config)

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
from building_change.whu_buildings import WhuBuildingsConfig, run_whu_building_comparison, WhuBuildingsError
from building_change.bit_inference import BITConfig, run_bit_change_detection, BITInferenceError
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
    """Run all detection channels, fuse results, and apply architectural filter."""
    results: dict = {}
    candidate_inputs: dict[str, dict] = {}
    # Load calibrated configuration if present
    calibrated_config_path = Path("config/calibrated_detection_config.json")
    calibrated_params = {}
    if calibrated_config_path.is_file():
        try:
            calibrated_data = json.loads(calibrated_config_path.read_text(encoding="utf-8"))
            calibrated_params = calibrated_data.get("optimal_parameters", {})
            logger.info("Loaded calibrated detection configuration from %s", calibrated_config_path)
            logger.info("Calibrated parameters: %s", json.dumps(calibrated_params))
        except Exception as exc:
            logger.warning("Failed to parse calibrated config: %s", exc)

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

    # ---- Channel 3: WHU-Building Unet++ Footprints ----
    if not skip_dino_buildings:
        logger.info("=" * 60)
        logger.info("CHANNEL 3: WHU-Building Footprint Extraction")
        logger.info("=" * 60)
        t0 = time.time()
        try:
            whu_bldg_dir = output_dir / "whu_buildings"
            whu_bldg_report = run_whu_building_comparison(
                before_image, after_image, whu_bldg_dir,
                config=WhuBuildingsConfig(),
                before_capture_date=before_date,
                after_capture_date=after_date,
            )
            results["whu_buildings"] = whu_bldg_report
            cand_count = whu_bldg_report.get("comparison", {}).get("candidate_count", 0)
            logger.info("WHU footprints: %d candidates (%.1fs)", cand_count, time.time() - t0)

            candidates_path = whu_bldg_dir / "after_whu_building_footprints.geojson"
            if candidates_path.exists():
                collection = json.loads(candidates_path.read_text(encoding="utf-8"))
                if collection.get("features"):
                    candidate_inputs["whu_footprints"] = collection
        except WhuBuildingsError as exc:
            logger.error("WHU footprints failed: %s", exc)
            results["whu_buildings"] = {"error": str(exc)}
    # ---- Channel 4: BIT Bitemporal Transformer (LEVIR-CD) ----
    logger.info("=" * 60)
    logger.info("CHANNEL 4: SOTA AdaptFormer LEVIR-CD Bitemporal Change Detection")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        bit_dir = output_dir / "bit_bitemporal"
        bit_report = run_bit_change_detection(
            before_image, after_image, bit_dir,
            config=BITConfig(tile_px=1024, stride_px=768, change_threshold=0.50),
            before_capture_date=before_date,
            after_capture_date=after_date,
        )
        results["bit_bitemporal"] = bit_report
        logger.info("BIT Bitemporal detection: %d candidates (%.1fs)",
                     bit_report["candidate_count"], time.time() - t0)

        candidates_path = bit_dir / "bit_change_candidates.geojson"
        if candidates_path.exists():
            collection = json.loads(candidates_path.read_text(encoding="utf-8"))
            if collection.get("features"):
                candidate_inputs["bit_bitemporal"] = collection
    except BITInferenceError as exc:
        logger.error("BIT Bitemporal detection failed: %s", exc)
        results["bit_bitemporal"] = {"error": str(exc)}
    except Exception as exc:
        logger.error("BIT Bitemporal detection failed unexpectedly: %s", exc)
        results["bit_bitemporal"] = {"error": str(exc)}

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

    # ---- Post-Fusion Architectural Filter ----
    if calibrated_params and results.get("fusion", {}).get("total_fused_candidates", 0) > 0:
        logger.info("=" * 60)
        logger.info("POST-FUSION: Applying calibrated architectural filter")
        logger.info("=" * 60)
        try:
            from building_change.architectural_filter import apply_architectural_filter
            fused_path = output_dir / "enhanced_fused_candidates.geojson"
            if fused_path.is_file():
                fused_collection = json.loads(fused_path.read_text(encoding="utf-8"))
                filtered_collection = apply_architectural_filter(
                    fused_collection,
                    calibrated_params,
                )
                filtered_path = output_dir / "filtered_building_candidates.geojson"
                filtered_path.write_text(
                    json.dumps(filtered_collection, indent=2), encoding="utf-8"
                )
                results["architectural_filter"] = {
                    "input_candidates": len(fused_collection.get("features", [])),
                    "output_candidates": len(filtered_collection.get("features", [])),
                    "eliminated": (
                        len(fused_collection.get("features", []))
                        - len(filtered_collection.get("features", []))
                    ),
                    "config_applied": calibrated_params,
                    "output": str(filtered_path),
                }
                logger.info(
                    "Architectural filter: %d → %d candidates (%d eliminated)",
                    results["architectural_filter"]["input_candidates"],
                    results["architectural_filter"]["output_candidates"],
                    results["architectural_filter"]["eliminated"],
                )
        except Exception as exc:
            logger.error("Architectural filter failed: %s", exc)
            results["architectural_filter"] = {"error": str(exc)}

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
