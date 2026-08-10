"""Footprint-led candidate assembly on the Murchison pair, with a size comparison."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from building_change.corroboration import CorroborationConfig, corroborate_footprints

FOOTPRINTS = ROOT / r"data\output\dino_fixed\footprint_change_candidates.geojson"
EVIDENCE = {
    "pixel_change": ROOT / r"data\output\preset_high_recall_reg\construction_change_candidates.geojson",
    "dinov2_semantic": ROOT / r"data\output\enhanced_test_v4\dino_semantic\dino_semantic_change_candidates.geojson",
}
OLD_FUSED = ROOT / r"data\output\enhanced_test_v4\enhanced_fused_candidates.geojson"
FLAT_FUSED = ROOT / r"data\output\fixed_pipeline\fused_candidates.geojson"
OUT = ROOT / r"data\output\footprint_led"


def areas(path):
    if not Path(path).exists():
        return np.zeros(0)
    feats = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    return np.array([f["properties"].get("area_m2", 0.0) for f in feats])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    footprints = json.loads(FOOTPRINTS.read_text(encoding="utf-8"))
    evidence = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in EVIDENCE.items()
        if path.exists()
    }

    report = corroborate_footprints(footprints, evidence, config=CorroborationConfig())
    print(json.dumps(report, indent=2))

    (OUT / "footprint_led_candidates.geojson").write_text(json.dumps(footprints, indent=2), encoding="utf-8")
    (OUT / "corroboration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    by_tier = {}
    for feature in footprints["features"]:
        tier = feature["properties"]["change_support"]["tier"]
        by_tier.setdefault(tier, []).append(feature["properties"].get("area_m2", 0.0))

    print("\ncandidate size by support tier")
    for tier, values in sorted(by_tier.items()):
        arr = np.array(values)
        print(f"  {tier:22} n={len(arr):4d}  median={np.median(arr):6.0f} m2  "
              f">=60m2={(arr>=60).sum():3d}  >=100m2={(arr>=100).sum():3d}")

    print("\ncandidate geometry, old flat fusion vs footprint-led")
    for label, path in [
        ("old pipeline (v4 fused)", OLD_FUSED),
        ("flat fusion, fixed channels", FLAT_FUSED),
        ("footprint-led", OUT / "footprint_led_candidates.geojson"),
    ]:
        arr = areas(path)
        if not arr.size:
            continue
        print(f"  {label:30} n={len(arr):4d}  median={np.median(arr):6.0f} m2  "
              f"total={arr.sum():8,.0f} m2  >=100m2={(arr>=100).sum():3d}")


if __name__ == "__main__":
    main()
