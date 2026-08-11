"""Draw a stratified review sample and score the labels it produces."""

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

from building_change.review_sampling import (  # noqa: E402
    SampleConfig,
    SamplingError,
    draw_sample,
    estimate_precision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a stratified review sample, or score its labels.")
    parser.add_argument("--candidates", required=True, type=Path, help="Candidate GeoJSON to sample from.")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the sample GeoJSON.")
    parser.add_argument("--target-size", type=int, default=25, help="Approximate number to review (default: 25).")
    parser.add_argument("--seed", type=int, default=20260811, help="Sampling seed, for reproducibility.")
    parser.add_argument("--labels", type=Path, help="Score an existing label file instead of drawing a new sample.")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))

    try:
        sample, plan = draw_sample(candidates, SampleConfig(target_size=args.target_size, seed=args.seed))
    except SamplingError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    if args.labels:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else sample
        result = estimate_precision(labels.get("labels", labels), plan, existing)
        print(json.dumps(result, indent=2))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    (args.output.parent / "sampling_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    template = {
        "dataset": args.candidates.stem,
        "instructions": "Set each value to 'correct' if the polygon really is the change it claims, else 'wrong'.",
        "labels": {
            str(feature["properties"]["candidate_id"]): "unlabelled"
            for feature in sample["features"]
        },
    }
    template_path = args.output.parent / "label_template.json"
    if not template_path.exists():
        template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")

    print(json.dumps(plan, indent=2))
    print(f"\nsample      {args.output}")
    print(f"label file  {template_path}")


if __name__ == "__main__":
    main()
