"""Validate the committed pilot-evidence registry without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from building_change.validation import ValidationManifestError, load_validation_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the locked pilot evidence used for model comparison.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmarks/validation_manifest.json"),
        help="Validation manifest path (default: data/benchmarks/validation_manifest.json).",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root containing the locked artifacts.")
    parser.add_argument("--output", type=Path, help="Optional report JSON path.")
    args = parser.parse_args()
    try:
        report = validate_manifest(load_validation_manifest(args.manifest), root=args.root)
    except ValidationManifestError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
