"""Create a human-review template from independent reference candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from building_change.annotations import AnnotationError, create_reference_label_template, load_annotation_document, validate_reference_label_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an unreviewed, human-label template for a fixed imagery pair.")
    parser.add_argument("--case-id")
    parser.add_argument("--reference", type=Path, help="Official/reference candidate GeoJSON; never copied into roof labels.")
    parser.add_argument("--before-date", help="Resolved fixed before-survey date (YYYY-MM-DD).")
    parser.add_argument("--after-date", help="Resolved fixed after-survey date (YYYY-MM-DD).")
    parser.add_argument("--output", type=Path, help="New draft JSON label document.")
    parser.add_argument("--validate", type=Path, help="Validate an existing annotation document instead of creating one.")
    parser.add_argument("--require-complete", action="store_true", help="With --validate, reject documents that still contain unreviewed candidates.")
    args = parser.parse_args()
    try:
        if args.validate:
            report = validate_reference_label_document(load_annotation_document(args.validate), require_complete=args.require_complete)
        else:
            required = {"--case-id": args.case_id, "--reference": args.reference, "--before-date": args.before_date, "--after-date": args.after_date, "--output": args.output}
            missing = [name for name, value in required.items() if value is None]
            if missing:
                parser.error(f"Missing required arguments when creating a template: {', '.join(missing)}")
            collection = load_annotation_document(args.reference)
            document = create_reference_label_template(
                case_id=args.case_id,
                reference_collection=collection,
                before_date=args.before_date,
                after_date=args.after_date,
                reference_path=args.reference.as_posix(),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
            report = validate_reference_label_document(document)
    except AnnotationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
