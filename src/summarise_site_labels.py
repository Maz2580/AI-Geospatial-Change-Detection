"""Report baseline metrics from manually labelled construction-site reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from building_change.evaluation import EvaluationError, load_and_summarise


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise manually labelled construction-site review results.")
    parser.add_argument("--labels", required=True, type=Path, help="Versioned human-label JSON file.")
    args = parser.parse_args()
    try:
        print(json.dumps(load_and_summarise(args.labels), indent=2))
    except EvaluationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
