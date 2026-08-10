"""Review change candidates with a vision model and record advisory labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
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

from dotenv import load_dotenv  # noqa: E402

from building_change.visual_review import (  # noqa: E402
    OpenAICompatibleProvider,
    ProviderConfig,
    VisualReviewConfig,
    VisualReviewError,
    review_candidates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def _list_models(config: ProviderConfig) -> None:
    import requests

    response = requests.get(
        f"{config.base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {config.resolve_key()}"},
        timeout=config.timeout_s,
    )
    if response.status_code >= 400:
        raise SystemExit(f"ERROR: model listing returned HTTP {response.status_code}.")
    names = sorted(item.get("id", "") for item in response.json().get("data", []))
    hints = ("vision", "vl", "vila", "neva", "llava", "maverick", "scout", "nemotron")
    likely = [name for name in names if any(hint in name.lower() for hint in hints)]
    print(f"{len(names)} models reachable. Likely vision-capable:\n")
    for name in likely:
        print("  ", name)
    if not likely:
        print("  (none matched the vision hints; showing all)\n")
        for name in names:
            print("  ", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach vision-model review labels to change candidates.")
    parser.add_argument("--candidates", type=Path, help="Candidate GeoJSON to review.")
    parser.add_argument("--before", type=Path, help="Older RGB GeoTIFF.")
    parser.add_argument("--after", type=Path, help="Newer RGB GeoTIFF.")
    parser.add_argument("--output", type=Path, help="Annotated GeoJSON output path.")
    parser.add_argument("--before-date", default="earlier date", help="Shown to the model as the before capture date.")
    parser.add_argument("--after-date", default="later date", help="Shown to the model as the after capture date.")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1", help="OpenAI-compatible endpoint.")
    parser.add_argument("--model", default="meta/llama-3.2-90b-vision-instruct", help="Vision model identifier.")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY", help="Environment variable holding the API key.")
    parser.add_argument("--max-candidates", type=int, help="Review only the N largest candidates.")
    parser.add_argument("--min-area-m2", type=float, default=0.0, help="Skip candidates below this area.")
    parser.add_argument("--crop-padding-m", type=float, default=12.0, help="Context margin around each candidate.")
    parser.add_argument("--list-models", action="store_true", help="List reachable models and exit.")
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    provider_config = ProviderConfig(base_url=args.base_url, model=args.model, api_key_env=args.api_key_env)

    if args.list_models:
        _list_models(provider_config)
        return

    missing = [name for name in ("candidates", "before", "after", "output") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"ERROR: --{', --'.join(missing)} required unless --list-models is used.")

    collection = json.loads(args.candidates.read_text(encoding="utf-8"))
    try:
        report = review_candidates(
            collection,
            args.before,
            args.after,
            OpenAICompatibleProvider(provider_config),
            before_date=args.before_date,
            after_date=args.after_date,
            config=VisualReviewConfig(
                crop_padding_m=args.crop_padding_m,
                max_candidates=args.max_candidates,
                min_area_m2=args.min_area_m2,
            ),
        )
    except VisualReviewError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    report["output"] = str(args.output)
    report["model"] = args.model
    (args.output.parent / "visual_review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
