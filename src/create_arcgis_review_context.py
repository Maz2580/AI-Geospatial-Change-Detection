"""Create compact before/after Nearmap context GeoTIFFs for ArcGIS review.

The crop extent is derived from each review-prompt layer, plus a configurable
buffer in metres.  It is intentionally separate from the editable GeoPackage:
the source imagery remains unchanged and the small crops are easy to add to an
ArcGIS Pro map beside the prompts and human-label layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = {
    "docklands_pilot_2020_2023": {
        "before": "data/output/docklands_pilot_2020_2023_imagery/imagery/before_2020-04-28_Vert_tiles.tif",
        "after": "data/output/docklands_pilot_2020_2023_imagery/imagery/after_2023-07-06_Vert_tiles.tif",
    },
    "melbourne_cbd_west_extension_a_2020_2023": {
        "before": "data/output/melbourne_cbd_west_extension_a_2020_2023_imagery/imagery/before_2020-11-08_Vert_tiles.tif",
        "after": "data/output/melbourne_cbd_west_extension_a_2020_2023_imagery/imagery/after_2023-11-10_Vert_tiles.tif",
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/arcgis_feasibility"))
    parser.add_argument("--buffer-metres", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _crop(gdal, source: Path, destination: Path, extent: tuple[float, float, float, float], overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists. Re-run with --overwrite to replace it.")
    min_x, max_x, min_y, max_y = extent
    options = gdal.TranslateOptions(
        format="GTiff",
        projWin=(min_x, max_y, max_x, min_y),
        projWinSRS="EPSG:3857",
        creationOptions=("TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"),
    )
    output = gdal.Translate(str(destination), str(source), options=options)
    if output is None:
        raise RuntimeError(f"Could not create {destination}")
    output = None


def main() -> None:
    args = _arguments()
    if args.buffer_metres < 0:
        raise ValueError("--buffer-metres must be zero or greater")
    try:
        from osgeo import gdal, ogr
    except ModuleNotFoundError as error:
        raise RuntimeError("Run this script with ArcGIS Pro's existing Python installation.") from error

    gdal.UseExceptions()
    ogr.UseExceptions()
    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    package_path = output_dir / "victoria_building_change_feasibility.gpkg"
    if not package_path.exists():
        raise FileNotFoundError(f"Create the label GeoPackage first: {package_path}")
    imagery_dir = output_dir / "imagery"
    imagery_dir.mkdir(parents=True, exist_ok=True)
    dataset = ogr.Open(str(package_path), 0)
    if dataset is None:
        raise RuntimeError(f"Could not open {package_path}")

    manifest_path = output_dir / "label_pack_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    context_paths: dict[str, dict[str, str]] = {}
    for case_id, images in CASES.items():
        layer = dataset.GetLayerByName(f"{case_id}_review_prompts")
        if layer is None or layer.GetFeatureCount() == 0:
            raise RuntimeError(f"No prompt geometry found for {case_id}")
        min_x, max_x, min_y, max_y = layer.GetExtent()
        extent = (
            min_x - args.buffer_metres,
            max_x + args.buffer_metres,
            min_y - args.buffer_metres,
            max_y + args.buffer_metres,
        )
        paths: dict[str, str] = {}
        for epoch, relative_source in images.items():
            source = root / relative_source
            if not source.exists():
                raise FileNotFoundError(source)
            destination = imagery_dir / f"{case_id}_{epoch}_context.tif"
            _crop(gdal, source, destination, extent, args.overwrite)
            paths[epoch] = str(destination)
        context_paths[case_id] = paths
        print(f"Created {case_id} context imagery: {extent}")
    dataset = None
    manifest["review_context_imagery"] = context_paths
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
