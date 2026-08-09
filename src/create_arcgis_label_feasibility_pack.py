"""Create a small ArcGIS Pro GeoPackage for manual Victorian roof labels.

This is a ground-truth feasibility step, not a model-training pipeline. Review
prompts locate sites only; a reviewer must trace the complete visible roof in
the after image rather than copying a prompt polygon.

Run this with ArcGIS Pro's existing Python installation. It uses the bundled
GDAL/OGR GeoPackage writer, not ArcPy, and does not alter any environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PACKS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "docklands_pilot_2020_2023",
        "before_image": "data/output/docklands_pilot_2020_2023_imagery/imagery/before_2020-04-28_Vert_tiles.tif",
        "after_image": "data/output/docklands_pilot_2020_2023_imagery/imagery/after_2023-07-06_Vert_tiles.tif",
        "prompts": "data/benchmarks/docklands_pilot_2020_2023/umami_dfine_change/umami_dfine_change_all_candidates.geojson",
        "items": (
            (1, "confirmed_target_change", "new_building", "Visible new building. Trace the complete visible after-date roof, not this prompt outline."),
            (2, "temporary_or_movable_change", "not_a_building", "Ship/cruise context. Record a decision, but do not create a roof polygon."),
            (3, "temporary_or_movable_change", "not_a_building", "Temporary white-roof structure. Record a decision, but do not create a roof polygon."),
        ),
    },
    {
        "case_id": "melbourne_cbd_west_extension_a_2020_2023",
        "before_image": "data/output/melbourne_cbd_west_extension_a_2020_2023_imagery/imagery/before_2020-11-08_Vert_tiles.tif",
        "after_image": "data/output/melbourne_cbd_west_extension_a_2020_2023_imagery/imagery/after_2023-11-10_Vert_tiles.tif",
        "prompts": "data/benchmarks/melbourne_cbd_west_2020_2023/umami_extension_a_dfine/all_candidates.geojson",
        "items": (
            (1, "confirmed_target_change", "building_extension", "Visible building extension. Trace the complete visible after-date roof/extension geometry."),
            (2, "not_target_change", "not_a_building", "Not a target building change. Record a decision, but do not create a roof polygon."),
            (3, "confirmed_target_change", "new_building", "Challenging visible change. Trace only a roof boundary you can see with confidence; otherwise mark ambiguous."),
        ),
    },
)

TEXT_FIELDS: tuple[tuple[str, int], ...] = (
    ("object_id", 64),
    ("change_type", 48),
    ("source_prompt_ids", 128),
    ("label_status", 32),
    ("label_source", 32),
    ("reviewer", 80),
    ("review_notes", 500),
)
PROMPT_FIELDS: tuple[tuple[str, int], ...] = (
    ("prompt_id", 80),
    ("prior_review", 48),
    ("prior_change_type", 48),
    ("review_instruction", 500),
    ("geometry_role", 48),
)
DECISION_FIELDS: tuple[tuple[str, int], ...] = (
    ("prompt_id", 80),
    ("prior_review", 48),
    ("prior_change_type", 48),
    ("assessment", 48),
    ("change_type", 48),
    ("decision_status", 32),
    ("reviewer", 80),
    ("review_notes", 500),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/arcgis_feasibility"))
    parser.add_argument("--overwrite", action="store_true", help="Replace the generated GeoPackage.")
    return parser.parse_args()


def _feature_lookup(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(feature["properties"]["candidate_id"]): feature for feature in payload["features"]}


def _spatial_references(osr: Any) -> tuple[Any, Any, Any]:
    source = osr.SpatialReference()
    target = osr.SpatialReference()
    source.ImportFromEPSG(4326)
    target.ImportFromEPSG(3857)
    # GeoJSON coordinates are longitude/latitude. Avoid GDAL 3 authority-axis swapping.
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return source, target, osr.CoordinateTransformation(source, target)


def _polygon_from_geojson(ogr: Any, geometry: dict[str, Any], transform: Any) -> Any:
    if geometry["type"] != "Polygon":
        raise ValueError(f"Only Polygon prompts are supported, got {geometry['type']}")
    polygon = ogr.Geometry(ogr.wkbPolygon)
    for coordinates in geometry["coordinates"]:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for longitude, latitude in coordinates:
            ring.AddPoint(float(longitude), float(latitude))
        polygon.AddGeometry(ring)
    if polygon.Transform(transform) != ogr.OGRERR_NONE:
        raise RuntimeError("Could not transform a prompt polygon to EPSG:3857")
    return polygon


def _add_fields(ogr: Any, layer: Any, fields: tuple[tuple[str, int], ...]) -> None:
    for name, width in fields:
        field = ogr.FieldDefn(name, ogr.OFTString)
        field.SetWidth(width)
        if layer.CreateField(field) != ogr.OGRERR_NONE:
            raise RuntimeError(f"Could not add field {name} to {layer.GetName()}")


def _create_feature_layer(ogr: Any, dataset: Any, name: str, spatial_ref: Any, fields: tuple[tuple[str, int], ...]) -> Any:
    layer = dataset.CreateLayer(name, spatial_ref, ogr.wkbPolygon, options=["SPATIAL_INDEX=YES"])
    if layer is None:
        raise RuntimeError(f"Could not create GeoPackage layer {name}")
    _add_fields(ogr, layer, fields)
    return layer


def _create_table(ogr: Any, dataset: Any, name: str) -> Any:
    layer = dataset.CreateLayer(name, geom_type=ogr.wkbNone)
    if layer is None:
        raise RuntimeError(f"Could not create GeoPackage table {name}")
    _add_fields(ogr, layer, DECISION_FIELDS)
    return layer


def _set_fields(feature: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        if value is not None:
            feature.SetField(name, value)


def _create_case(ogr: Any, dataset: Any, root: Path, config: dict[str, Any], target_sr: Any, transform: Any) -> dict[str, str]:
    case_id = config["case_id"]
    prompt_name = f"{case_id}_review_prompts"
    label_name = f"{case_id}_after_roof_labels"
    decision_name = f"{case_id}_review_decisions"
    prompts = _create_feature_layer(ogr, dataset, prompt_name, target_sr, PROMPT_FIELDS + (("source_candidate_id", 16),))
    labels = _create_feature_layer(ogr, dataset, label_name, target_sr, TEXT_FIELDS)
    decisions = _create_table(ogr, dataset, decision_name)
    candidates = _feature_lookup(root / config["prompts"])
    for candidate_id, prior_review, prior_change_type, instruction in config["items"]:
        source_feature = candidates.get(candidate_id)
        if source_feature is None:
            raise ValueError(f"Candidate {candidate_id} not found in {config['prompts']}")
        prompt_id = f"{case_id}-candidate-{candidate_id}"
        prompt = ogr.Feature(prompts.GetLayerDefn())
        _set_fields(
            prompt,
            {
                "prompt_id": prompt_id,
                "source_candidate_id": str(candidate_id),
                "prior_review": prior_review,
                "prior_change_type": prior_change_type,
                "review_instruction": instruction,
                "geometry_role": "review_prompt_only",
            },
        )
        prompt.SetGeometry(_polygon_from_geojson(ogr, source_feature["geometry"], transform))
        if prompts.CreateFeature(prompt) != ogr.OGRERR_NONE:
            raise RuntimeError(f"Could not write {prompt_id}")
        prompt = None
        decision = ogr.Feature(decisions.GetLayerDefn())
        _set_fields(
            decision,
            {
                "prompt_id": prompt_id,
                "prior_review": prior_review,
                "prior_change_type": prior_change_type,
                "decision_status": "pending_human_review",
                "review_notes": instruction,
            },
        )
        if decisions.CreateFeature(decision) != ogr.OGRERR_NONE:
            raise RuntimeError(f"Could not write decision for {prompt_id}")
        decision = None
    # Keep an explicit reference so the empty editable target layer is not optimised away.
    labels.SyncToDisk()
    prompts.SyncToDisk()
    decisions.SyncToDisk()
    return {"review_prompts": prompt_name, "after_roof_labels": label_name, "review_decisions": decision_name}


def main() -> None:
    args = _arguments()
    try:
        from osgeo import ogr, osr
    except ModuleNotFoundError as error:
        raise RuntimeError("Run this script with ArcGIS Pro's existing Python installation.") from error

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "victoria_building_change_feasibility.gpkg"
    driver = ogr.GetDriverByName("GPKG")
    if driver is None:
        raise RuntimeError("ArcGIS Pro's GDAL installation does not provide the GPKG driver.")
    if package_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{package_path} already exists. Re-run with --overwrite to recreate it.")
        if driver.DeleteDataSource(str(package_path)) != ogr.OGRERR_NONE:
            raise RuntimeError(f"Could not replace {package_path}")
    dataset = driver.CreateDataSource(str(package_path))
    if dataset is None:
        raise RuntimeError(f"Could not create {package_path}")
    _, target_sr, transform = _spatial_references(osr)
    manifest: dict[str, Any] = {
        "purpose": "Manual roof-label feasibility pack; it is not a model-training dataset.",
        "format": "GDAL/OGR GeoPackage (EPSG:3857), editable in ArcGIS Pro.",
        "package": str(package_path),
        "cases": [],
    }
    for config in PACKS:
        for key in ("before_image", "after_image", "prompts"):
            source_path = root / config[key]
            if not source_path.exists():
                raise FileNotFoundError(source_path)
        manifest["cases"].append(
            {
                "case_id": config["case_id"],
                "before_image": str((root / config["before_image"]).resolve()),
                "after_image": str((root / config["after_image"]).resolve()),
                "datasets": _create_case(ogr, dataset, root, config, target_sr, transform),
            }
        )
    dataset = None
    manifest_path = output_dir / "label_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created label pack: {package_path}")
    print(f"Created manifest: {manifest_path}")


if __name__ == "__main__":
    main()
