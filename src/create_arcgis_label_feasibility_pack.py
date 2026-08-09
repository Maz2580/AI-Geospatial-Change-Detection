"""Create a small ArcGIS-readable GeoPackage for manual Victorian roof labels.

This is a ground-truth feasibility step, not a model-training pipeline. Review
prompts locate sites only; a reviewer must trace the complete visible roof in
the after image rather than copying a prompt polygon.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from datetime import datetime, timezone
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/arcgis_feasibility"))
    parser.add_argument("--overwrite", action="store_true", help="Replace the generated GeoPackage.")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _feature_lookup(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(feature["properties"]["candidate_id"]): feature for feature in payload["features"]}


def _web_mercator(longitude: float, latitude: float) -> tuple[float, float]:
    """Project a WGS84 coordinate to EPSG:3857 without an environment dependency."""
    latitude = max(min(latitude, 85.05112878), -85.05112878)
    radius = 6378137.0
    x = radius * math.radians(longitude)
    y = radius * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
    return x, y


def _polygon_blob(geojson_geometry: dict[str, Any]) -> tuple[bytes, tuple[float, float, float, float]]:
    if geojson_geometry["type"] != "Polygon":
        raise ValueError(f"Only Polygon prompts are supported, got {geojson_geometry['type']}")
    rings = [[_web_mercator(float(x), float(y)) for x, y in ring] for ring in geojson_geometry["coordinates"]]
    flat = [point for ring in rings for point in ring]
    min_x = min(point[0] for point in flat)
    max_x = max(point[0] for point in flat)
    min_y = min(point[1] for point in flat)
    max_y = max(point[1] for point in flat)
    wkb = bytearray(struct.pack("<BII", 1, 3, len(rings)))
    for ring in rings:
        wkb.extend(struct.pack("<I", len(ring)))
        for x, y in ring:
            wkb.extend(struct.pack("<dd", x, y))
    # GeoPackage header: little-endian, XY envelope, EPSG:3857, then WKB polygon.
    header = b"GP" + b"\x00" + b"\x03" + struct.pack("<i", 3857) + struct.pack("<dddd", min_x, max_x, min_y, max_y)
    return bytes(header + wkb), (min_x, max_x, min_y, max_y)


def _create_metadata(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA application_id = 1196437808;
        PRAGMA user_version = 10300;
        PRAGMA foreign_keys = ON;
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        );
        CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
            srs_id INTEGER,
            CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        );
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            PRIMARY KEY (table_name, column_name),
            CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
            CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        );
        """
    )
    connection.executemany(
        "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
        (
            ("Undefined Cartesian", -1, "NONE", -1, "undefined", "undefined Cartesian coordinate reference system"),
            ("Undefined geographic", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
            ("WGS 84 geodetic", 4326, "EPSG", 4326, "GEOGCS[\"WGS 84\",DATUM[\"World Geodetic System 1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]", ""),
            ("WGS 84 / Pseudo-Mercator", 3857, "EPSG", 3857, "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"World Geodetic System 1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1]]", ""),
        ),
    )


def _add_feature_table(connection: sqlite3.Connection, table_name: str, description: str, empty: bool = False) -> None:
    connection.execute(
        f"""CREATE TABLE \"{table_name}\" (
            fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            geom BLOB,
            object_id TEXT,
            change_type TEXT,
            source_prompt_ids TEXT,
            label_status TEXT,
            label_source TEXT,
            reviewer TEXT,
            review_notes TEXT
        )"""
        if empty
        else f"""CREATE TABLE \"{table_name}\" (
            fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            geom BLOB NOT NULL,
            prompt_id TEXT NOT NULL,
            source_candidate_id INTEGER,
            prior_review TEXT,
            prior_change_type TEXT,
            review_instruction TEXT,
            geometry_role TEXT NOT NULL
        )"""
    )
    connection.execute(
        "INSERT INTO gpkg_contents (table_name, data_type, identifier, description, last_change, srs_id) VALUES (?, 'features', ?, ?, ?, 3857)",
        (table_name, table_name, description, _now()),
    )
    connection.execute(
        "INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', 'POLYGON', 3857, 0, 0)",
        (table_name,),
    )


def _add_decision_table(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"""CREATE TABLE \"{table_name}\" (
            fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            prompt_id TEXT NOT NULL,
            prior_review TEXT,
            prior_change_type TEXT,
            assessment TEXT,
            change_type TEXT,
            decision_status TEXT NOT NULL,
            reviewer TEXT,
            review_notes TEXT
        )"""
    )
    connection.execute(
        "INSERT INTO gpkg_contents (table_name, data_type, identifier, description, last_change) VALUES (?, 'attributes', ?, ?, ?)",
        (table_name, table_name, "Human decisions for the review prompts", _now()),
    )


def _create_case(connection: sqlite3.Connection, root: Path, config: dict[str, Any]) -> dict[str, str]:
    case_id = config["case_id"]
    prompt_table = f"{case_id}_review_prompts"
    label_table = f"{case_id}_after_roof_labels"
    decision_table = f"{case_id}_review_decisions"
    _add_feature_table(connection, prompt_table, "Reference prompts only; never copy their geometry into a roof label.")
    _add_feature_table(connection, label_table, "Human-drawn, complete visible after-date roof polygons only.", empty=True)
    _add_decision_table(connection, decision_table)
    candidates = _feature_lookup(root / config["prompts"])
    bounds: list[tuple[float, float, float, float]] = []
    for candidate_id, prior_review, prior_change_type, instruction in config["items"]:
        feature = candidates.get(candidate_id)
        if feature is None:
            raise ValueError(f"Candidate {candidate_id} not found in {config['prompts']}")
        prompt_id = f"{case_id}-candidate-{candidate_id}"
        blob, bbox = _polygon_blob(feature["geometry"])
        bounds.append(bbox)
        connection.execute(
            f"INSERT INTO \"{prompt_table}\" (geom, prompt_id, source_candidate_id, prior_review, prior_change_type, review_instruction, geometry_role) VALUES (?, ?, ?, ?, ?, ?, 'review_prompt_only')",
            (blob, prompt_id, candidate_id, prior_review, prior_change_type, instruction),
        )
        connection.execute(
            f"INSERT INTO \"{decision_table}\" (prompt_id, prior_review, prior_change_type, decision_status, review_notes) VALUES (?, ?, ?, 'pending_human_review', ?)",
            (prompt_id, prior_review, prior_change_type, instruction),
        )
    min_x = min(item[0] for item in bounds)
    max_x = max(item[1] for item in bounds)
    min_y = min(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    connection.execute(
        "UPDATE gpkg_contents SET min_x=?, min_y=?, max_x=?, max_y=? WHERE table_name=?",
        (min_x, min_y, max_x, max_y, prompt_table),
    )
    return {"review_prompts": prompt_table, "after_roof_labels": label_table, "review_decisions": decision_table}


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "victoria_building_change_feasibility.gpkg"
    if package_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{package_path} already exists. Re-run with --overwrite to recreate it.")
        package_path.unlink()
    manifest: dict[str, Any] = {
        "purpose": "Manual roof-label feasibility pack; it is not a model-training dataset.",
        "format": "GeoPackage (EPSG:3857), editable in ArcGIS Pro without an ArcPy environment.",
        "package": str(package_path),
        "cases": [],
    }
    with sqlite3.connect(package_path) as connection:
        _create_metadata(connection)
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
                    "datasets": _create_case(connection, root, config),
                }
            )
    manifest_path = output_dir / "label_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created label pack: {package_path}")
    print(f"Created manifest: {manifest_path}")


if __name__ == "__main__":
    main()
