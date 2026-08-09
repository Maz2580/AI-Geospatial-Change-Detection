# ArcGIS Pro Victorian label-feasibility pack

This pack is the first controlled test before any new model training. It creates a small, editable GeoPackage with two fixed Victorian Nearmap image pairs and six previously reviewed prompts.

The prompts are **not** training labels. They only locate sites. A real label is a human-drawn polygon around the complete, visible roof in the after image.

## Create the pack

Run this in the repository root with ArcGIS Pro's existing default Python:

```powershell
& 'C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe' src\create_arcgis_label_feasibility_pack.py --overwrite
```

It uses ArcGIS Pro's bundled GDAL/OGR writer, not ArcPy. No packages are installed, no ArcGIS environment is cloned or changed, and no model is trained.

It creates:

- `data/labels/arcgis_feasibility/victoria_building_change_feasibility.gpkg`
- `data/labels/arcgis_feasibility/label_pack_manifest.json`

The GeoPackage is EPSG:3857, the same projection as the frozen Nearmap GeoTIFFs. It contains native GeoPackage polygons that ArcGIS Pro can display and edit.

## Open and check it in ArcGIS Pro

1. Open `victoria_building_change_feasibility.gpkg` in the Catalog pane.
2. Right-click `<case>_review_prompts` and select **Add To Current Map**. Then right-click it in the Contents pane and select **Zoom To Layer**. Each prompt layer must show **three** polygons.
3. Add the matching before and after GeoTIFFs listed in `label_pack_manifest.json` to the same map. The GeoTIFFs are deliberately not copied into the GeoPackage.
4. Add the matching `<case>_after_roof_labels` layer. It is supposed to start with **zero** rows: this is the editable target layer for the human roof polygons.
5. Add `<case>_review_decisions` as a table. It has three review rows per case.

If a prompt layer has rows in its attribute table but no polygons after **Zoom To Layer**, stop and regenerate the pack with the command above. Do not label a package whose prompts do not draw on the map.

## Label protocol

1. In the after image, trace one complete, visible roof polygon for each clear new building or extension. Do not trace the prompt boundary, shadow, car, ship, tree canopy, paving, pool, or garden.
2. For every roof polygon, set `object_id`, `change_type` (`new_building` or `building_extension`), `source_prompt_ids`, `label_status=confirmed`, `label_source=manual_after_roof`, and your reviewer name.
3. For every prompt, complete the matching decision-table row. Use `assessment=confirmed_target_change`, `not_target_change`, `temporary_or_movable_change`, or `ambiguous`; set `decision_status=reviewed`. A negative or ambiguous review gets no roof polygon.

The `prior_review` fields retain our current evidence; they are not a substitute for a fresh human decision. Candidate geometry is deliberately kept out of the roof-label layer so it cannot accidentally become ground truth.

## Checkpoint, not training

The first checkpoint is three or more clearly visible, complete roof polygons with decisions for all six prompts. We then inspect label quality and decide whether it is worth building a broader local benchmark. This small pack is far too small to train a reliable deep-learning change detector; it is intended to prevent another long training loop based on poor labels.

If the labels prove sound, a later option is ArcGIS's supervised ChangeDetector workflow, which expects paired before/after tiles and actual binary change labels rather than prompt boxes. See the [Esri ChangeDetector sample](https://developers.arcgis.com/python/latest/samples/change-detection-of-buildings-from-satellite-imagery/).
