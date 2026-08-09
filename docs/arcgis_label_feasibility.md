# ArcGIS Pro Victorian label-feasibility pack

This pack is the first controlled test before any new model training. It creates a small, editable GeoPackage with two fixed Victorian Nearmap image pairs and six previously reviewed prompts.

The prompts are **not** training labels. They only help locate a site. A real label is a human-drawn polygon around the complete, visible roof in the after image.

## Create the pack

Run this in the repository root with the project's existing virtual environment:

```powershell
.\venv\Scripts\python.exe src\create_arcgis_label_feasibility_pack.py
```

It creates:

- `data/labels/arcgis_feasibility/victoria_building_change_feasibility.gpkg`
- `data/labels/arcgis_feasibility/label_pack_manifest.json`

The GeoPackage is EPSG:3857, the same projection as the frozen Nearmap GeoTIFFs. It is editable in ArcGIS Pro. No packages are installed, no ArcGIS environment is cloned or changed, and no model is trained.

## Label in ArcGIS Pro

1. Open `victoria_building_change_feasibility.gpkg` in ArcGIS Pro. The manifest lists the exact before and after GeoTIFF paths for each case; add both images to the same map.
2. Add the three datasets for the case:
   - `<case>_review_prompts` — reference prompts only.
   - `<case>_after_roof_labels` — the only layer where roof polygons are drawn.
   - `<case>_review_decisions` — records non-buildings and ambiguous cases.
3. In the after image, trace one complete, visible roof polygon for each clear new building or extension. Do not trace the prompt boundary, shadow, car, ship, tree canopy, paving, pool, or garden.
4. For every roof polygon, set `object_id`, `change_type` (`new_building` or `building_extension`), `source_prompt_ids`, `label_status=confirmed`, `label_source=manual_after_roof`, and your reviewer name.
5. For every prompt, complete the matching decision-table row. Use `assessment=confirmed_target_change`, `not_target_change`, `temporary_or_movable_change`, or `ambiguous`; set `decision_status=reviewed`. A negative or ambiguous review gets no roof polygon.

The `prior_review` fields retain our current evidence; they are not a substitute for a fresh human decision. Candidate geometry is deliberately kept out of the roof-label layer so it cannot accidentally become ground truth.

## Checkpoint, not training

The first checkpoint is three or more clearly visible, complete roof polygons with decisions for all six prompts. We then inspect label quality and decide whether it is worth building a broader local benchmark. This small pack is far too small to train a reliable deep-learning change detector; it is intended to prevent another long training loop based on poor labels.

If the labels prove sound, a later option is ArcGIS's supervised ChangeDetector workflow, which expects paired before/after tiles and actual binary change labels rather than prompt boxes. See the [Esri ChangeDetector sample](https://developers.arcgis.com/python/latest/samples/change-detection-of-buildings-from-satellite-imagery/).
