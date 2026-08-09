# ArcGIS Pro review imagery

Run the following from the repository root after creating the label GeoPackage:

```powershell
& 'C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe' src\create_arcgis_review_context.py --overwrite
```

The command creates a small before/after Nearmap crop around each prompt set under `data/labels/arcgis_feasibility/imagery/`. The crops are in EPSG:3857 and align directly with the GeoPackage prompt and label layers.

For each case, add the matching `_before_context.tif`, `_after_context.tif`, and `_review_prompts` layer to the same ArcGIS Pro map. Put the after image above the before image, then use transparency or swipe to inspect the actual reference evidence before drawing into the empty `_after_roof_labels` layer.

The context imagery is a convenience copy for review only. It is not committed and does not change the original Nearmap files.
