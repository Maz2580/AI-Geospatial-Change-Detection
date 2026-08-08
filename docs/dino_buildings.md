# DINOv3 building-footprint experiment (rejected)

`hotosm/dinov3s-buildings` was tested as a separate local building-footprint
source on the Murchison estate 2022--2026 pair. It did **not** meet the review
standard: it detected shadow/unrelated areas and missed real buildings. It is
therefore not an accepted candidate source and must not be fused into an
operational change result.

The adapter remains in the repository for reproducible research only. Its
result is always dated object evidence, never a confirmed building:

- the model has not been trained or calibrated on the project's Nearmap data;
- building polygons and aerial roofs are not the same geometry;
- it does not classify pools, gardens, driveways, footpaths, soil, vehicles,
  or shadows;
- lowering its threshold added false positives faster than it added real
  buildings in the estate review.

Do not spend more time tuning this model on that single estate. The next stage
is the independent Australian benchmark in
[`official_footprint_benchmark.md`](official_footprint_benchmark.md): first
validate reference changes on exact aerial dates, then measure every detector
against the approved labels. Only a detector with measured improvement on that
held-out benchmark can re-enter the fusion workflow.

For archival reproduction only, install the optional dependencies in the
project virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

The command remains available but is experimental:

```powershell
.\venv\Scripts\python.exe src\run_dino_buildings.py `
  --before path\to\before.tif --after path\to\after.tif `
  --output data\output\experimental_dino
```

Do not interpret its output as a building label without independent review.
