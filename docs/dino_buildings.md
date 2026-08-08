# DINOv3 building-footprint evidence

`run_dino_buildings.py` adds a separate local model that is specialised for
building outlines. It runs `hotosm/dinov3s-buildings` on the before and after
Nearmap images independently, then compares the two footprint sets.

This is intended to address the current estate baseline: the RGB change stage
found only 9 of 19 known buildings and often supplied partial outlines. It is
not a replacement for change evidence or human review:

- a DINO footprint present only after is a **new-building candidate**, not a
  confirmation;
- it cannot identify pools, gardens, driveways, footpaths, soil disturbance,
  vehicles, or shadows as their own classes;
- a small date-to-date alignment change can look like an extension, so the
  footprint comparison uses distance and overlap tolerances.

Install the optional dependencies in the existing project virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

Run it with two already matched imagery dates. The first run downloads the
model to `data/output/model_cache`, which is ignored by Git. `HF_TOKEN` from
`.env` is used automatically when it is needed.

```powershell
.\venv\Scripts\python.exe src\run_dino_buildings.py `
  --before data\output\estate\imagery\before_2022-07-30_Vert_tiles.tif `
  --after data\output\estate\imagery\after_2026-04-18_Vert_tiles.tif `
  --before-capture-date 2022-07-30 --after-capture-date 2026-04-18 `
  --output data\output\estate\dino_buildings
```

The exact published defaults are a 256-pixel window, 192-pixel stride, and
building-probability threshold `0.4371`. Start with those values. The model is
run on CPU intentionally, keeping the setup reliable and isolated to `venv`;
enabling GPU runtime later does not change the algorithm or the review rule.

Review `footprint_change_candidates.geojson` using the normal before/after
review generator:

```powershell
.\venv\Scripts\python.exe src\generate_change_review.py `
  --before data\output\estate\imagery\before_2022-07-30_Vert_tiles.tif `
  --after data\output\estate\imagery\after_2026-04-18_Vert_tiles.tif `
  --candidates data\output\estate\dino_buildings\footprint_change_candidates.geojson `
  --output data\output\estate\dino_buildings_review
```

Only after this review should DINO footprints be fused with the RGB/D-FINE/
SegFormer evidence. Keep the source provenance in the output instead of
treating several model polygons as independent confirmations of the same roof.
