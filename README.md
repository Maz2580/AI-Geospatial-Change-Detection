# AI Geospatial Change Detection

This project detects construction change between two high-resolution aerial-image dates. It supports local GeoTIFFs and fixed-date Nearmap imagery.

## Operational workflow

1. Select an older and newer Nearmap survey for one location.
2. Download matching imagery for the same AOI.
3. Reproject and register the older image against the newer survey.
4. Combine colour and edge-change evidence, then vectorise significant regions.
5. If paired DSMs are available, use height gain to identify likely new buildings and height loss to identify likely demolitions.

Without elevation data, the result is labelled `likely_building_change`, not a confirmed new building: RGB imagery cannot always tell new construction from demolition, vegetation, or a large surface change.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `NEARMAP_API_KEY` in `.env`. `HF_TOKEN` remains available for the original DINOv2 and SAM experiments, but the operational baseline does not require a foundation model.

## Run with standard Nearmap Tile API

This is the default and works with standard `Vert` imagery access. The downloader retrieves all tiles for the small AOI from two exact surveys and creates matching Web Mercator GeoTIFF mosaics.

```powershell
.\venv\Scripts\python.exe src\run_building_change.py nearmap `
  --longitude 145.406921 --latitude -36.336606 --radius-m 100 `
  --before-date 2021-12-01 --after-date 2026-04-18 `
  --source tiles --tile-zoom 20 `
  --output data\output\example_site
```

`--before-date` selects the latest survey on or before the date. `--after-date` selects the earliest survey on or after it; omit it to use the latest available survey. Omit `--tile-zoom` to use the survey's native maximum zoom. Zoom 20 is a practical starting point for a 100 m radius because it uses far fewer tile requests while retaining building-scale detail.

## Optional Staticmap, True Ortho, and DSM workflow

If your Nearmap subscription enables Staticmap True Ortho and DSM, use the following. True Ortho reduces building lean and DSM height evidence makes construction direction much more reliable.

```powershell
.\venv\Scripts\python.exe src\run_building_change.py nearmap `
  --longitude 145.406921 --latitude -36.336606 --radius-m 100 `
  --before-date 2021-12-01 --after-date 2026-04-18 `
  --source staticmap --content-type TrueOrtho --with-dsm `
  --output data\output\example_site_dsm
```

Staticmap AOIs are limited to 1–100 m radius and 5000 × 5000 pixels.

## Run with existing GeoTIFFs

```powershell
.\venv\Scripts\python.exe src\run_building_change.py local `
  --before path\to\older.tif --after path\to\newer.tif `
  --before-dsm path\to\older_dsm.tif --after-dsm path\to\newer_dsm.tif `
  --output data\output\my_site
```

DSM inputs are optional but must be supplied as a pair.

## Outputs

- `change_score.tif` — continuous 0–1 change evidence.
- `change_mask.tif` — thresholded candidate regions.
- `construction_change_candidates.geojson` — all significant changes, with area, score, shape, date, and optional height evidence.
- `likely_new_buildings.geojson` — height-confirmed building gains, or conservative RGB-only building candidates when DSM is unavailable.
- `run_report.json` — parameters, registration result, and output paths.

Tune `--change-percentile` (default `98.5`), `--min-area-m2` (default `20`), and `--morphology-m` (default `0.6`) for your imagery and expected building size. A lower percentile detects more changes; a higher value reduces false positives.

## Validation

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The original CFNet, DINOv2, YOLO, and SAM scripts remain under `src/pixel_change`, `src/semantic_change`, and `src/vectorization` as experiments; they are not the main operational workflow.
