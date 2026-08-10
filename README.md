# AI Geospatial Change Detection

This project detects construction change between two high-resolution aerial-image dates. It supports local GeoTIFFs and fixed-date Nearmap imagery.

## Operational workflow

1. Select an older and newer Nearmap survey for one location.
2. Download matching imagery for the same AOI.
3. Reproject and register the older image against the newer survey.
4. Combine colour and edge-change evidence, then vectorise significant regions.
5. Review each candidate in an annotated before/after report.
6. If paired DSMs are available, use height gain to identify likely new buildings and height loss to identify likely demolitions.

Without elevation data, a result is labelled `likely_building_change`, not a confirmed new building: RGB imagery cannot always tell new construction from demolition, vegetation, or a large surface change.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `NEARMAP_API_KEY` in `.env`. `HF_TOKEN` remains available for the original DINOv2 and SAM experiments, but the operational baseline does not require a foundation model. `NVIDIA_API_KEY` is optional and only used by the vision-model review step.

Optional model stages need `requirements-models.txt` as well:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

## Detection presets

The detector ships two measured parameter sets. Both were scored against the
Melbourne CBD West 2020→2023 reference benchmark, which has 7 human-confirmed
visible changes (`data/benchmarks/.../baseline_measurements.json`).

| Preset | Parameters | Reference recall |
| --- | --- | --- |
| `balanced` (default) | percentile 98.5, min area 20 m², morphology 0.6 m, shadow deferred | **0.00** |
| `high-recall` | percentile 96.0, min area 10 m², morphology 0.2 m, shadow filter off | **1.00** from 22 candidates |

`balanced` is deliberately strict and defers shadow-affected components to
`uncertain_shadow_candidates.geojson`. On a dense or heavily shadowed scene it can
defer everything and return nothing, so check `candidate_count` before concluding
there was no change. On the Murchison 2021→2026 pair it returns 0 candidates while
`high-recall` returns 113.

`high-recall` trades precision for coverage: it recovers every confirmed benchmark
change but also produces candidates that need review. Use it when a miss costs more
than a false positive.

```powershell
.\venv\Scripts\python.exe src\run_building_change.py local `
  --before path\to\older.tif --after path\to\newer.tif `
  --preset high-recall --output data\output\my_site
```

Individual flags still override the preset.

## Footprint regularisation

Segmentation masks vectorise into pixel-staircase polygons. Every candidate outline
is passed through a dominant-orientation regularisation step
(`buildingregulariser`, MIT) that snaps edges parallel and perpendicular to each
building's own dominant axis, and optionally aligns neighbours onto a shared estate grid.

Measured on the Murchison DINO footprints: edge angle error 15.87° → 4.97° (−69%),
vertices per polygon 30.7 → 24.8, area preserved 97.9%. On the pixel detector,
vertices per polygon 338.4 → 29.6 (−91%).

Set the tolerance to 2–3× the imagery pixel size (0.15–0.225 m for 7.5 cm imagery).
Disable with `--no-regularise` on `run_dino_buildings.py`.

## Run with standard Nearmap Tile API

This is the default and works with standard `Vert` imagery access. The downloader retrieves all tiles for the small AOI from two exact surveys and creates matching Web Mercator GeoTIFF mosaics.

```powershell
.\venv\Scripts\python.exe src\run_building_change.py nearmap `
  --longitude 145.406921 --latitude -36.336606 --radius-m 100 `
  --before-date 2021-12-01 --after-date 2026-04-18 `
  --source tiles --tile-zoom 20 `
  --output data\output\example_site
```

`--before-date` selects the latest survey on or before the date. `--after-date` selects the earliest survey on or after it; omit it to use the latest available survey. Omit `--tile-zoom` to use the survey's native maximum zoom. Zoom 20 is a practical starting point for a 100 m radius because it uses fewer tile requests while retaining building-scale detail.

## Review a completed run

The review report creates a self-contained HTML page plus annotated before/after image chips. Open `review\index.html` in a browser and verify every RGB-only candidate before treating it as a new building.

```powershell
.\venv\Scripts\python.exe src\generate_change_review.py `
  --before data\output\example_site\imagery\before_2021-12-01_Vert_tiles.tif `
  --after data\output\example_site\imagery\after_2026-04-18_Vert_tiles.tif `
  --candidates data\output\example_site\construction_change_candidates.geojson `
  --output data\output\example_site
```

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
- `uncertain_shadow_candidates.geojson` — components deferred by the shadow filter. Check this when `candidate_count` is 0.
- `review/index.html` — annotated before/after candidate report for manual QA.
- `run_report.json` — parameters, registration result, and output paths.

Tune `--change-percentile` (default `98.5`), `--min-area-m2` (default `20`), and `--morphology-m` (default `0.6`) for your imagery and expected building size. A lower percentile detects more changes; a higher value reduces false positives. Prefer `--preset high-recall` over hand-tuning these, since that combination is measured.

## Building footprint extraction

Runs `hotosm/dinov3s-buildings` on each date separately and compares the two
footprint sets. Footprints are dated evidence, not confirmed construction.

```powershell
.\venv\Scripts\python.exe src\run_dino_buildings.py `
  --before path\to\older.tif --after path\to\newer.tif `
  --before-capture-date 2021-12-01 --after-capture-date 2026-04-18 `
  --output data\output\footprints
```

The model wraps a frozen DINOv3 ViT-S/16 backbone and therefore requires
ImageNet-normalised input. Feeding it raw 0–255 pixels saturates the logits and
collapses the building channel: on the Murchison estate that produced 899 m² of
building across 21.6 ha (0.4% coverage, median footprint 15 m²). With correct
normalisation the same scene yields 36,007 m² (16.6% coverage, median footprint
86 m², 112 footprints above 100 m²). If you adapt this module for another ONNX
segmentation model, check its expected input range first.

## Vision-model candidate review (optional)

Classifies what changed inside each candidate — new building, extension, solar
panels, hardscape, vegetation — using a hosted vision model. The label is advisory
and never overwrites `classification`.

```powershell
.\venv\Scripts\python.exe src\review_candidates_visually.py --list-models
```

See [docs/nvidia_visual_review.md](docs/nvidia_visual_review.md) for setup, models,
API details, and troubleshooting.

## Outputs

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe -m pytest tests -q
```

Or with the standard library runner:

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Score a candidate set against a human-labelled reference benchmark:

```powershell
.\venv\Scripts\python.exe src\evaluate_reference_benchmark.py `
  --reference data\benchmarks\melbourne_cbd_west_2020_2023\reference\footprint_change_candidates.geojson `
  --labels data\labels\melbourne_cbd_west_2020_2023_reference_review_labels.json `
  --predictions data\output\my_site\construction_change_candidates.geojson
```

The original CFNet, DINOv2, YOLO, and SAM scripts remain under `src/pixel_change`, `src/semantic_change`, and `src/vectorization` as experiments; they are not the main operational workflow. One-off investigations live in `src/experiments`.
