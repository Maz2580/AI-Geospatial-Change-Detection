# ChangeStar zero-shot baseline

`src/run_changestar_baseline.py` is an optional, **zero-training** experiment
using the published Changen2-pretrained ChangeStar ViT-B model. It is kept
outside the ordinary project `venv` because it needs PyTorch and a 400 MB model
weight. It must not replace the established workflow until it passes a broader,
human-reviewed Victorian validation set.

## What it separates

For a fixed, matching before/after RGB GeoTIFF pair, it writes four distinct
review layers:

- `change_probability.tif`: semantic observed-change evidence;
- `before_building_probability.tif` and `after_building_probability.tif`:
  date-specific building proposals;
- `after_only_building_mask.tif`: an after-date building proposal absent from
  the before-date proposal; and
- `building_change_agreement_mask.tif`: the conservative intersection of
  change and after-only building evidence.

These are evidence layers, not authoritative cadastral or roof footprints.
Keeping them separate prevents a shadow-fragmented change map from being shown
as if it were a complete building outline.

## Isolated environment

Use a separate Python 3.11 environment. Do not add these dependencies to the
project's standard `venv`.

```powershell
py -3.11 -m venv C:\tmp\change_detection_venvs\changestar
C:\tmp\change_detection_venvs\changestar\Scripts\python.exe -m pip install `
  --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu torchvision==0.22.1+cpu
C:\tmp\change_detection_venvs\changestar\Scripts\python.exe -m pip install `
  torchange==0.0.4 albumentations==2.0.8 ever-beta==0.6.1 `
  rasterio scikit-image shapely opencv-python-headless timm huggingface-hub
```

Download the public published weight outside the repository, then record its
SHA-256 in the run report:

```powershell
curl.exe --fail --location --output C:\tmp\change_detection_models\s1c1_cstar_vitb_1x256.pth `
  https://huggingface.co/EVER-Z/Changen2-ChangeStar1x256/resolve/main/s1c1_cstar_vitb_1x256.pth
```

## Fixed baseline command

```powershell
C:\tmp\change_detection_venvs\changestar\Scripts\python.exe src\run_changestar_baseline.py `
  --before data\output\docklands_pilot_2020_2023_imagery\imagery\before_2020-04-28_Vert_tiles.tif `
  --after data\output\docklands_pilot_2020_2023_imagery\imagery\after_2023-07-06_Vert_tiles.tif `
  --weights C:\tmp\change_detection_models\s1c1_cstar_vitb_1x256.pth `
  --output data\output\docklands_pilot_2020_2023_changestar_s1_vitb `
  --device cpu
```

The frozen protocol uses 256-pixel patches and a 0.5 probability threshold.
Do not tune those settings on the two pilot cases. Runtime outputs remain under
`data/output` and are intentionally ignored by Git.

Use `src/generate_raster_grid_review.py` for its vectors. Unlike the ordinary
review utility, these GeoJSON files use the imagery's projected grid rather
than WGS84 coordinates.

```powershell
C:\tmp\change_detection_venvs\changestar\Scripts\python.exe src\generate_raster_grid_review.py `
  --before <before.tif> --after <after.tif> `
  --candidates <run-output>\building_change_agreement_candidates.geojson `
  --output <run-output>\agreement_review `
  --candidate-source "ChangeStar agreement: change and after-only building"
```

See `data/benchmarks/changestar_zero_shot_baseline.md` for the first two-case
result. It is mixed and therefore not approved for promotion.
