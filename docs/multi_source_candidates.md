# Multi-source construction candidates

This workflow treats every detector as a candidate generator, not as ground truth. A new building can be retained if one source detects it; agreement from independent sources only changes its review priority.

## 1. Run the local Nearmap workflow

Use the existing `nearmap` command for the repeatable RGB/DSM comparison. Its construction candidate GeoJSON is one input to fusion.

## 2. Request remote D-FINE or SegFormer candidates

The optional `umami` command connects directly to a compatible Encroachment API. Add these values to the ignored local `.env` file; never commit real credentials.

```text
UMAMI_BASE_URL=https://geollm.idigitaltwin.org/TEST-UMAMI
UMAMI_USERNAME=
UMAMI_PASSWORD=
```

Run a small AOI (roughly one estate block, not an entire suburb) for each model:

```powershell
.\venv\Scripts\python.exe src\run_building_change.py umami `
  --west 145.40265 --south -36.33742 --east 145.40400 --north -36.33650 `
  --before-date 2022-07-30 --after-date 2026-04-18 `
  --detector dfine --mode change `
  --output data\output\estate\umami_dfine

.\venv\Scripts\python.exe src\run_building_change.py umami `
  --west 145.40265 --south -36.33742 --east 145.40400 --north -36.33650 `
  --before-date 2022-07-30 --after-date 2026-04-18 `
  --detector segformer --mode change `
  --output data\output\estate\umami_segformer
```

Each run writes both `*_all_candidates.geojson` and `*_relevant_candidates.geojson`. The latter matches the service's own `relevant=true` filter, but it can still miss a real building. It is useful as an input, not a decision.

## 3. Fuse and review

Fuse the local candidate file with the remote filtered candidate files. Inputs must use WGS84 GeoJSON.

```powershell
.\venv\Scripts\python.exe src\fuse_change_candidates.py `
  --candidate pixel=data\output\estate\construction_change_candidates.geojson `
  --candidate dfine=data\output\estate\umami_dfine\umami_dfine_change_relevant_candidates.geojson `
  --candidate segformer=data\output\estate\umami_segformer\umami_segformer_change_relevant_candidates.geojson `
  --merge-distance-m 2 `
  --output data\output\estate\fusion
```

`fused_candidates.geojson` stores `candidate_sources`, `source_count`, source classifications, and `agreement`:

- `multi_source_agreement`: spatial agreement between two or more independent sources. Review first.
- `single_source_candidate`: detected by only one source. Still review it, particularly near new estate construction, pools, sheds, and driveway extensions.

Generate a normal before/after review report using `fusion\fused_candidates.geojson`. The report now supports both polygon and multipolygon fused footprints.

## What this does not claim

Neither D-FINE nor SegFormer, and neither the local RGB detector, can confirm construction on its own. Shadows, vehicles, vegetation, imagery misregistration, and capture-date differences still need human QA. The next quality step is an Australian labelled evaluation set with known new buildings, extensions, pools, driveways, shadows, and vehicles; use that to choose thresholds and a final ranking model.
