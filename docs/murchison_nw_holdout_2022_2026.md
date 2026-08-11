# Murchison north-west holdout: 2022 to 2026

This is a small independent check of the existing experimental channels. It is
not a labelled benchmark and must not be reported as precision or recall.
Its purpose is to prevent the Murchison estate tuning area from being reused as
evidence that an approach generalises.

## Frozen input

| Item | Value |
| --- | --- |
| centre | 145.404000, -36.334900 |
| radius | 100 m |
| before survey | 2022-02-02, Vert tiles, zoom 21 |
| after survey | 2026-04-18, Vert tiles, zoom 21 |
| relation to previous estate centre | approximately 320 m north-west; non-overlapping 100 m circles |

The imagery is deliberately stored under the ignored `data/output` directory.
This keeps the evaluation reproducible for a user with Nearmap access without
committing licensed imagery to Git.

## Results

| Channel | Result | Interpretation |
| --- | --- | --- |
| measured high-recall pixel detector | 46 candidates; 35 tagged likely-building | high coverage evidence, but its mask is not a reliable roof outline |
| DINOv3 building footprints | 71 before and 87 after footprints; 48 change proposals (14 new, 34 extension) | often follows a true new roof well, but date-to-date segmentation variation produces many false extension proposals |
| pixel support for DINO proposals | 15 weakly supported; 33 footprint-only | pixel absence must not silently discard a DINO proposal, but pixel support does not validate a building either |

Visual review of candidates 1--3 found unchanged homes incorrectly proposed as
extensions. Candidates 4, 11 and 16 are clear new structures with useful roof
outlines. Candidate 26 is a road/footpath proposal, not a building. Therefore
the DINO footprint comparator remains experimental and is **not** an accepted
primary candidate source.

## NVIDIA visual-review diagnostic

Seven deliberately mixed examples were sent as one labelled before/after image
per request: 1, 2, 3, 4, 11, 16 and 26. The reviewer correctly classified the
three false extension proposals and the road as non-building categories, but
only called one of the three clear new roofs `new_building`; another was
`unclear` and one was labelled `hardscape`.

This confirms the hosted vision model is useful as advisory ranking and an
explanation for a human reviewer. It is not safe to automatically accept or
reject a building-change candidate.

## Reproduction

```powershell
.\venv\Scripts\python.exe src\run_building_change.py nearmap `
  --longitude 145.404000 --latitude -36.334900 --radius-m 100 `
  --before-date 2022-02-02 --after-date 2026-04-01 `
  --source tiles --tile-zoom 21 --preset high-recall `
  --output data\output\murchison_estate_holdout_nw_2022_2026_pixel_high_recall

.\venv\Scripts\python.exe src\run_dino_buildings.py `
  --before data\output\murchison_estate_holdout_nw_2022_2026_pixel_high_recall\imagery\before_2022-02-02_Vert_tiles.tif `
  --after data\output\murchison_estate_holdout_nw_2022_2026_pixel_high_recall\imagery\after_2026-04-18_Vert_tiles.tif `
  --before-capture-date 2022-02-02 --after-capture-date 2026-04-18 `
  --output data\output\murchison_estate_holdout_nw_2022_2026_dino_footprints
```

Use `data/output/murchison_estate_holdout_nw_2022_2026_dino_footprints/review/index.html`
for the local visual record. It is not committed because it contains Nearmap
imagery.
