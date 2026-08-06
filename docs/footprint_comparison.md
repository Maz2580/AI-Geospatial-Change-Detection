# Building-footprint comparison

Pixel and semantic change proposals say that an area changed; they do not
reliably define the building that now occupies it. This stage compares building
footprint proposals extracted separately at the before and after dates.

It produces review-only evidence:

- `new_building_footprint_candidate`: an after-date footprint has no close or
  overlapping before-date footprint.
- `building_extension_footprint_candidate`: an after-date footprint overlaps a
  before footprint but has a material area outside it.

These are not confirmed buildings. Footprint models may merge nearby homes,
miss a roof, or shift an outline. They must **not** be counted as an independent
vote in `fuse_change_candidates.py`; use them to refine and investigate an
existing site candidate.

## Extract two footprint collections

Use a small AOI and run the optional UMAMI command once per date. The duplicated
date is intentional: `extract_footprints` is an absolute-object operation, not
a change operation.

```powershell
.\venv\Scripts\python.exe src\run_building_change.py umami `
  --west 145.40265 --south -36.33742 --east 145.40400 --north -36.33650 `
  --before-date 2022-07-30 --after-date 2022-07-30 `
  --detector segformer --mode extract_footprints --long-side 768 `
  --output data\output\estate\footprints_before

.\venv\Scripts\python.exe src\run_building_change.py umami `
  --west 145.40265 --south -36.33742 --east 145.40400 --north -36.33650 `
  --before-date 2026-04-18 --after-date 2026-04-18 `
  --detector segformer --mode extract_footprints --long-side 768 `
  --output data\output\estate\footprints_after
```

For this mode use `*_all_candidates.geojson`: the remote service does not set
its change-specific `relevant` flag on footprints.

## Compare the objects

```powershell
.\venv\Scripts\python.exe src\compare_building_footprints.py `
  --before data\output\estate\footprints_before\umami_segformer_extract_footprints_all_candidates.geojson `
  --after data\output\estate\footprints_after\umami_segformer_extract_footprints_all_candidates.geojson `
  --match-distance-m 6 --match-iou 0.10 --extension-outside-fraction 0.25 `
  --output data\output\estate\footprint_comparison
```

The 6 m distance absorbs segmentation and registration movement. Do not lower
it merely to increase results: an extension beside an older home can be close
to its existing footprint and needs review.
