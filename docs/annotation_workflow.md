# Independent benchmark annotation workflow

This workflow creates labels for a fixed Nearmap pair **without using a
detector prediction as truth**. An official dated footprint difference may
suggest a review location, but it is not copied into a roof label because map
footprints and visible roof edges can differ.

## 1. Freeze the inputs

For each new case, record its AOI and the *resolved* Nearmap survey dates. The
Docklands pilot uses 28 April 2020 and 6 July 2023; its reference is a single
City of Melbourne 2020-to-2023 footprint difference.

## 2. Create the draft

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\scaffold_reference_labels.py `
  --case-id docklands_pilot_2020_2023 `
  --reference data\benchmarks\docklands_pilot_2020_2023\reference\footprint_change_candidates.geojson `
  --before-date 2020-04-28 --after-date 2023-07-06 `
  --output data\labels\docklands_pilot_2020_2023_reference_review_labels.draft.json
```

The output deliberately marks every reference candidate `unreviewed` and
contains no roof geometry. It is safe to create and inspect, but is not part of
any benchmark score yet.

## 3. Human review

For each reference candidate, set exactly one assessment:

- `real_visible_change` plus a permanent `visible_change_type`, for example
  `new_building` or `building_extension`;
- `mapping_only_or_not_visible` with `visible_change_type: not_applicable`;
- `inconclusive_due_occlusion` with `visible_change_type: not_applicable` and
  a review note. Use this when vegetation, cloud, shadow, poor quality, or
  another occlusion prevents a reliable decision. It is neither a positive nor
  a negative label and is excluded from scoring.

For a real building change, manually digitise the complete after-date **roof**
into `after_roof_labels`. Use only `human_review` as its label source. Do not
trace the reference map polygon or a model proposal.

## 4. Validate before promotion

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\scaffold_reference_labels.py `
  --validate data\labels\docklands_pilot_2020_2023_reference_review_labels.draft.json `
  --require-complete
```

Only a completed, reviewed document can be frozen into a later benchmark
manifest revision. One pilot case is not enough for model selection; it is
simply the first independent example.
