# Validation protocol before model training

This project will not begin a new training loop until a model family has shown
clear value on frozen Victorian evidence. The current registry is deliberately
called **pilot evidence**, not a production benchmark.

## What is locked now

`data/benchmarks/validation_manifest.json` hashes the committed review labels
and City of Melbourne reference candidates used in the work so far. Validate
them before a comparison:

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\validate_validation_manifest.py
```

This command performs no imagery download, GPU inference, model training, or
file update. It only verifies that the human-reviewed evidence has not changed
silently.

The current pilot cases are:

- `melbourne_cbd_west_2020_2023`: seven human-confirmed visible changes and
  three mapping-only differences. It supports a limited candidate-recall
  measurement, not outline IoU or complete precision.
- `murchison_estate_2022_2026`: documented failures involving shadows, missed
  buildings, pools, gardens, driveways, fragmented outlines, and merged roofs.
  It supports failure-mode review, not a complete independent numeric score.

Neither case may be used to choose a model/checkpoint/threshold and then be
reported as final performance. A new model may be explored on these cases, but
any decision must later be verified on a geographically independent hold-out
set.

## Required evidence for a real hold-out benchmark

Before fine-tuning any model, create separate Victorian areas covering a
shaded suburb, low-rise estate, industrial area, and CBD. Each area needs:

1. Fixed before and after Nearmap survey dates and a reproducible AOI.
2. Full roof outlines at both dates, kept distinct from government wall/base
   footprint geometry.
3. Permanent-change labels: `new_building`, `extension`, `demolition`, `pool`,
   `driveway`, `garden_or_soil`, `shadow`, and `vehicle`.
4. Explicit negatives, so an unmatched prediction can be measured as a false
   positive rather than merely sent for review.

Keep one complete area hidden from model, threshold, and checkpoint choice.
This is the only area used for the final go/no-go decision.

## Decision gates

1. Run frozen pretrained models once and record model version, source,
   parameters, image dates, and repository commit.
2. Reject a model family if it does not improve building detection while also
   reducing shadow/vehicle errors on independent review.
3. Only then consider one bounded GPU fine-tuning run with the training areas.
   Set the label set, time budget, and acceptance metric before the run.
4. Evaluate the chosen checkpoint once on the untouched hold-out area. Do not
   repeat training because that one result looks disappointing.

For every method, report separately: building candidate recall, change-type
confusion, shadow false positives, review workload, and after-date roof
boundary quality. A high pixel-change score alone is not building accuracy.
