# UC5 gold set v2 — snapshot copy

Human-drawn building roof outlines, used here as the **outline accuracy ruler**.
This is a read-only snapshot. The authoritative copy lives with the UC5
encroachment project:

```
S:\CSDILA 2026\change detection sole panel glm\Arcgispro_project
  \v2-20260809T014246Z-1-001\v2\uc5_goldset_v2\uc5_goldset_v2
```

Snapshot taken 2026-08-13, covering the nine chips labelled at that date:
900, 908, 909, 910, 911, 912, 913, 914, 915. Never edit the files here; edit the
authoritative copy in ArcGIS Pro and re-snapshot.

## Evaluation only

These labels must never train, tune, fine-tune, seed, or select a detector.
A model fitted to them stops being measurable by them, and this project has
already been burned once by scoring a model against its own training labels
(SegFormer against Microsoft/OSM footprints). Scoring only.

The same rule bars auto-snapping, auto-simplifying, or auto-regularising the
label geometry: their value is that they are the one unbiased geometry available.

## What the labels are

Drawn to the UC5 protocol (`INSTRUCTIONS.md`, decided 2026-07-29):

- anything with a roof from about 10 m²: houses, detached garages, sheds,
  carports, patio roofs, warehouses
- **not** pools, tanks, solar panels, uncovered decks, vehicles, trees, fences
- one outline per *connected* roof; a detached garage is its own polygon
- the roof as seen from above, not an estimate of the walls beneath
- `edge = 1` when the chip border cuts the building, `unsure = 1` when doubtful;
  both are excluded from the strict score

Imagery is Nearmap at 0.118 m ground sample distance, roughly 96 m × 96 m per
chip, stored in EPSG:3857.

## Measurement CRS

Scoring reprojects to **EPSG:7855** (GDA2020 / MGA zone 55). EPSG:3857 is a
projected CRS whose units are not ground metres: at Melbourne's latitude it
inflates area by 1.602×. Measuring in place would overstate every building by
60%.

## Chip status at snapshot

Six chips still hold unresolved SAM 3 parent/part containment, where one roof is
present both as a whole and as its sections. A chip in that state scores a
correct single detection as one hit plus several misses, so it is excluded.

| chip | shapes | containment clusters | usable |
| --- | --- | --- | --- |
| 909 | 32 | 0 | yes |
| 912 | 24 | 0 | yes |
| 913 | 24 | 0 | yes |
| 910 | 26 | 2 | not yet |
| 911 | 29 | 2 | not yet |
| 908 | 29 | 4 | not yet |
| 914 | 12 | 1 (11 units inside one 6,150 m² outline) | not yet |
| 915 | 14 | 2 | not yet |
| 900 | 35 | 3 (one holds 15 shapes) | not yet |

Chips 914 and 915 additionally need a rule decision that the protocol leaves
ambiguous: rule 2 says a connected roof is one polygon, while the chip exists to
test *several distinct units*. Whichever is chosen changes whether a model that
finds eleven units scores eleven hits or one hit and ten false alarms.

## Reproducing the baseline

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\evaluate_outline_goldset.py
```

Writes `outline_baseline.json`. See [docs/outline_accuracy.md](../../../docs/outline_accuracy.md).
