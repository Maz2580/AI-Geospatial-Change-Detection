# Official building-footprint benchmark

The DINO footprint experiment failed the Murchison estate review: it detected
shadows and unrelated areas while missing real buildings. It is therefore not
an accepted candidate source and must not be fused into operational results.

The next step is to evaluate detectors against independent Australian reference
data. City of Melbourne publishes dated Building Footprint snapshots (including
2020 and 2023) under CC BY 4.0. This allows us to construct a compact test set
with a before/after aerial pair and reference building polygons.

The reference data is not automatic construction truth:

- it maps the wall/base footprint while aerial imagery generally shows roofs;
- City footprints can contain stacked shapes for a single building's different
  built-form levels;
- a snapshot difference can be a mapping improvement, an extension, or a new
  building. It must be checked on the aerial pair before it becomes a label.

## Download a compact dated benchmark

The initial exploration identified a dense western-CBD cell. Its 2020 and 2023
snapshots contain 39 and 70 reference footprints respectively. This is an
evaluation area only, not a claim that all differences are new construction.

```powershell
.\venv\Scripts\python.exe src\download_city_footprint_benchmark.py `
  --west 144.951 --south -37.811 --east 144.953 --north -37.809 `
  --before-year 2020 --after-year 2023 `
  --output data\benchmarks\melbourne_cbd_west_2020_2023\reference
```

The output stores both source snapshots, licence/source metadata, and
`footprint_change_candidates.geojson`. Download matching Nearmap dates for the
same small AOI, then render those reference candidates in the existing review:

```powershell
.\venv\Scripts\python.exe src\generate_change_review.py `
  --before data\output\melbourne_cbd\imagery\before_DATE_Vert_tiles.tif `
  --after data\output\melbourne_cbd\imagery\after_DATE_Vert_tiles.tif `
  --candidates data\benchmarks\melbourne_cbd_west_2020_2023\reference\footprint_change_candidates.geojson `
  --candidate-source "City of Melbourne dated footprint comparison (review prompt only)" `
  --output data\benchmarks\melbourne_cbd_west_2020_2023\reference_review
```

Mark each card as `new building`, `extension`, `mapping-only`, or `not
visible`. Only the first two become benchmark labels. The resulting labelled
examples can then measure detector precision, recall, and outline overlap. We
will not tune or accept another model against the estate alone.

## Operational role after validation

Vicmap's statewide Building Polygon layer is a useful independent current
reference for larger Victorian buildings, but it has no guaranteed image-date
match and cannot represent newly completed buildings that have not yet entered
the map. It may support a future `mapped-building evidence` flag; it must never
silently suppress a real unmapped new building or confirm one by itself.
