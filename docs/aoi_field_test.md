# Field test on unseen Victorian imagery

The gold set measures outline geometry on single-date chips. It cannot say
whether the change pipeline behaves on ground it has never seen. This records a
run on four fresh Nearmap AOIs, 2021 to 2026, none of which overlaps anything
used for tuning: the gold set covers Box Hill, Altona North, Sunbury and
Derrimut, and the pipeline was developed on Murchison and inner Melbourne.

Each AOI is a 200 m square at zoom 21, roughly 6 cm ground resolution. Tarneit
was pinned to zoom 21 because its native zoom needs 784 tiles against a 256-tile
cap; without pinning, its figures would not be comparable with the rest.

## Results

| AOI | dates | buildings before | after | candidates | new | extension | change m² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clyde North | 2021-12-23 → 2026-08-01 | 6 | 4 | 2 | 0 | 2 | 106 |
| Kalkallo | 2021-12-17 → 2026-04-25 | 1 | 1 | 0 | 0 | 0 | 0 |
| Tarneit | 2021-12-01 → 2026-08-02 | 31 | 34 | 13 | 5 | 8 | 1,706 |
| Camberwell | 2021-12-24 → 2026-04-25 | 107 | 91 | 65 | 5 | 60 | 5,036 |

## Two mistakes in the test design, recorded so they are not repeated

**Two AOIs were chosen from suburb names without checking land use.** Clyde North
landed on market gardens and Kalkallo on bare paddock. Their near-zero counts are
correct — there was nothing to find — but they measure nothing. Coordinates for a
growth corridor must be scouted from a coarse overview first, which costs a
fraction of a full-resolution AOI.

**Camberwell was chosen as a stable control and is not one.** The intent was an
established suburb where almost nothing had been built, so that its candidate
count would estimate the false-positive floor. The imagery shows heavy
knockdown-rebuild activity across the AOI. Its 65 candidates are therefore *not*
65 false alarms, and reporting them as a noise floor — which the count alone
invites — would have been wrong.

A count is not a result until the imagery behind it has been looked at.

## What the imagery shows

Judged by eye on the six largest extension candidates, five are genuine rebuilds
or completions and one is segmentation noise on an unchanged roof. That is an
impression from six crops, not a precision figure; it needs labels before it is
quotable.

The clearest single case is candidate 10, 280 m²: the 2021 image shows a house as
bare timber framing and the 2026 image shows it finished with solar panels. A
structure part-built at the earlier date and completed by the later one is
exactly the case the tool exists to catch, and it was caught on imagery the
pipeline has never seen.

Candidate 1, a 442 m² change on a 560 m² roof, reads correctly as a near-total
rebuild. Before the change-geometry fix the same site would have been reported as
a 560 m² extension, which describes a house rather than what was done to it.

## Where the candidates are trustworthy

Changed area separates the two populations usefully. Median candidate is 53 m²,
p25 is 24 m², and 33 of 65 are 50 m² or larger. The large candidates are
predominantly real; the small tail is where segmentation disagreement collects.
Sorting a review list by `changed_area_m2` is therefore worthwhile, which it was
not before that field described the change rather than the whole building.

## The remaining fault, quantified

The detector found **107 buildings in 2021 and 91 in 2026** in an area where
nothing was demolished. Sixteen buildings disappear between dates purely because
each date is segmented independently.

That is the same instability measured against the gold set — the boundary moves
0.67 m at the median for an arbitrary threshold change — and it feeds directly
into `after − before`. It is why extension candidates carry more noise than
new-building candidates, and why per-date detection stability, not filtering, is
the next thing worth fixing.

## Reproducing

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\run_building_change.py nearmap `
  --longitude 145.0580 --latitude -37.8420 --radius-m 100 `
  --before-date 2021-12-24 --after-date 2026-04-25 `
  --output data\output\aoi_camberwell

.\venv\Scripts\python.exe src\run_dino_buildings.py `
  --before data\output\aoi_camberwell\imagery\before_2021-12-24_Vert_tiles.tif `
  --after  data\output\aoi_camberwell\imagery\after_2026-04-25_Vert_tiles.tif `
  --before-capture-date 2021-12-24 --after-capture-date 2026-04-25 `
  --output data\output\aoi_camberwell_dino
```

Imagery is licensed and gitignored; the commands above re-fetch it.
