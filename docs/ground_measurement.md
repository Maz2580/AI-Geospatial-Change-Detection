# Ground measurement

Every area and distance this project reported from Nearmap tile imagery was
inflated. This records what was wrong, how large it was, and what changed as a
result.

## The fault

`_area_m2` returned a geometry's area unchanged whenever its CRS satisfied
`crs.is_projected`:

```python
if source_crs and source_crs.is_projected:
    return float(geometry.area)
```

EPSG:3857 satisfies that test and is not a ground-metre CRS. Web Mercator
preserves angles by stretching distance away from the equator, so at latitude φ
a map metre is `cos(φ)` ground metres:

| | Melbourne (−37.81°) | Murchison (−36.34°) |
| --- | --- | --- |
| distance inflation, sec φ | 1.266× | 1.241× |
| area inflation, sec² φ | 1.602× | 1.541× |

Every Nearmap tile mosaic is EPSG:3857 — verified by opening the rasters — and
`footprints.py`, `corroboration.py`, `sites.py`, `fusion.py` and
`reference_evaluation.py` each transformed *into* EPSG:3857 before measuring.

The downloader already knew. `nearmap.py` converts a requested ground radius into
map metres before choosing tiles, and says so in a comment. The knowledge existed
and never reached the measurement code.

Runs on EPSG:7855 imagery — the Murchison estate pair behind
`pipeline_measurements.json` — were always correct. Ratios such as "16.6% of the
scene is building" were also always correct, because the inflation cancels.
Absolute areas and distances from any tile-based run were not.

## The fix

`building_change/geodesy.py` is now the only place that knows the difference.

- **Areas** are integrated on the WGS84 ellipsoid via `pyproj.Geod`, so they are
  exact and need no zone choice near a zone boundary.
- **Distances, intersections and unions** need a plane, so they use the local UTM
  zone chosen from the geometry itself. Within one area of interest the scale
  factor is effectively constant, leaving at most about 0.2% residual against the
  60% error it replaces.
- One CRS is chosen per comparison, not per feature, so two footprints either
  side of a zone boundary are still measured against each other in one plane.

Verified on a Melbourne block of known size: 10,555.9 m² true, 16,939.9 m² as
Web Mercator reported it.

## What changed in the output

Re-running the Murchison north-west holdout footprint comparison, with the old
effective thresholds restored, reproduces the shipped 48 candidates / 14 new /
34 extension exactly — so nothing but the units moved.

Areas fall by the predicted factor. The largest candidate was reported as
695.8 m² and is 450.5 m², a ratio of 1.545 against the predicted 1.541.

The thresholds themselves were never rescaled, so their **effective** ground
values changed when the units were fixed:

| match distance | min area | candidates | new | |
| --- | --- | --- | --- | --- |
| 4.83 m | 6.49 m² | 48 | 14 | what the shipped run actually did |
| 6.00 m | 6.49 m² | 48 | 11 | distance corrected only |
| 4.83 m | 10.00 m² | 35 | 7 | area corrected only |
| 6.00 m | 10.00 m² | 35 | 5 | both, at the values the config states |

Two separate effects:

**Thirteen candidates disappear** because they measured between 6.49 m² and
10 m² of true ground. The configuration asked for a 10 m² floor and was silently
filtering at 6.49 m². The filter is now honest. If small sheds at that size
matter — and for encroachment they do — lower `min_area_m2` deliberately rather
than relying on an accident.

**Nine of fourteen new-building candidates become extensions** because the
matching radius is now a real 6 m instead of 4.83 m, so more after-date
footprints find a before-date counterpart. This moves in the direction earlier
work wanted: commit 3532188 reasoned that "a spurious 2021 footprint only
suppresses one candidate while a missed one invents one", and a wider matching
radius suppresses the same phantoms.

That reasoning is not evidence. **These thresholds were chosen while the units
were wrong, so their tuning cannot be assumed to carry over.** They need
re-checking against labelled data before any of these counts is quoted as a
result.

## Reproducing

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe -m unittest tests.test_geodesy
```
