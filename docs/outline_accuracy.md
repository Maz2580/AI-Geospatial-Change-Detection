# Outline accuracy

Until now this project measured whether a change was **found**, never where its
edge **sits**. The regularisation work reported edge angle error falling from
15.87° to 4.97°, which is a measure of angular consistency: an outline a metre
off the roof can still have perfectly parallel edges. This document records the
first measurement of outline position, against human-drawn roofs.

## Why boundary F1, not IoU

Intersection-over-union saturates with size. The same absolute error scores
completely differently depending on how big the building is:

| roof | 1 m outline error | IoU | verdict under an IoU ≥ 0.5 gate |
| --- | --- | --- | --- |
| 10 m² shed | 1 m too large | 0.38 | scored as a miss **and** a false alarm |
| 200 m² house | 1 m too large | 0.77 | scored as a hit |
| 6,000 m² warehouse | 1 m too large | 0.96 | scored as a hit |

Sheds, garages and carports are the structures encroachment detection exists to
catch. An IoU gate would excuse the same error on the buildings that matter least
and condemn it on the ones that matter most — reintroducing the size bias the
gold set was built to remove.

Boundary F1 asks a direct question instead: what fraction of the outline lies
within *x* metres of the true roof edge, measured symmetrically so that tracing
one wall well does not pass. On a 10 × 20 m house offset by one metre, IoU reads
0.818 while boundary F1 at 0.25 m reads 0.325.

Boundaries are sampled **by distance, not by vertex**. A regularised 12-vertex
rectangle and a 340-vertex staircase describe the same edge, and vertex sampling
would separate them for a reason unrelated to accuracy.

## Instance matching

A prediction corresponds to a label when their intersection covers at least half
of whichever is smaller. Not IoU, which turns a delineation error into a
detection error; not centroid containment, which breaks on L-shaped and
courtyard roofs whose centroid falls outside themselves, and on a blob covering
many units.

Correspondence is many-to-many on purpose, so **merges and splits survive into
the report** instead of being flattened into hit-and-miss counts. `matches`
reduces them to one-to-one pairs, best IoU first, for boundary scoring: scoring a
blob against all eleven roofs it covers would count one boundary eleven times.

## Measured baseline

DINOv3 `hotosm/dinov3s-buildings` at the published threshold 0.4371, with
simplification and dominant-orientation regularisation as currently shipped.
Scored on UC5 gold chips 909, 912 and 913 — the three with no unresolved
parent/part containment — 80 human-labelled roofs, measured in EPSG:7855.

Two configurations are recorded: what shipped before this work, and what ships
now (ground measurement corrected, regularisation off, `min_area_m2` 6.0).

### Outline position

| chip | bF1@0.25 m | bF1@0.5 m | bF1@1 m | bF1@2 m | IoU | H95 | area error | vertices pred/label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 909 *before* | 0.293 | 0.722 | 0.904 | 0.972 | 0.677 | 1.69 m | −10.7% | 17 / 15 |
| **909 now** | **0.619** | **0.804** | **0.947** | **1.000** | **0.797** | **1.00 m** | −13.8% | 16 / 14 |
| 912 *before* | 0.405 | 0.630 | 0.759 | 0.843 | 0.736 | 4.00 m | −4.1% | 36 / 22 |
| **912 now** | **0.600** | **0.709** | **0.774** | **0.852** | **0.750** | 4.03 m | −2.8% | 34 / 22 |
| 913 *before* | 0.272 | 0.484 | 0.581 | 0.874 | 0.468 | 2.90 m | +3.9% | 39 / 24 |
| **913 now** | **0.361** | 0.476 | **0.604** | 0.863 | **0.474** | 2.94 m | +5.7% | 30 / 24 |

Boundary F1 at 0.25 m roughly doubles on 909 and rises by half on 912. Chip 913
barely moves, and that is consistent: it is the townhouse chip, where the
dominant error is merging rather than edge placement.

Even so, at 0.25 m — two pixels of this imagery — **more than half the predicted
boundary is still off the roof edge** on two of three chips.

### Detection

| chip | labels | predicted | matched | missed | false alarms | splits | merges | recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 909 | 32 | 21 | 19 | 10 | 2 | 0 | 2 | 0.688 |
| 912 | 24 | 20 | 17 | 0 | 0 | 2 | 3 | 1.000 |
| 913 | 24 | 16 | 16 | 1 | 0 | 0 | 6 | 0.958 |
| **all now** | **80** | **57** | **52** | **11** | **2** | **2** | **11** | **0.863** |
| all *before* | 80 | 54 | 50 | 13 | 1 | 2 | 11 | 0.838 |

### By building size, matched pairs pooled

| size | n | bF1@0.25 m | bF1@0.5 m | IoU | H95 |
| --- | --- | --- | --- | --- | --- |
| 0–25 m² | 13 | 0.750 | 1.000 | 0.832 | 0.40 m |
| 25–100 m² | 10 | 0.371 | 0.500 | 0.689 | 1.84 m |
| 100–400 m² | 29 | 0.410 | 0.528 | 0.679 | 4.23 m |

## What the numbers say

**Merging is the dominant structural failure.** Eleven of 54 predictions cover
more than one labelled roof — 6 of 16 on chip 913, the townhouse development.
Semantic segmentation cannot separate roofs that touch, because nothing in a
per-pixel building/not-building output encodes where one building ends.

**Small structures under tree cover are missed.** Chip 909 recalls 0.625: twelve
of 32 roofs found nothing at all. That chip has the smallest buildings in the
clean set, median 24.7 m². These are the sheds and carports the tool exists to
catch.

**Area is right while the boundary is wrong.** Median area error is −10.7%,
−4.1% and +3.9%, but H95 reaches 4.44 m. The boundary wanders around the true
edge rather than sitting consistently inside or outside it, so an area check
looks healthy while the outline is unusable. Any quality gate based on area alone
would pass this.

**Outline quality degrades with building size** — H95 rises 0.55 m → 1.66 m →
4.44 m across the size bands. Larger roofs have more planes and materials, so the
probability field varies more across a single building and the threshold contour
cuts through it. This is consistent with the threshold-sensitivity measurement:
moving the threshold from 0.4371 to 0.70 shifts the boundary 0.67 m at the median
and 4.11 m at p90, so the boundary position is set by an arbitrary parameter
rather than by the image.

**Caveat: the small-building figures are survivorship.** Only matched pairs are
scored, so the 0–25 m² band describes the eleven small roofs the model found, not
the ones it missed. Read it alongside chip 909's recall, not instead of it.

## Threshold and regularisation sweep

Two settings in the same pipeline could explain the error: the probability
threshold places the boundary, and dominant-orientation regularisation then moves
it again. Inference does not depend on either, so the probability raster is
cached per chip and every variant costs seconds.

All three clean chips pooled, 80 labels, measured after the ground-measurement
fix in [ground_measurement.md](ground_measurement.md) — so `min_area_m2 = 10`
now excludes footprints under a true 10 m² rather than a nominal 6.5 m²:

| threshold | regularised | predicted | matched | missed | merges | bF1@0.25 | bF1@0.5 | IoU | H95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2500 | yes | 43 | 42 | 13 | 16 | 0.340 | 0.508 | 0.714 | 3.08 m |
| 0.3500 | yes | 46 | 43 | 16 | 12 | 0.298 | 0.554 | 0.722 | 3.31 m |
| 0.4371 | yes | 48 | 44 | 19 | 11 | 0.324 | 0.577 | 0.683 | 3.04 m |
| 0.5500 | yes | 51 | 45 | 22 | 9 | 0.298 | 0.520 | 0.688 | 3.55 m |
| 0.7000 | yes | 50 | 41 | 30 | 8 | 0.278 | 0.473 | 0.559 | 4.56 m |
| 0.2500 | **no** | 45 | 43 | 12 | 16 | 0.479 | 0.616 | **0.755** | 2.93 m |
| **0.3500** | **no** | 46 | 43 | 16 | 12 | **0.495** | **0.655** | 0.747 | 3.30 m |
| 0.4371 | **no** | 50 | 46 | 17 | 11 | 0.457 | 0.626 | 0.704 | 2.94 m |
| 0.5500 | **no** | 51 | 45 | 22 | 9 | 0.421 | 0.594 | 0.696 | 3.70 m |
| 0.7000 | **no** | 50 | 41 | 31 | 8 | 0.347 | 0.518 | 0.567 | 4.59 m |

### Regularisation makes the outlines worse

At every threshold tested, switching regularisation off improves boundary
accuracy. At the published threshold, boundary F1 at 0.25 m rises from 0.324 to
0.457 — a 41% relative gain — while IoU rises 0.683 → 0.704 and misses fall
19 → 17.

**Regularisation is therefore off by default.** `RegularisationConfig.enabled`
is `False`, `--regularise` turns it back on, and a test pins the default so it
cannot drift back without a decision. The two recorded experiments that were
measured with it enabled now say so explicitly.

The published threshold 0.4371 is *not* changed. Threshold 0.35 scores slightly
better here, but on 80 roofs from three chips that margin does not justify
departing from the model card.

This does not contradict the earlier regularisation measurement; it reinterprets
it. Edge angle error genuinely fell from 15.87° to 4.97°, but that measures
angular *consistency*, not position. The regulariser assumes its input boundary
is roughly correct and merely ragged, so it estimates a dominant orientation and
snaps edges to it. When the input is a threshold contour that is already
systematically wrong, the estimated orientation is drawn from the error, and
snapping propagates it along edges that had been closer to correct. The output
looks more like a building and sits further from one.

Best configuration measured: **threshold 0.35, regularisation off** — boundary F1
at 0.25 m of 0.511 against 0.340 shipped, IoU 0.757 against 0.683, ten misses
against thirteen.

### No threshold buys both recall and separated buildings

Misses and merges move in opposite directions across the whole range, and the
trade is monotonic: at 0.25 only 8 roofs are missed but 16 predictions each cover
several buildings; at 0.70 merges fall to 8 while misses rise to 26. There is no
setting that delivers both.

That is the signature of an architectural limit rather than a tuning problem. A
semantic segmenter emits building/not-building per pixel, and two townhouses
sharing a wall are an unbroken run of building pixels — no threshold on that
raster can insert a boundary that the raster does not encode. Separating them
needs instance segmentation, or a boundary channel, or an edge-aware step that
reads the image rather than the probability.

### The minimum-area filter removes real buildings

Misses are concentrated in small structures. Of the 17 missed at the published
threshold, 14 are under 20 m², and the median missed roof is 12.3 m² against a
median labelled roof of 80.5 m².

Lowering `min_area_m2` shows a clear knee:

| min_area_m2 | predicted | matched | missed | missed under 20 m² |
| --- | --- | --- | --- | --- |
| 10.0 | 50 | 46 | 17 | 14 |
| **6.0** | **57** | **52** | **11** | **8** |
| 4.0 | 60 | 52 | 11 | 8 |

Ten to six recovers six real buildings for one extra unmatched prediction. Six to
four adds three predictions and no matches at all — pure false alarms.

The mechanism is not that 6 m² buildings exist in quantity: only 8 of the 80
labels are under 10 m². It is that the segmenter **under-segments** small roofs,
so a real 25 m² shed arrives as an 8 m² blob and the 10 m² filter deletes it. A
detector filtering its own noisy output needs headroom below the size it is
meant to find.

Sheds, garages and carports are what encroachment detection exists to catch, so
the default is **6.0 m²**. Raise it with `--min-area-m2` where false alarms cost
more than misses.

## Why unstable boundaries break extension detection

An extension candidate is the after-date footprint minus the before-date one. That
subtraction inherits the boundary error of *both* dates, so it is roughly twice as
noisy as either.

Measured on the Murchison north-west holdout, 48 shipped candidates:

| | shipped | corrected |
| --- | --- | --- |
| candidates | 48 | 21 |
| new building | 14 | 3 |
| extension | 34 | 18 |
| total reported change area | 4,841 m² | 1,662 m² |

Three separate corrections, each verified in isolation:

1. **Ground metres.** Restoring the old effective thresholds in the corrected CRS
   reproduces the shipped 48/14/34 exactly, which proves the CRS change altered
   no behaviour. The counts then fall because `min_area_m2 = 10` finally filters
   at a true 10 m² instead of 6.5 m².
2. **The change, not the object.** Candidate 1 was reported as a 695.8 m²
   "extension"; it is a 160 m² change on a 450 m² roof.
3. **Sliver rejection.** Differencing two independent segmentations of one
   unchanged roof leaves a thin ribbon around the perimeter. Eroding by half of
   `min_change_width_m` removes it while a real wing survives.

**What is still wrong.** Sliver rejection removes ribbons, not blobs. On this
holdout, houses that did not change still produce 78–109 m² "extensions", because
the two dates disagree about the roof edge by more than a metre in places — not
by a few pixels. That is the same instability measured above: moving the
threshold from 0.4371 to 0.70 shifts the boundary 0.67 m at the median and 4.11 m
at p90.

The consequence is structural, and it sets the work order:

- **New-building candidates are the trustworthy signal today.** They require no
  before-date footprint at all, so they inherit one boundary's error rather than
  the difference of two.
- **Extension detection cannot be made reliable by filtering.** Every filter
  tried so far trades real extensions against false ones, because the two
  populations overlap in size and shape. It needs a boundary that is stable
  between dates, which means placing the boundary from the imagery rather than
  from a threshold on a soft probability field.

For an enforcement pathway this distinction matters: "a building appeared where
there was none" is currently far better supported by the evidence than "this
building grew".

## Placing the boundary from the image

If the probability contour cannot say where the edge is, the imagery can: a roof
edge is a sharp radiometric discontinuity at 12 cm, and it is in the same place
on every date.

The probability field is therefore used only for what it is reliable at — saying
a building is *here* — and the edge is placed by segmenting the image, seeded
from the confident core and the confident background. Refinement runs per
building, because a local colour model separates one roof from its own
surroundings far better than a scene-wide one that would have to describe every
roof material and every garden at once.

Same three chips, 80 labels, threshold 0.35, regularisation off:

| variant | pred | matched | missed | merges | bF1@0.25 | bF1@0.5 | IoU | H95 | vertices |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| threshold only | 54 | 50 | 10 | 12 | 0.511 | 0.659 | 0.757 | 2.81 | 28 |
| core 0.75 / bg 0.10 | 55 | 50 | 12 | 12 | 0.617 | 0.701 | 0.779 | 2.94 | 21 |
| core 0.85 / bg 0.05 | 54 | 50 | 11 | 12 | 0.617 | 0.721 | 0.762 | 2.88 | 25 |
| **core 0.60 / bg 0.20** | 55 | 50 | **9** | 13 | **0.614** | 0.681 | 0.773 | **2.74** | 25 |
| wider context 8 m | 56 | 49 | 13 | 11 | **0.635** | **0.754** | **0.809** | 2.73 | 23 |

**Adopted: core 0.60 / background 0.20.** Boundary F1 at 0.25 m rises from 0.511
to 0.614, a 20% relative gain, and it is the only variant that also *lowers*
misses, 10 to 9. The 8 m context scores higher on every boundary measure but
misses three more buildings, which is the wrong trade for a screening tool: a
missed unpermitted structure is not recoverable downstream, a slightly worse
outline is.

Predicted vertices also fall from 28 towards the human median of 22–24. An
outline that follows a real edge needs fewer vertices than one tracing a
threshold contour through noise.

### A measurement that nearly went the wrong way

The first run of this sweep showed refinement gaining only 0.033 on boundary F1
while losing detections — enough to reject it. The diagnostic that caught the
error was area moved: refinement *added* 699 m² and removed **exactly zero**, in
every variant. A method that only ever grows a boundary is not choosing one.

The cause was in the refinement code, not the model: the threshold mask was
retained and the refined result unioned on top, so the two could only add. Once a
refined core *replaced* its threshold blob rather than supplementing it, the same
sweep moved from +0.033 to +0.103 and boundaries began moving in both directions
— 421 m² added, 759 m² removed.

Worth keeping as a habit: when a promising method reports a suspiciously small
effect, check that it is able to act in both directions before concluding it does
not work.

## Reproducing

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\evaluate_outline_goldset.py
.\venv\Scripts\python.exe src\experiments\goldset_outline_sweep.py
.\venv\Scripts\python.exe src\experiments\boundary_refinement_sweep.py
```

Compare a variant by passing `--threshold`, `--no-regularise`, or `--min-area-m2`
with a distinct `--label` and `--output`.
