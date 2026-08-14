# Agent brief — UC5 gold evaluation set

You are helping on a laptop with **ArcGIS Pro** (and possibly QGIS). The person you
are helping is hand-labelling 15 aerial image chips. This file tells you what the job
is, the one thing you must not do, and the things you genuinely can do to help.

---

## ⛔ The one rule that matters

**Do not label the buildings for them. Do not run any model, tool, or script that
creates, deletes, or reshapes building polygons in bulk.**

This is not a productivity task where automation helps. These 15 chips are being
labelled *specifically because* every automated label source we have is biased, and
their human judgement is the only unbiased thing in the pipeline. If you generate the
labels, the resulting scoreboard measures whatever model you used, and months of work
built on top of it will be wrong in a way nobody can detect later.

Concretely, **never**:

- run a building-detection or segmentation model to fill in or "improve" polygons
- bulk-delete polygons by size, shape, or score
- auto-snap, auto-simplify, or auto-regularise the geometry
- use *Eliminate*, *Aggregate Polygons*, or similar generalisation tools on the labels
- "finish the remaining chips" while they are away

If they ask you to do any of the above, say what it would cost and confirm before
proceeding. Fixing one polygon they point at is fine. Reshaping all of them is not.

---

## What this is and why it exists

**Project:** UC5 encroachment / change detection — finding new or unauthorised
structures by comparing aerial imagery over time. Imagery is Nearmap at ~15 cm/pixel.

**The problem being fixed.** Every accuracy number in this project so far was measured
against Microsoft GlobalML ∪ OpenStreetMap building footprints. Two things were wrong
with that:

1. Those were also the training labels for one of the models being compared, so it was
   marked using its own answer sheet.
2. Microsoft/OSM omit most sheds, garages and carports — which are exactly the
   structures encroachment detection exists to catch. Measured: 29% of one model's
   scored "false positives" were real buildings the reference simply never mapped.

The result was a scoreboard that consistently disagreed with what the user saw on
screen. They were right and the metric was wrong, twice.

**This gold set is the fix**: 15 chips in suburbs no model has trained on, labelled to
the user's own definition of a building, used only for evaluation and never for
training.

---

## The labelling rules they agreed to

Use these when answering their questions, so their decisions stay consistent.

**Draw anything with a roof, about 10 m² and up** — houses, detached garages, sheds,
carports, patio/pergola roofs, warehouses.
**Do not draw** pools, water tanks, solar panels, uncovered decks or paving, vehicles,
trees, fences, walls.

**One outline per connected roof.** House + attached garage + rear extension = one
polygon. A garage standing apart from the house = its own separate polygon. If roofs
touch, one shape; if there is open ground between them, two shapes.

**Terraces/townhouses:** draw them separately if the dividing line is visible;
otherwise draw the row as one and set `unsure = 1`.

**Trace the roof as seen from above**, not an estimate of the walls. Follow real roof
shapes — L, T and cross-gable outlines, not bounding rectangles.

**Edge of chip:** draw the visible part and set `edge = 1`.

**Unsure:** set `unsure = 1` and move on. A doubtful case costs nothing because it is
excluded from strict scoring; a confidently wrong label corrupts every future
comparison. Encourage them to use this liberally.

### Fields

| field | values |
|---|---|
| `kind` | `house`, `garage_shed`, `carport`, `industrial`, `other` |
| `unsure` | 0 / 1 |
| `edge` | 0 / 1 |
| `notes` | free text |
| `sam_score` | SAM 3's confidence — **read only, leave it alone** |

---

## What's in the folder

```
chip_9xx.tif      georeferenced aerial image, EPSG:3857
labels_9xx.shp    proposed outlines to correct  (15 chips)
goldset.gpkg      all labels in one GeoPackage, same content
INSTRUCTIONS.md   the human-facing guide
```

Chip IDs: 900–903, 905–915. (904 was dropped; 914/915 are industrial estates.)

The proposals came from **Meta's SAM 3** prompted with the word "building" — chosen
because its errors are unrelated to our own models' errors. Expect it to be decent on
obvious houses and unreliable on: small sheds in shadow (missed), carports
(inconsistent), and large warehouse roofs (sometimes split into panels).

---

## How you CAN help

### 1. Set up the map

```python
import arcpy, os
aprx = arcpy.mp.ArcGISProject("CURRENT")
m = aprx.listMaps()[0]
folder = r"C:\path\to\uc5_goldset"
cid = 900
m.addDataFromPath(os.path.join(folder, f"chip_{cid}.tif"))
m.addDataFromPath(os.path.join(folder, f"labels_{cid}.shp"))
```

Then set the label layer to a **hollow fill with a bright 2 pt outline** so the imagery
stays visible underneath. This matters — a solid fill hides the roof they are tracing.

### 2. Bulk-set `kind` on a selection they have made themselves

Safe, because they chose the features:

```python
arcpy.management.CalculateField(
    "labels_900", "kind", "'garage_shed'", "PYTHON3")   # applies to current selection
```

### 3. Run QA checks (do this, it's valuable)

Report problems; let them decide the fix.

```python
import arcpy
fc = "labels_900"
arcpy.management.CheckGeometry(fc, r"in_memory\geom_problems")

flags = []
with arcpy.da.SearchCursor(fc, ["OID@", "SHAPE@AREA", "kind", "unsure"]) as cur:
    for oid, area, kind, unsure in cur:
        if area < 8:                 # m^2 - below the 10 m^2 rule, probably a sliver
            flags.append((oid, f"tiny: {area:.1f} m2"))
        if area > 20000:             # bigger than a warehouse - probably a bad merge
            flags.append((oid, f"huge: {area:.0f} m2"))
        if kind in (None, "", " "):
            flags.append((oid, "kind not set"))
for oid, why in flags:
    print(oid, why)
```

Also worth checking and reporting:

- polygons that **overlap each other** (usually a split that didn't finish)
- polygons whose vertices sit outside the chip extent
- chips where **every** polygon still has its original SAM geometry — suggests that
  chip was opened but not actually reviewed
- how many features have `kind` still at the default `house` in a chip that obviously
  contains sheds

### 4. Track progress

Per chip: feature count, count by `kind`, number `unsure`, number `edge`. A simple
table across all 15 chips helps them see what's left and keeps their judgement
consistent.

### 5. Answer rule questions

Use the rules above. When genuinely ambiguous, the right answer is almost always
"draw it and set `unsure = 1`" rather than a long deliberation.

### 6. Final validation before they send it back

```python
import geopandas as gpd, glob
for f in sorted(glob.glob("labels_*.shp")):
    g = gpd.read_file(f)
    print(f, len(g),
          "invalid:", int((~g.geometry.is_valid).sum()),
          "no kind:", int(g["kind"].isna().sum() + (g["kind"] == "").sum()),
          "unsure:", int(g["unsure"].sum()))
```

Everything must be valid geometry, in **EPSG:3857**, with `kind` set on every feature.

---

## When they're done

They zip the folder and return it to the server, where it gets locked as
evaluation-only and used to re-score every model with imagery alongside the numbers.

If they ask what happens next: the gold numbers decide whether the next step is
retraining, changing model architecture (instance segmentation, so touching buildings
can't merge), or something else. Nobody knows yet — that's the point of measuring
first.

---

## Useful background if they ask

- **Two models are being compared.** "D-FINE" is a box detector trained on the user's
  own hand labels (which include sheds). "SegFormer" is a segmenter trained on
  Microsoft/OSM footprints (which don't). SegFormer scored much better on the old
  metric; the user's eye said otherwise; the eye was right.
- **A third was tested**: geoai's pretrained US Mask R-CNN. It scored 0.464 vs our own
  models, but only after being run at 0.6 m/pixel — at our native 0.15 m/pixel it
  scored 0.042 because it fragments roofs it was never trained to see that close.
- **Known failure**: the box detector finds 0 of 3 warehouses on large-roof chips.
  That's why chips 914 and 915 are in the set.
