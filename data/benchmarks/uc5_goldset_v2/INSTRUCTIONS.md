# Gold evaluation set — how to label it

**What this is for.** Fifteen chips that become our permanent scoreboard. Every model
we have or ever train gets judged on these, and **they are never trained on**. Once
you have labelled them, a number finally means "did it find the buildings", instead
of "did it agree with Microsoft".

**What's in the set.** Eleven residential chips spanning dense established streets
(Box Hill, Altona North), tree-shaded suburbs, large-lot outer areas (Sunbury) and a
townhouse development (Derrimut, chip 913 — the merging test), one commercial strip
(chip 900), one large warehouse (chip 905) and two industrial estates with several
distinct units each (chips 914, 915 — the case D-FINE currently scores 0/3 on).

**Why we're doing it.** Every accuracy number reported so far was measured against
Microsoft/OSM footprints. Those were also the SegFormer's training labels, and they
leave out most sheds, garages and carports — the very things encroachment is about.
That is why the numbers kept saying one thing and your eye kept saying another.

**Time needed.** Roughly 20–40 minutes per chip across 15 chips, so a solid day or two half-days. Do it in one
or two sittings if you can, so your judgement stays consistent across chips.

---

## The rules (decided 2026-07-29)

### 1. What to draw

Draw **anything with a roof, about 10 m² and bigger**:

| draw it | don't draw it |
|---|---|
| houses | swimming pools |
| detached garages | water tanks |
| sheds and outbuildings | solar panels (they sit *on* a roof) |
| carports (roof on posts, open sides) | decks and paving with no roof |
| patio and pergola roofs | vehicles, caravans, shipping containers* |
| warehouses and industrial units | trees, fences, walls |

\* A shipping container being used as a permanent structure **is** worth drawing —
mark it `unsure` if you're not certain, see below.

About 10 m² is roughly a 3 m × 3 m square — a bit bigger than a small garden shed.
Below that, skip it.

### 2. One outline per connected roof

If it's **joined to the house**, it's part of the same outline:

```
   house + attached garage + rear extension   ->  ONE polygon
   house  |  garage 8 m away                  ->  TWO polygons
```

This is what you described: you draw the separate outline when the garage is away
from the building. If two roofs touch, they're one shape. If there's a visible gap
of open ground between them, they're two.

**Terraces and townhouses:** if you can see the dividing line between two dwellings
in a row, draw them separately. If you genuinely can't tell where one ends, draw the
row as one and set `unsure = 1`.

### 3. Trace the roof, as you see it from above

Draw the edge of the **roof**, not a guess at the walls underneath. At 15 cm imagery
the roof is what's visible, and it's what every model predicts, so this keeps the
comparison honest. Don't try to correct for the slight lean on tall buildings.

Follow the actual roof shape — L-shapes, T-shapes and cross-gables should have their
real outline, not a rectangle around them.

### 4. Buildings at the edge of the chip

If a building is **cut off by the chip border**, draw the part you can see, and set
`edge = 1`. We'll exclude those from strict scoring, because a model can't fairly be
judged on half a building.

### 5. When you're not sure

Set **`unsure = 1`** and move on. Don't agonise. Anything marked unsure is reported
separately and kept out of the strict score, so a doubtful case costs us nothing —
whereas a wrong confident label quietly corrupts every future comparison.

Good reasons to mark unsure: heavy tree shadow, can't tell shed from carport, can't
tell if a structure is permanent, can't find the dividing line in a terrace.

---

## The fields to fill

The layer comes with these already set up. Only `kind` needs a choice; the rest
default to 0.

| field | values | meaning |
|---|---|---|
| `kind` | `house`, `garage_shed`, `carport`, `industrial`, `other` | what it is |
| `unsure` | 0 or 1 | you couldn't tell — excluded from strict scoring |
| `edge` | 0 or 1 | cut off by the chip border |
| `notes` | free text | optional, anything odd worth remembering |

`kind` matters more than it looks: it lets us answer "does the model miss **sheds**
specifically?" rather than just "the model missed things".

### The `nest` field (added in v2)

Labelling chips 900 and 913 turned up a systematic SAM 3 fault: on complex or joined
roofs it draws the building **twice or three times** — once per roof section, and
again as one shape covering the lot. Measured across the set, **150 of 450 proposals
(33%) sit inside a larger proposal**, concentrated almost entirely in the industrial
and commercial chips.

Two read-only fields now mark it:

| field | values | meaning |
|---|---|---|
| `nest` | `parent` | this shape swallows other shapes |
| | `part` | this shape sits inside a bigger one |
| | `both` | inside one, containing others |
| | *(blank)* | no conflict |
| `nest_grp` | integer | cluster id; everything sharing a number is one pile-up, `0` = not in one |

**Select By Attributes → `nest <> ''`** lists every conflict in a chip.

To resolve one, look at the imagery and ask **is this one building or several?**

- one building SAM split into sections → keep the `parent`, delete the `part` shapes
- several buildings SAM also drew as one blob → delete the `parent`, keep the `part`s

This deliberately is not automated. The two cases are geometrically identical — a big
polygon containing small ones — and only the imagery separates them. A script picking
automatically would guess, and would guess wrong most often on exactly the large-roof
chips the gold set exists to measure.

---

## How to do it in ArcGIS Pro

You'll get a folder `uc5_goldset/` containing, for each of the 15 chips:

```
chip_9xx.tif          the aerial image, already georeferenced
labels_9xx.shp        SAM 3's proposed outlines - what you correct
```

plus `goldset.gpkg` holding the same labels if you'd rather use a GeoPackage.

1. **New Map**, then drag `chip_9xx.tif` in. It'll land in the right place on its own.
2. Drag `labels_9xx.shp` on top. Give it a hollow fill with a bright outline so you
   can see the imagery through it.
3. **Edit → Modify Features.** Then, per building:
   - **wrong** (a tree, a pool, nothing there) → select, delete
   - **roughly right** → *Vertices* / *Reshape* to fix the edges
   - **one shape covering two separate buildings** → *Split* them apart
   - **two shapes on one connected roof** → *Merge* them
   - **missing entirely** → *Create* → draw a new polygon
4. Set `kind` on each — the attribute table is fastest for this. Select all the
   obvious houses, set `kind` once for the selection, then handle the rest.
5. **Save** the edits (Edit → Save). Do this often.

> **The one thing I need you to do deliberately: hunt for what isn't there.**
> Pre-drawn outlines make people fix what they can see and miss what's absent. Before
> finishing a chip, sweep it once ignoring the outlines completely, just looking for
> roofs with nothing drawn on them. Backyard sheds and carports under trees are the
> ones that get missed, and those are exactly what encroachment is about.

### QGIS instead

Same thing: add the `.tif`, add the `.shp`, toggle editing (pencil icon), use the
vertex tool and the add-polygon tool. Save the layer when done.

---

## When you're finished

Zip the `uc5_goldset` folder and put it back on the server, or tell me where it is.
I'll then:

1. Check every polygon is valid and inside its chip.
2. Report what you labelled: counts by `kind`, how many `unsure`, how it compares to
   what Microsoft/OSM has for the same areas.
3. Lock it as **evaluation-only, permanently** — it never enters any training set.
4. Re-score D-FINE, SegFormer and the geoai model against it, with pictures.

That last number is the first one either of us should trust.

---

## Notes on what you'll be correcting

SAM 3 is Meta's segmentation model, prompted with the word "building". It has never
seen Melbourne specifically and was not trained on our labels — which is the point.
Its mistakes are unrelated to our models' mistakes, so correcting it doesn't quietly
teach the scoreboard our own blind spots.

Expect it to be good at obvious houses, and less reliable on:

- small sheds in shadow (often missed → you'll be adding these)
- carports (roof but no walls — it may or may not call these buildings)
- large warehouse roofs (may be split into panels → you'll merge)
- roof sections read as separate buildings (→ merge)

If it turns out to be more trouble than help on a chip, delete everything and draw
from scratch. That's a legitimate outcome and worth telling me about.
