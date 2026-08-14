# Gold set v2 — start here

This replaces the first download. Same 15 chips, same rules — two fixes based on what
you found in 900 and 913.

## What changed

**1. Chip 900 is repaired.** Your file had come back with every shape duplicated
(69 rows, only 35 real ones — a paste accident in ArcGIS Pro, not your labelling).
The copies are gone. Your 35 shapes are exactly as you drew them.

**2. Every chip now has a `nest` field** marking the fault you spotted — where SAM 3
drew a complex building three times: once per section *and* once for the whole thing.

**3. Chip 913 is finished.** You already cleaned it. Don't touch it again.

## Where you are

| | chips | what's needed |
|---|---|---|
| ✅ done | 913 | nothing |
| ⚠️ nearly done | 900 | resolve the nested shapes, set `kind` |
| ⬜ to do | 901–912, 914, 915 | the full pass |

## The new `nest` field — this is the time-saver

| value | meaning |
|---|---|
| `parent` | this shape swallows other shapes |
| `part` | this shape sits inside a bigger one |
| `both` | it does both — inside one, containing others |
| *(blank)* | no conflict, ignore |

`nest_grp` numbers the clusters. Everything with `nest_grp = 3` belongs to the same
pile-up, so you can select and judge a whole cluster at once instead of hunting FIDs.

**In ArcGIS Pro:** Map → *Select By Attributes* → `nest <> ''` shows you every
conflict in the chip. There are **126 of them across the 13 remaining chips**, and
they are heavily concentrated — chips 914, 915, 905 and 900 hold most of it, while
912 and 906 have almost none.

### How to resolve one

Look at the imagery and ask one question: **is this one building or several?**

- **One building** that SAM split into roof sections → keep the `parent`, delete the
  `part` shapes.
- **Several buildings** (a row of shops, a terrace) that SAM also drew as one big
  blob → delete the `parent`, keep the `part` shapes.

That's the whole decision. I can't automate it — both cases look identical in the
geometry, and only the picture tells them apart. That's also why it's worth your eye
rather than another model's.

If a cluster is genuinely unclear, keep your best guess and set `unsure = 1`.

## Suggested order

Start with **912, 906, 909** — barely any nesting, quick wins to get your rhythm.
Leave **914, 915, 905** (the industrial ones) for last: they have the most conflicts
but only 6–15 shapes each, so they go fast once you've got the pattern.

## Don't forget `kind`

Both 900 and 913 came back with everything still set to the default `house`. In 900
that's a commercial strip, so most of it should be `industrial` or `other`. It's the
field that lets us answer *"does the model specifically miss sheds?"*, so it's worth
the extra minute per chip.

Full rules are in `INSTRUCTIONS.md`.

## One habit worth keeping

When you save and zip a chip, glance at the feature count in the attribute table. If
it has roughly doubled, that's the paste accident again — undo and re-save. It cost
us nothing this time because it was caught on chip 2 of 15.
