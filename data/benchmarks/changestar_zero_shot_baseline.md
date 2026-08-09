# Changen2 ChangeStar zero-shot baseline record

Status: **not promoted**. This is a bounded CPU-only, no-training experiment,
not a production score.

## Fixed configuration

- Model: `Changen2-ChangeStar1x256:s1c1_cstar_vitb`
- Packages: `torchange 0.0.4`, `torch 2.7.1+cpu`, `albumentations 2.0.8`
- Published weight SHA-256:
  `ba540ed456001c0a909fe0f464ed4c720d1e5b8f373e811164a09ac1da574b75`
- Device: CPU
- Patch size: 256 px
- Inspection threshold: 0.5
- Minimum vector area: 20 m²
- No per-site threshold, tile, or model change was made.

## Docklands pilot, 2020-04-28 to 2023-07-06

Frozen input SHA-256 values:

- before: `58f5d534bb83adcbf2bfbd4ee10115121f2fdfb97014ae4ee0f22e837f5f8007`
- after: `281aa929f145d8bf41150f615fc5d686f0027bdead8cb3a000eff129fea4f554`

Runtime: about 87 seconds on CPU. The run produced 14 after-only building
proposals and 12 change-and-after-building agreement proposals.

For a diagnostic spatial comparison only, the best after-only proposal covered
97.69% of the previous human-confirmed D-FINE building prompt (84.47% IoU).
The conservative agreement proposal covered 82.06% (74.71% IoU). Neither layer
overlapped the human-reviewed ship/cruise or temporary-structure prompts.

This is encouraging on this one case, but it is not an independent ground-truth
score: the D-FINE candidate geometry is a review prompt, not a digitised roof
label.

## Melbourne CBD extension A, 2020-11-08 to 2023-11-10

Frozen input SHA-256 values:

- before: `bb2c09a42a5161a1f7d0f65df2bf7059fc44b5e3a010b81baf956d2c3307424c`
- after: `6ab338c801424a25a75508b1f5bc0d9c2e41987c43bbbafcf43c4767d2c0db88`

Runtime: about 79 seconds on CPU. The run produced 48 after-only building
proposals and 9 agreement proposals.

The previously reviewed real building-extension prompt had only 7.29% coverage
from the best after-only proposal, and neither that target nor the reviewed
new-building/site-change prompt overlapped an agreement proposal. The reviewed
non-change prompt also had no agreement overlap.

## Decision

ChangeStar demonstrates that a bi-temporal model can provide substantially
better semantic building geometry than RGB difference on the Docklands case,
but it fails the CBD extension case under the same frozen configuration.
Therefore it is retained as a reproducible research baseline only. Do **not**
tune it on these two cases or present it as an Australian production model.

The next evidence needed is manual after-date roof digitisation for several
clear Victorian changes, followed by a held-out comparison. Until then, retain
the output as separate site-alert and after-building-proposal layers and keep
human review in the decision path.
