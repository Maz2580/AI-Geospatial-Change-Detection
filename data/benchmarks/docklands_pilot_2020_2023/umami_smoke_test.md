# Docklands UMAMI smoke test — 8 August 2026

This is a **single-scene diagnostic**, not a model-selection score. The frozen
Nearmap pair is 28 April 2020 to 6 July 2023. The test AOI is deliberately
small (`144.9447, -37.8171, 144.9457, -37.8166`) and contains one
human-confirmed visibly new building from the Docklands pilot.

The City of Melbourne 2020-to-2023 footprint difference supplies the review
location only. Its geometry is not a pixel-perfect roof label. Candidate-to-
reference matching used the pre-existing 8 m tolerance and is therefore a
coverage check, not outline-IoU.

| Candidate source | Service-filtered? | Candidates | Matched model IDs | Unmatched IDs needing review |
| --- | --- | ---: | --- | --- |
| UMAMI D-FINE, `change`, long side 768, percentile 90 | No | 4 | 1, 4 | 2, 3 |
| UMAMI D-FINE, same request | Yes | 2 | 1, 4 | none |
| UMAMI SegFormer, `change`, long side 768, percentile 90 | No | 6 | 1, 6 | 2, 3, 4, 5 |
| UMAMI SegFormer, same request | Yes | 2 | 1 | 3 |

Both full candidate sets and both service-filtered sets detected the one
confirmed reference candidate. This does **not** establish model accuracy or
precision: the reference scene has only one confirmed change and no complete
negative labels. In particular, the unmatched candidates must remain review
items rather than being called false positives.

The visual question still outstanding is outline quality: whether the matching
candidate traces the visible roof/building acceptably or only overlaps it. The
HTML reviews under `data/output` are intentionally ignored by Git; the
secret-free GeoJSON candidates, request reports, and reference-evaluation
reports in the sibling `umami_dfine_change` and `umami_segformer_change`
directories are versioned.
