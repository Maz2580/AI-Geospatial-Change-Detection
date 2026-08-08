# Melbourne CBD Extension A UMAMI paired test — 8 August 2026

This compact non-waterfront scene centres on the large previously
human-reviewed real visible change represented by Candidate 1 in the parent
Melbourne CBD West reference review. It uses the fixed Nearmap pair 8 November
2020 to 10 November 2023 and the small UMAMI AOI
`144.9506, -37.8105, 144.9514, -37.80975`.

Both requests used `mode=change`, `long_side=768`, `percentile=90`, SAM
refinement, and footprint regularisation. The only requested difference was
the detector: D-FINE versus SegFormer.

## Observed service behaviour

Both requests returned three candidates and marked all three relevant. The
candidate geometries, classifications, and labels are identical pair by pair:

1. 445.7 m² `new building` / `Change on an existing building`;
2. 12.1 m² `new building` / `New construction or cleared land`;
3. 7.4 m² `cleared / demolition` / `Likely vehicle (transient — on road/parking)`.

The request reports retain different `requested_detector` and `fuse_detector`
values, so both detector requests reached the service. However, identical
visible candidate output means this scene cannot distinguish the detector
branches: a shared detection, semantic, or post-processing stage may dominate
at this scale. Do not select a model from this test.

The raw secret-free outputs are versioned in the sibling
`umami_extension_a_dfine` and `umami_extension_a_segformer` directories. Their
HTML visual reviews are ignored under `data/output`; they require a human
assessment before this test can be used for any candidate-quality comparison.
