# Targeted Candidate Expansion

Use this optional refinement after inspecting a conservative change-detection result. It grows only the candidate polygons you provide into nearby, lower-confidence change pixels; it cannot create a new candidate elsewhere in the AOI.

This is useful when a roof, building pad, or construction footprint appears fragmented in the first mask.

```powershell
.\venv\Scripts\python.exe src\expand_confirmed_candidates.py `
  --score data\output\example_site\change_score.tif `
  --seed-candidates data\output\example_site\construction_change_candidates.geojson `
  --grow-percentile 96.5 --max-distance-m 6 --closing-m 1 `
  --output data\output\example_site_expanded
```

- `--grow-percentile`: lower values recover more weak change pixels; begin at `96.5`.
- `--max-distance-m`: the furthest distance a footprint can grow from its supplied seed; begin at `6` m.
- `--closing-m`: connects small internal gaps; begin at `1` m.

Review the new output before replacing an original result. This procedure increases footprint completeness, but it does not turn RGB-only evidence into confirmed new construction; DSM height evidence remains the stronger directional signal.
