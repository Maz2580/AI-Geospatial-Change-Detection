# Construction-site review candidates

One development site may include a new building, an extension, a pool, a
driveway, landscaping, and incidental shadows. Showing each raw polygon as a
separate result hides that relationship and encourages false building labels.

`build_construction_sites.py` combines dated change evidence with dated
footprint-comparison evidence into a review site. It does not claim every
object in the site is a building.

```powershell
.\venv\Scripts\python.exe src\build_construction_sites.py `
  --changes data\output\estate\fusion\fused_candidates.geojson `
  --footprints data\output\estate\footprint_comparison\footprint_change_candidates.geojson `
  --anchor-merge-distance-m 4 --site-radius-m 20 `
  --output data\output\estate\construction_sites
```

Use the tight values shown above as the initial setting for suburban lots.
Larger distances can connect an entire new estate through a chain of nearby
model polygons; reduce the merge distance before assuming one large geometry is
one construction site.

Each site preserves:

- `change_candidate_ids` and `change_sources` for dated change evidence;
- `footprint_candidate_ids` for the object outlines seen at the after date;
- `review_priority`, which ranks evidence but does not determine object type;
- a manual review question covering building, extension, pool, hardscape,
  garden, or no permanent construction.

Footprint evidence is refinement, not an independent model vote. In particular,
a driveway may produce a footprint-like polygon, and multiple model votes can
still be wrong about a building type.
