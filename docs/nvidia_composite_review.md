# NVIDIA visual review: current hosted API path

The NVIDIA NIM key and `/v1/models` endpoint are reachable from this machine. Current hosted NIM accepts one inline image reliably but returns HTTP 400 when a request contains two image parts. Use `src/review_candidates_nvidia_composite.py`, which places the dated crops side by side in one labelled JPEG and calls the OpenAI-compatible `/v1/chat/completions` endpoint.

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\review_candidates_nvidia_composite.py `
  --candidates data\benchmarks\docklands_pilot_2020_2023\umami_dfine_change\umami_dfine_change_all_candidates.geojson `
  --before data\output\docklands_pilot_2020_2023_imagery\imagery\before_2020-04-28_Vert_tiles.tif `
  --after data\output\docklands_pilot_2020_2023_imagery\imagery\after_2023-07-06_Vert_tiles.tif `
  --before-date 2020-04-28 --after-date 2023-07-06 `
  --max-candidates 4 `
  --output data\output\docklands_pilot_2020_2023_nvidia_smoke\candidates_reviewed.geojson
```

The result is advisory evidence (`visual_review`) and never changes the geometry pipeline's `classification`. Start with a small, human-reviewed sample and measure agreement before using it at scale. The source crops leave the machine for NVIDIA inference, so only use imagery that you are permitted to submit.
