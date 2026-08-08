# Model-candidate review workflow

Model candidates are not automatically construction targets. Record a review
for every candidate shown in a model report using one of these outcomes:

- `confirmed_target_change`: a visible, permanent new building, extension, or
  demolition. Record `target_type` and an assessed outline quality.
- `temporary_or_movable_change`: a visible change such as a ship, vehicle, or
  temporary structure. It is a real observation but outside the permanent
  construction target.
- `not_target_change`: no target change; for example a transient vehicle or
  paving/colour appearance change.
- `inconclusive_due_image_ambiguity`: the pixels do not permit a reliable
  decision, even by visual inspection.

For example, validate the committed Docklands UMAMI review with:

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\validate_candidate_review.py `
  --review data\benchmarks\docklands_pilot_2020_2023\umami_candidate_human_review.json
```

Do not convert temporary/movable or inconclusive candidates into false-positive
metrics until a target definition and complete negative labels have been frozen
for a held-out case.
