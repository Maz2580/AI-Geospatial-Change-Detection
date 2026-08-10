# Vision-model candidate review (NVIDIA NIM)

Runs a vision model over each change candidate and records what it thinks changed.
The geometry pipeline is good at finding *where* something changed and poor at
saying *what*: a solar array added to an existing roof scores like a new building.

The label is advisory. It is written to a `visual_review` property and never
overwrites `classification`.

## Status

Implemented and unit-tested, but **not yet exercised against the live endpoint**.
The development machine's network blocks `integrate.api.nvidia.com` (see
[Network requirement](#network-requirement)). Everything up to the HTTP call is
verified, including crop rendering and payload size.

## Prerequisites

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your key to `.env`:

```
NVIDIA_API_KEY=nvapi-...
```

Get a key from <https://build.nvidia.com> — sign in, open any model, and use
"Get API Key". `.env` is gitignored. Never commit it or paste the key into chat.

> Watch the spelling. `NVIDIA_API_KEY`, not `VIDIA_API_KEY`.

## Network requirement

The API host must be reachable. Check before anything else:

```powershell
Test-NetConnection integrate.api.nvidia.com -Port 443
Invoke-WebRequest -Uri https://integrate.api.nvidia.com/v1/models -Method Head -UseBasicParsing
```

On the original development machine the result was:

| Host | Result |
| --- | --- |
| `integrate.api.nvidia.com` | DNS ok, TCP 443 ok, **TLS reset / timeout** |
| `api.nvcf.nvidia.com` | HTTP 401 (reachable) |
| `build.nvidia.com` | HTTP 202 (reachable) |
| `huggingface.co` | HTTP 200 (reachable) |

TCP connects but TLS is killed, which is SNI-based filtering rather than a
certificate problem. No code change works around it. If you hit the same thing,
ask for `integrate.api.nvidia.com` to be allowlisted — it is a single host, so it
can be a narrow exception rather than a broad rule.

## Verify the key

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\review_candidates_visually.py --list-models
```

Lists what the key can reach and flags which of the known vision models are
available to it. If this fails, nothing below will work.

## Run

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src\review_candidates_visually.py `
  --candidates data\output\dino_fixed\footprint_change_candidates.geojson `
  --before "data\input\EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000\EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif" `
  --after  "data\input\EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000\EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif" `
  --before-date 2021-12-01 --after-date 2026-04-18 `
  --max-candidates 20 --min-area-m2 40 `
  --output data\output\dino_fixed\candidates_reviewed.geojson
```

Start with `--max-candidates 20` to confirm the labels are sensible before
spending a full run on all 142.

## Models

Verified against the NIM **Multimodal APIs** catalogue, not the LLM catalogue —
vision models do not appear on the LLM APIs page.

| Model | Notes |
| --- | --- |
| `meta/llama-3.2-11b-vision-instruct` | **Default.** Free tier, ample for a six-way crop classification. |
| `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | NVIDIA's own 8B VLM, also light. |
| `meta/llama-3.2-90b-vision-instruct` | Same API, much heavier. Use only if the small models disagree with human labels. |

Override with `--model`.

## Labels

One of these, or the reply is rejected:

| Label | Meaning |
| --- | --- |
| `new_building` | A structure that did not exist before now stands there |
| `building_extension` | An existing structure grew |
| `solar_panels` | Panels added to an existing roof |
| `hardscape` | Driveway, path, slab, pool, fence, other ground surface work |
| `vegetation` | Planting, clearing, growth, or seasonal difference only |
| `no_visible_change` | Nothing meaningful differs |
| `unclear` | Too small, blurred, or obscured to judge |

Written per feature as:

```json
"visual_review": {
  "label": "solar_panels",
  "confidence": 0.82,
  "reason": "panels added to existing roof",
  "parsed": true,
  "evidence_role": "advisory_visual_review"
}
```

`parsed: false` means the reply could not be validated and the label was forced
to `unclear`. A summary lands in `visual_review_report.json` beside the output.

## API details worth knowing

Verified against <https://docs.api.nvidia.com/nim/reference/multimodal-apis>.

- **Invocation path is per-model**: `POST https://integrate.api.nvidia.com/v1/{model}`.
  It is *not* `/v1/chat/completions` — that is the LLM route. Use `--openai-route`
  for a non-NIM provider that expects the OpenAI path.
- **202 means poll, not fail.** A pending result returns `202` with an
  `NVCF-REQID` header; the client must poll
  `https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{requestId}`. Handled
  automatically, up to `poll_timeout_s` (default 180 s).
- **Images go inline as data URIs**: content parts with `type: "image_url"` and
  `image_url.url` set to `data:image/jpeg;base64,{...}`, under a single
  `role: "user"` message. Media is only allowed on the user role.
- **Inline payload cap is 180 kB** for both base64 images combined. Crops are
  encoded to a byte budget (`max_image_bytes`, default 62 kB each), stepping JPEG
  quality down and then halving the crop until they fit. A measured run came in
  at ~158 kB per request.
- **Bounds**: `temperature` 0–2, `max_tokens` 1–8192. Both validated locally.

## Using a different provider

`ReviewProvider` is a protocol with one method, so swapping providers does not
touch the review logic:

```python
class ReviewProvider(Protocol):
    def classify(self, before_jpeg: bytes, after_jpeg: bytes, prompt: str) -> str: ...
```

For another OpenAI-compatible endpoint:

```powershell
.\venv\Scripts\python.exe src\review_candidates_visually.py `
  --base-url https://your-endpoint/v1 --model your-model --openai-route `
  --api-key-env YOUR_KEY_ENV_VAR ...
```

## Security notes

- The key is read from the environment only, never written to any output or log.
- HTTP error handling reports the status code only. Response bodies can echo the
  submitted request, so they are not logged.
- **Model replies are untrusted input.** `parse_verdict` accepts JSON in a code
  fence or surrounded by prose, then rejects any label outside the allowlist,
  clamps confidence to 0–1, and truncates the reason. A reply containing an
  injected instruction instead of a verdict degrades to `unclear` / `0.0` rather
  than propagating. There is a test for exactly that.
- Imagery crops leave your machine. Confirm that is acceptable under your Nearmap
  licence before running against a real AOI.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `No API key found in $NVIDIA_API_KEY` | Key missing from `.env`, or the name is misspelled |
| Connection reset / timeout | Host blocked by the network, see above |
| `HTTP 401` | Key invalid or expired |
| `HTTP 404` | Model name wrong, or `--openai-route` used against a NIM path |
| `HTTP 422` | Payload rejected. Usually the images exceed the inline cap; lower `--crop-padding-m` |
| `HTTP 429` | Free-tier rate limit. Reduce `--max-candidates` or retry later |
| All labels `unclear` with `parsed: false` | Model is not returning JSON. Try a different `--model` |

## Next step once it runs

Score the labels against the 14 human-reviewed sites in
`data/labels/murchison_estate_2022_2026_site_review_labels.json` to get an
agreement rate. Do not let the vision model gate anything until that number
exists.
