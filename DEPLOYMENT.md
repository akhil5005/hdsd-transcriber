# Deployment notes

Frontend: **Vercel**. Database: **Neon** (managed Postgres). Backend: see below.

## What the backend actually needs

| Resource | Requirement | Why |
| --- | --- | --- |
| RAM | **~3 GB** resident, 4 GB to be safe | Whisper-small is 244 M params in FP32 ≈ 973 MB of weights. `pretrained`, `full_ft` and `lora` are cached independently (PEFT wraps its own base copy), so "compare all 3" holds ~2.9 GB plus activations and the torch runtime. |
| Disk | **~5 GB** | 967 MB full-FT checkpoint + ~967 MB base-model snapshot + ~1 GB torch install + uploads. |
| CPU | 2 vCPU usable | Greedy decoding, FP32, no GPU required. Roughly 10–30 s per 30 s window per model. |
| Storage | Ephemeral is acceptable | Uploads are deletable by design (`MEDIA_RETENTION_DAYS`); transcripts live in Postgres. |

**Render's free tier is 512 MB RAM, so it cannot run inference at all.** `render.yaml`
is kept as an API-only deployment with `INFERENCE_ENABLED=false`: auth, the research
benchmark, history and exports work; the transcription endpoints return 503.

## Hosts that can run the models

Checked October 2026. Prices are list prices for the smallest configuration that fits.

| Host | Cost | RAM / CPU | Deploy | Notes |
| --- | --- | --- | --- | --- |
| **Railway Hobby** | **$5/mo, includes $5 usage** | 8 GB / 8 vCPU per service | git push | Per-second billing, scale-to-zero. RAM is $10/GB-month and CPU $20/vCPU-month, so ~4 GB + 2 vCPU costs ≈ $0.11/active-hour — the included $5 covers roughly 45 active hours per month. Same `uvicorn` start command as Render. |
| **Hugging Face Spaces + PRO** | $9/mo | 16 GB / 2 vCPU, 50 GB disk | git push (Dockerfile) | ML-native: the pinned base revision pulls straight from the Hub. Sleeps after 48 h idle, not 15 min. Docker Spaces have required PRO since July 2026. |
| **Modal** | $0 (with $30/mo credits) | per-second CPU or GPU | `modal deploy` | Credits cover ≈ 50 h/month of T4 GPU, which makes transcription near-realtime. Serves FastAPI via `@modal.asgi_app()`. Least conventional shape; weights live in a Modal Volume. |
| **Oracle Cloud Always Free** | $0 forever | 2 OCPU / 12 GB ARM, 200 GB | manual | Genuinely free and always on. Was 4 OCPU / 24 GB until Oracle halved it on 15 June 2026. No git-push deploy — systemd, nginx and certs are yours to run. |
| **Google Cloud Run** | ~$0 within free tier | 4 GiB / 2 vCPU | container | Free tier is 180k vCPU-s + 360k GiB-s per month ≈ 25 instance-hours at this size, then ~$0.10/h. Scale-to-zero, but a cold start pays the full model load (30–90 s). |

Ruled out on price or capacity:

- **Render paid** — Standard is 2 GB ($25/mo), still short; Pro 4 GB is $85/mo.
- **Fly.io** — 4 GB lands around $25/mo after the 2026 RAM increase; free allowances are gone.
- **Hetzner** — CPX22 went from €6.49 to €19.99 in 2026. CX23 (2 vCPU / 4 GB) is €5.49 but availability is patchy, and 4 GB is tight for three cached models.
- **Koyeb** — 512 MB, and a $29 card hold since February 2026.

**Recommendation: Railway Hobby.** It is the smallest change from what is already
built (identical start command and env vars), has more headroom than the app needs,
and the included credit covers normal demo traffic.

## The checkpoint-copying question

Every option above requires the **full fine-tuned checkpoint (967 MB) to be copied out
of the research archive** into a container image, volume or VM. `PROMPT.md` §3 and §11
forbid copying anything inside `<RESEARCH_ARCHIVE>`,
and these are patient-derived weights, so that is a decision to make deliberately rather
than a side effect of deploying.

The two lighter models do not have this problem:

- `pretrained` — public `openai/whisper-small` at the pinned revision, downloaded by the
  host. Nothing leaves this machine.
- `lora` — the adapter is **7 MB**, and it loads on top of that same public base model.

So a deployment serving `pretrained` + `lora` needs only a 7 MB upload, while adding
`full_ft` means moving the 967 MB checkpoint. Worth deciding before Phase 3 wires the
model registry to a deployed host.
