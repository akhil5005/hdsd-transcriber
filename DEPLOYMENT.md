# Deployment notes

Frontend: **Vercel**. Database: **Neon** (managed Postgres). Backend: see below.

## What is live

| | |
| --- | --- |
| Frontend | <https://hdsd-transcriber.vercel.app> (Vercel, Hobby) |
| API (inference) | <https://35-168-45-255.sslip.io> (AWS EC2 `m7i-flex.large`, us-east-1) |
| API (benchmark only, no weights) | <https://hdsd-transcriber-api.onrender.com> (Render free, kept as a fallback) |
| Database | Neon project `hdsd-transcriber`, AWS us-west-2, free plan |

### Why there are two API hosts

Render's free instance is 512 MB and cannot hold Whisper-small, so it runs
`INFERENCE_ENABLED=false` and transcription returns 503 there. The EC2 box has
8 GB and the real checkpoints, so it is the one that transcribes. Both share
the same Neon database, so history and the benchmark are identical on either.

### The EC2 host

| | |
| --- | --- |
| Instance | `m7i-flex.large`, 2 vCPU / 8 GiB, Ubuntu 24.04, 20 GB gp3 |
| Why not `t3.medium` | the AWS Free Plan blocks it; only t3.micro/small, t4g.micro/small and the two flex larges are allowed, and only the flex larges have enough RAM |
| Address | Elastic IP `35.168.45.255` so stop/start does not change it |
| TLS | Caddy + Let's Encrypt on `35-168-45-255.sslip.io`. `sslip.io` resolves a dashed IP to that IP, so a real certificate needs no domain registration |
| Service | `hdsd-api.service`, runs `alembic upgrade head` then uvicorn on 127.0.0.1:8000 |
| Weights | `~/research/full_ft` and `~/research/lora`, uploaded by scp and checked with sha256 |
| Speed | CPU only: roughly 50 s for a 4 s clip across all three models (RTF 1.5-2.2) |

**HTTPS is not optional.** The frontend is HTTPS on Vercel, and a browser
blocks requests from an HTTPS page to a plain-HTTP backend as mixed content.
A bare EC2 IP cannot get a trusted certificate, which is what `sslip.io`
solves.

### Cost control

`m7i-flex.large` is roughly $0.09/hour, about $69/month running continuously,
so $200 of Free Plan credit lasts about three months. Stop it between demos:

```bash
aws ec2 stop-instances  --instance-ids i-0423b69eb0b53c20b
aws ec2 start-instances --instance-ids i-0423b69eb0b53c20b
```

A stopped instance bills only the disk (~$1.60/month) and the Elastic IP
(~$3.60/month). The Elastic IP means the hostname and certificate survive the
restart; without it every restart would need a new Caddyfile and a new
`VITE_API_URL`.

**The Free Plan ends six months after account creation**, and the account is
closed or starts billing at that point. Railway Hobby ($5/month) is the
cheaper long-term home if this needs to stay up.

Both redeploy from `main` automatically. Two settings are easy to get wrong:

- **Vercel Root Directory must be `frontend`.** The repo root holds the backend
  too, and Vercel reads `render.yaml` and offers to import the API as a second
  service. Import the `frontend` app alone.
- **`VITE_API_URL` is inlined at build time**, not read at runtime, so it has to
  exist before the build. Changing it in the Vercel dashboard does nothing until
  the next deploy.

CORS needs no change for a `*.vercel.app` domain -- `CORS_ORIGIN_REGEX` in
`render.yaml` already matches one. A custom domain would have to be added to
`CORS_ORIGINS` on the Render service.

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

## Seeding a hosted database

Migrations run on every deploy (`alembic upgrade head` is the first half of the
start command), but they only create empty tables. The benchmark figures come
from the `final_v3` CSVs, which live in the research archive and are therefore
not on the host. Seed from a machine that *does* have the archive, pointing at
the hosted database:

```powershell
$env:DATABASE_URL = "<the provider's connection string>"
python -m app.scripts.seed_benchmark
$env:DATABASE_URL = $null
```

That writes 21 per-run rows, 9 seed-spread rows, 7 efficiency rows and the
three-model registry — aggregate metrics only (WER/CER percentages, parameter
counts, checkpoint sizes), no transcripts, audio or weights. It is idempotent,
so re-run it after a schema change. Without it the benchmark endpoints answer
200 with empty charts rather than failing.

**Corpus sentences are deliberately not seeded to a hosted database.**
`data/hdsd_prompts.csv` is gitignored, and on a host where transcription is
disabled the prompts have nothing to score anyway.

Use the provider's **direct** connection string, not a pooled one: asyncpg
keeps prepared statements, which PgBouncer in transaction mode rejects, and
the engine sets no `statement_cache_size=0`.

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
