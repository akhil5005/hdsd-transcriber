# HDSD Dysarthric-Speech Hindi Transcriber

Upload an MP4 or audio recording of a dysarthric speaker, pick one of three
Whisper-small variants (or all three), and get a Hindi transcript in Devanagari
with model-comparison metrics — both the saved research benchmark and live
scores on your own uploads.

> **Research prototype.** Exploratory results on the HDSD dataset. Not for
> diagnosis or clinical decisions.

---

## What it does

| | |
|---|---|
| **Three models** | Pretrained Whisper-small (pinned revision), full fine-tune (lr 5e-6, seed 123), LoRA adapter (lr 5e-5, seed 2026, r=16, α=32, q/v) |
| **Frozen decoding** | FP32, greedy (`num_beams=1`, `do_sample=False`), `language="hi"`, `task="transcribe"`, no timestamps, `max_new_tokens=96` per 30 s window |
| **Live comparison** | Side-by-side transcripts, WER/CER/S-D-I against an optional reference, word-level diff, inference time and real-time factor |
| **Research benchmark** | The frozen `final_v3` tables, seeded into the database and charted per evaluation case |

The app reproduces the research pipeline exactly: running the five verified
`test`-split recordings through the app gives **15/15 byte-identical matches**
with the saved `final_v3` predictions (see [Verification](#verification)).

---

## Requirements

| Component | Version used | Notes |
|---|---|---|
| Python | **3.11.9** | 3.12/3.13 will not work — `numpy==1.26.4` has no wheels for them |
| Node.js | 22.19 | any 18+ should do |
| PostgreSQL | **17** (local service) | 16 is fine too |
| ffmpeg | 9.0.2 | needed for audio extraction |
| GPU | optional | CUDA 11.8; CPU works and the UI says so |

### Verified environment

Windows 11, Python 3.11.9, torch 2.7.1+cu118, transformers 4.57.1, peft 0.17.1,
numpy 1.26.4, NVIDIA GTX 1650 (4 GB).

---

## Setup (Windows)

### 1. ffmpeg

```powershell
winget install Gyan.FFmpeg
```

winget edits `PATH` but **does not refresh already-open shells**, so either open
a new terminal or set absolute paths in `backend\.env`:

```
FFMPEG_BINARY=C:/Users/<you>/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_.../bin/ffmpeg.exe
FFPROBE_BINARY=C:/Users/<you>/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_.../bin/ffprobe.exe
```

Check with `ffmpeg -version`.

### 2. PostgreSQL

There is **no Docker in this project** — it uses a local PostgreSQL service.
Install PostgreSQL 16 or 17, then create a dedicated role and database (do not
use the `postgres` superuser):

```sql
CREATE ROLE hdsd WITH LOGIN PASSWORD 'choose-a-password';
CREATE DATABASE hdsd OWNER hdsd;
```

### 3. Backend

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate

# Web/API/DB/auth/metrics only — fast, no ML stack:
pip install -r requirements-dev.txt

# The ML stack (~2.5 GB). GPU build:
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

CPU-only machines can skip the CUDA index line; `pip install -r requirements.txt`
pulls the CPU build of torch.

Copy `backend\.env.example` to `backend\.env` and fill in at least:

```
DATABASE_URL=postgresql+asyncpg://hdsd:<password>@localhost:5432/hdsd
SECRET_KEY=<python -c "import secrets; print(secrets.token_urlsafe(64))">
```

Then:

```powershell
alembic upgrade head
python -m app.scripts.seed_benchmark
uvicorn app.main:app --reload --port 8000
```

`alembic` and the seed only need re-running after a schema change or a
checkpoint-path change.

### 4. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** and register an account. The Vite dev server
proxies `/api` to `http://127.0.0.1:8000`, so there is no CORS hop in dev.

---

## Running it

Two terminals:

```powershell
# Terminal 1
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend
npm run dev
```

API docs at <http://localhost:8000/docs>, health at
<http://localhost:8000/api/v1/health>.

---

## Model paths and the read-only rule

The weights live in the research archive and are **read in place** — the app
never copies, moves or writes to them.

```
HDSD_ROOT    = .../hdsd_simple/runs
FULL_FT_DIR  = .../runs/full_lr5e-06_seed123/best      (model.safetensors)
LORA_DIR     = .../runs/lora_lr5e-05_seed2026/best     (adapter_model.safetensors)
BENCHMARK_DIR= .../runs/final_v3/tables
BASE_MODEL_ID       = openai/whisper-small
BASE_MODEL_REVISION = 973afd24965f72e36ca33b3055d56a652f456b4d
```

Startup fails loudly, naming the path, if either checkpoint is missing.

Only five files are ever read from the archive: the two checkpoint folders and
`per_run.csv` / `model_summary.csv` / `efficiency.csv`.
`app/core/research_assets.py` enforces that with an allow-list, because the rest
of that directory holds private patient transcripts.

**Switching seeds.** `FULL_FT_DIR`, `LORA_DIR`, `FULL_FT_SEED` and `LORA_SEED`
are config values. Point them at a different run and the benchmark page follows
— the "selected" flag is derived from the configured checkpoint path, not
hard-coded.

---

## CPU vs GPU

`DEVICE=auto` uses CUDA when available, otherwise CPU. On CPU everything still
works, just slower, and the UI shows a banner saying so.

Whisper-small in FP32 is ~973 MB of weights per model. To fit all three on a
4 GB card:

| Setting | Default | Effect |
|---|---|---|
| `SHARE_BASE_MODEL` | `true` | `pretrained` and `lora` share one base instance (the adapter is toggled off with PEFT's `disable_adapter()`), saving **923 MB** |
| `MAX_RESIDENT_MODELS` | `0` (unlimited) | Set to `1` to load one model at a time and unload after use |

Measured peak VRAM in compare-all-3 mode on a GTX 1650:

| Configuration | Peak allocated | of 4096 MiB |
|---|---|---|
| Sharing on (default) | **1982 MiB** | comfortable |
| Sharing off | 2905 MiB | tight |
| `MAX_RESIDENT_MODELS=1`, sharing off | 1861 MiB | slowest; reloads per use |

A CUDA OOM falls back to CPU for that model with a named log line rather than
failing the whole comparison. Sharing is proven output-neutral by a test that
compares a shared-base `pretrained` against an independently loaded one.

---

## Scripts

```powershell
# Transcribe one file from the command line and assert the decode config
# matches across models.
python -m app.scripts.smoke --model all --file sample.mp4

# Seed the research benchmark (idempotent).
python -m app.scripts.seed_benchmark
python -m app.scripts.seed_benchmark --dry-run

# Reproduce the research predictions end to end.
python -m app.scripts.verify_against_research
python -m app.scripts.verify_against_research --sample-size 10 --mp4-check
```

`verify_against_research` is read-only against both the archive and the dataset,
and prints only identifiers, counts and scores — never transcript or reference
text.

---

## Verification

Five `test`-split recordings, matched to their audio by SHA-256 through the
frozen manifest (`utterance_id → m2_path → sha256(file) == audio_sha256`):

| Model | Exact matches | app WER / CER | saved WER / CER |
|---|---|---|---|
| `pretrained` | **5/5** | 63.40 / 30.47 | 63.40 / 30.47 |
| `full_ft` | **5/5** | 45.77 / 24.20 | 45.77 / 24.20 |
| `lora` | **5/5** | 56.86 / 24.01 | 56.86 / 24.01 |

All 45 test recordings resolve cleanly through the matching chain (0 hash
disagreements, 0 missing files).

---

## Tests

```powershell
cd backend
.venv\Scripts\activate

pytest -q                                   # 303 passed, 20 skipped
HDSD_RUN_MODEL_TESTS=1 pytest -q            # 323 passed (~3.5 min on a GTX 1650)
ruff check .
```

Frontend lint (ESLint 9 flat config, `frontend/eslint.config.js`):

```powershell
cd frontend
npm run lint                                # 0 errors, 10 warnings
```

The 10 warnings are `react-refresh/only-export-components` on `common.jsx` and
`AuthContext.jsx`, which deliberately export hooks and shared helpers next to
components — it costs those two files fast-refresh granularity, nothing else.

Tests run against in-memory SQLite by default — no database needed. Point
`TEST_DATABASE_URL` at a throwaway PostgreSQL database to exercise the real
dialect.

The 20 skipped tests load real checkpoints and need `HDSD_RUN_MODEL_TESTS=1`.
They are the ones that prove the saved full-FT `generation_config.json` really
is missing task tokens 50358/50359 (hence the base-snapshot reload), that all
three models resolve one identical decode config, and that the VRAM sharing does
not change output.

---

## Environment variables

Full list with comments in `backend/.env.example`. The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | — | Managed or local Postgres. `postgres://` and `?sslmode=require` are normalized for asyncpg automatically |
| `SECRET_KEY` | — | JWT signing. Generate a long random value |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 30 | |
| `REFRESH_TOKEN_EXPIRE_DAYS` | 7 | Refresh tokens rotate; a replayed token is rejected |
| `CORS_ORIGINS` | Vite dev origins | Comma-separated exact origins |
| `INFERENCE_ENABLED` | `true` | `false` on a host without weights: auth/benchmark/history work, transcription returns 503 |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `MAX_UPLOAD_MB` | 200 | Enforced while streaming, not after buffering |
| `MEDIA_RETENTION_DAYS` | 30 | Auto-delete window for uploaded media |
| `HDSD_DATASET_DIR` | — | Verification script only; the app never reads it |

---

## Security and privacy

- Upload type is decided by **content**, not filename: extension allow-list,
  then a magic-byte sniff, then ffprobe, which must find a decodable audio
  stream. A text file renamed `.wav` is rejected and the partial file removed.
- Uploads are stored outside any web root with UUID filenames; the client's
  filename never reaches the filesystem.
- Passwords are argon2. Refresh tokens are stored only as SHA-256 and rotate on
  use. Login is rate-limited per email+IP.
- A clinician sees only their own jobs; an admin sees everything. Someone else's
  job id returns **404, not 403** — whether an id exists is not information owed
  to a stranger.
- Transcript and reference text are **never logged**. `app/core/logging.py`
  documents the rule and provides `redact()`.
- Deleting a job removes its transcripts and the stored media file. The deleter
  refuses any path outside `UPLOAD_DIR`.

---

## Project layout

```
backend/
  app/
    api/v1/      auth, models, transcriptions, benchmark, admin, health
    core/        config, security, logging, rate limiting, research-asset guard
    db/          engine, session, base types, provider URL normalization
    models/      SQLAlchemy models (7 tables)
    schemas/     Pydantic request/response models
    scripts/     smoke, seed_benchmark, verify_against_research
    services/    transcription, audio, generation, metrics, normalization,
                 comparison, jobs, media, benchmark, auth
  alembic/       migrations (0001 initial, 0002 requested_model_key)
  tests/         303 tests
frontend/
  src/
    api/         axios client with refresh interceptor, endpoints
    components/  Layout, ProtectedRoute, Dropzone, ModelSelector, JobResults
    context/     AuthContext
    pages/       Login, Register, Transcribe, History, Benchmark
render.yaml      API-only deployment (see DEPLOYMENT.md)
frontend/vercel.json
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ffmpeg not found` | winget PATH not refreshed | New terminal, or set `FFMPEG_BINARY`/`FFPROBE_BINARY` to absolute paths |
| `InvalidPasswordError` on `/health` | wrong `DATABASE_URL` | Check the password; confirm the `hdsd` role exists |
| Startup fails naming a checkpoint | `FULL_FT_DIR`/`LORA_DIR` wrong | Fix the path in `.env` |
| `numpy` wheel build fails | Python 3.12/3.13 | Use Python 3.11 |
| CUDA not used | CPU torch installed | Reinstall torch from the cu118 index |
| CUDA out of memory | three FP32 models | Keep `SHARE_BASE_MODEL=true`, or set `MAX_RESIDENT_MODELS=1` |
| Frontend can't reach the API | backend not running | Start uvicorn on port 8000; the UI shows a banner |
| Registration rejects your email | `email-validator` rejects reserved TLDs | `.test`/`.invalid` are not valid; use a real domain form |

---

## Known limitations

See the end of this file's companion, `DEPLOYMENT.md`, for hosting analysis.

- **Single process.** The job queue is an in-process `asyncio.Queue` and the
  login rate limiter is an in-process dict. Both are correct for one uvicorn
  worker, which is what the GPU lock implies anyway. Multiple workers would need
  Redis or a real broker.
- **Tokens in `localStorage`.** Convenient and standard for a prototype, but
  readable by any XSS on the page. httpOnly cookies would be the hardening step.
- **Long audio is windowed, not diarized.** Fixed 30 s windows with no overlap
  by default; the training clips were short sentences, so quality on long
  recordings is untested.
- **No live speaker-macro WER.** The benchmark shows it from the research
  tables; live jobs are single recordings, so there is nothing to macro-average.
