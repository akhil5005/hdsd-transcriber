# Build: HDSD Dysarthric-Speech Hindi Transcriber (FastAPI + React + PostgreSQL)

## 1. Goal

Build a full-stack web app where an authenticated user uploads an **MP4 video (or audio file) of a dysarthric speaker**, picks one of **three Whisper-small models** (or all three), and gets a **Hindi transcript in Devanagari script**. The app also shows **comparison metrics between the three models**, both from the saved research benchmark and live on the user's own uploads.

Ask me before deviating from the stack, the model choices, or the decoding settings below. Work in phases (Section 10). Run the tests at the end of each phase and stop for my review after each phase.

## 2. Tech stack (fixed)

- **Backend:** Python 3.11, FastAPI, Uvicorn, SQLAlchemy 2.x (async with asyncpg), Alembic, Pydantic v2, pydantic-settings.
- **ML:** `torch==2.7.1`, `transformers==4.57.1`, `peft==0.17.1`, `librosa==0.11.0`, `soundfile==0.13.1`, `accelerate==1.10.1`, `huggingface-hub==0.36.0`, `numpy==1.26.4`. These are the exact versions the models were trained and evaluated with. Do not upgrade them.
- **Media:** ffmpeg (system binary) to extract audio from MP4.
- **Database:** PostgreSQL 16, run through `docker-compose.yml`.
- **Auth:** JWT (access token about 30 min plus a refresh token about 7 days), passwords hashed with argon2 (`passlib[argon2]` or `argon2-cffi`).
- **Frontend:** React 18 in **JavaScript (not TypeScript)** with Vite, React Router, Axios, Recharts for charts, and plain CSS or Tailwind.
- **OS:** I develop on Windows. Make scripts and paths cross-platform (use `pathlib` and `.env`, and avoid bash-only commands in the README).

## 3. The three models (exact files; already on my disk)

The weights are already on my disk in my research folder. **Do NOT copy, move or duplicate them.** Read them directly from these absolute paths, which go in `.env` (forward slashes work on Windows):

```
HDSD_ROOT=<RESEARCH_ARCHIVE>/hdsd_simple/runs
FULL_FT_DIR=<RESEARCH_ARCHIVE>/hdsd_simple/runs/full_lr5e-06_seed123/best
LORA_DIR=<RESEARCH_ARCHIVE>/hdsd_simple/runs/lora_lr5e-05_seed2026/best
BENCHMARK_DIR=<RESEARCH_ARCHIVE>/hdsd_simple/runs/final_v3/tables
BASE_MODEL_ID=openai/whisper-small
BASE_MODEL_REVISION=973afd24965f72e36ca33b3055d56a652f456b4d
DEVICE=auto        # cuda if available, else cpu
```

At startup, check that `FULL_FT_DIR` contains `model.safetensors` and `LORA_DIR` contains `adapter_model.safetensors`. If either is missing, fail with a clear message naming the path.

**The research folder (`<RESEARCH_ARCHIVE>`) is READ-ONLY.** Never write, edit, move, rename or delete anything inside it, and don't create files there. From it, read only the two model folders above and the three CSVs in `BENCHMARK_DIR`. Do not open any other file there: the rest contains private patient transcripts.

| Key | Display name | Source | How to load |
|---|---|---|---|
| `pretrained` | Whisper-small (pretrained, OpenAI) | Hugging Face `openai/whisper-small`, **revision pinned** to `973afd24965f72e36ca33b3055d56a652f456b4d` | `WhisperForConditionalGeneration.from_pretrained(id, revision=...)` |
| `full_ft` | Full fine-tune (lr 5e-6, seed 123) | `FULL_FT_DIR` (contains `model.safetensors`, config, tokenizer, processor) | `WhisperForConditionalGeneration.from_pretrained(FULL_FT_DIR)` |
| `lora` | LoRA adapter (lr 5e-5, seed 2026, r=16, α=32, q/v) | `LORA_DIR` (contains `adapter_model.safetensors`, `adapter_config.json`, tokenizer, processor) | Load the pinned base model, then `PeftModel.from_pretrained(base, LORA_DIR)`. The adapter is **not** a standalone model. |

**Why these checkpoints:** the learning rates were already selected in the research (full 5e-6, LoRA 5e-5). The only remaining choice was the training seed, and I picked the seed with the lowest **development-set WER** (full: seed 123 with dev WER 65.15; LoRA: seed 2026 with dev WER 62.24). This avoids choosing on the test set. Make the selected seed a config value so I can switch it later.

## 4. Inference: must match the frozen research protocol

Implement a single `TranscriptionService` that loads each model **once** (lazily on first use, then cached) and exposes `transcribe(model_key, wav_path) -> result`.

1. **Audio pipeline:** use ffmpeg to convert the MP4 (or any audio/video) to **16 kHz, mono, 16-bit PCM WAV**, then load it with soundfile or librosa. Reject files with no audio stream with a clear error.
2. **Generation config (critical):** for **all three models**, load `GenerationConfig` from the **pinned base model snapshot**, not from the fine-tuned checkpoint folder. The saved full-FT checkpoint has an incomplete `suppress_tokens` list: it is missing task-token IDs 50358 and 50359. Then apply the shared settings: `language="hi"`, `task="transcribe"`, `return_timestamps=False`, `do_sample=False`, `num_beams=1` (greedy), and `max_new_tokens=96` per 30-second chunk. Set the Hindi/transcribe/no-timestamps decoder prefix and clear `forced_decoder_ids`.
3. **Pass `generation_config=model.generation_config` explicitly in every `generate()` call.** PEFT forwards `generate()` to the base model, so setting the config only on the wrapper is not enough. At startup, assert that the resolved generation config is identical across the three models, and log its SHA-256 hash.
4. **Precision:** FP32, no autocast, `model.eval()` and `torch.inference_mode()`. Set `torch.backends.cuda.matmul.allow_tf32=False` and `torch.backends.cudnn.allow_tf32=True`.
5. **Long files:** the training clips were short sentences. For audio longer than 30 s, split it into 30 s windows (a small overlap is fine, but keep it simple and deterministic), transcribe each window, and join the results. Store the per-chunk text too.
6. **Output normalization:** return the raw Devanagari text and a normalized version. To normalize, apply Unicode NFC, replace Unicode punctuation with spaces, and collapse whitespace. Do not transliterate.
7. Record for every run: model key, checkpoint path or revision, the generation-config hash, the device, audio duration, inference time in milliseconds, and the real-time factor (inference time ÷ audio duration).
8. Run inference outside the request thread, using a background job (FastAPI `BackgroundTasks` or a small `asyncio` worker with a queue; no Celery needed). The frontend polls the job status. GPU access must be serialized: one inference at a time, behind a lock.

## 5. Comparison metrics (two kinds)

### 5a. Research benchmark (static, seeded into the DB)

Seed a `benchmark_metrics` table by reading `per_run.csv`, `model_summary.csv` and `efficiency.csv` directly from `BENCHMARK_DIR` (read-only; do not copy them). Write an idempotent seed script. Store all seeds, but flag the selected seed for each model.

Values for the **selected checkpoints** (WER/CER in %, lower is better):

| Evaluation case (n recordings) | Pretrained WER / CER | Full FT s123 WER / CER | LoRA s2026 WER / CER |
|---|---|---|---|
| Unfamiliar speaker + unfamiliar sentence (45), **primary** | 71.51 / 43.69 | 57.82 / 32.05 | 65.92 / 34.34 |
| Unfamiliar speaker + familiar sentence (189) | 93.03 / 57.61 | 16.72 / 9.77 | 35.88 / 19.33 |
| Familiar speaker + unfamiliar sentence (213) | 75.17 / 43.05 | 61.75 / 34.17 | 61.93 / 32.97 |

Seed mean ± SD across 3 seeds (WER, primary case): pretrained 71.51 (single run); full 63.04 ± 6.97; LoRA 63.31 ± 3.18.

Efficiency: trainable parameters are 0 for pretrained, 240,582,912 for full, and 1,769,472 for LoRA. Checkpoint size is about 973 MB for full and about 12.7 MB for LoRA (the LoRA adapter also needs the base model). Peak training GPU memory was about 4.6 GiB for full and about 1.9 GiB for LoRA.

Also show substitutions, deletions and insertions, plus speaker-macro WER, from `per_run.csv`.

**Benchmark page:** grouped bar charts of WER and CER per model for each evaluation case, a seed mean ± SD view with error bars, an S/D/I stacked breakdown, and an efficiency table. Show this note under the charts: *"Exploratory research results on the HDSD dataset; the three evaluation cases differ in speakers and sentences and must not be pooled. Not a clinical assessment."*

### 5b. Live comparison on the user's upload

- **"Compare all 3" mode:** run the same file through all three models and show the transcripts **side by side**, with inference time and real-time factor for each model.
- **Optional reference transcript field** (Devanagari text). If the user provides one, compute **WER, CER, and substitution/deletion/insertion counts** for each model, using the **same normalization** as Section 4.6. WER counts words and CER counts Unicode code points including spaces; use Levenshtein alignment with tie priority substitution, then deletion, then insertion. Show a word-level diff highlight against the reference for each model. Implement the WER/CER code yourself or with `jiwer`, but unit-test it against hand-computed examples.
- If no reference is provided, show only the transcripts, timings, and pairwise agreement between models (WER of each model's output against each other model's output, labeled "agreement, not accuracy").
- **History page:** a user's past jobs with filters by model and date, plus aggregate live metrics over every job that had a reference.

## 6. Database schema (PostgreSQL, Alembic migrations)

- `users`: id (UUID), email (unique), full_name, hashed_password, role (`admin` | `clinician`), is_active, created_at.
- `refresh_tokens`: id, user_id, token_hash, expires_at, revoked_at.
- `models_registry`: key, display_name, kind (`pretrained` | `full` | `lora`), learning_rate, seed, path_or_revision, trainable_params, checkpoint_bytes, is_active.
- `media_files`: id, user_id, original_filename, stored_path, mime_type, size_bytes, duration_sec, sha256, created_at.
- `transcription_jobs`: id, user_id, media_id, mode (`single` | `compare`), status (`queued` | `running` | `done` | `failed`), error, reference_text (nullable), created_at, finished_at.
- `transcriptions`: id, job_id, model_key, text_raw, text_normalized, chunks (JSONB), inference_ms, rtf, device, gen_config_hash, wer, cer, substitutions, deletions, insertions (the metric columns are nullable).
- `benchmark_metrics`: model_key, run, seed, is_selected, split, utterances, wer, cer, macro_speaker_wer, substitutions, deletions, insertions, reference_units.

## 7. API (all under `/api/v1`; every route except auth and health requires a JWT)

- `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout`, `GET /auth/me`.
- `GET /models`: the three models with their metadata and whether each is loaded.
- `POST /transcriptions`: multipart upload with `file`, `model_key` (`pretrained` | `full_ft` | `lora` | `all`), and an optional `reference_text`. Returns `job_id`.
- `GET /transcriptions/{job_id}`: status plus results. `GET /transcriptions`: the user's history, paginated.
- `DELETE /transcriptions/{job_id}`: deletes the job and its stored media.
- `GET /benchmark`: the seeded metrics. `GET /benchmark/summary`: data shaped for the charts.
- `GET /transcriptions/{job_id}/export?format=txt|json|csv`: export the transcript.
- `GET /health`: database and model status.

Admin only: list users and deactivate a user.

## 8. Security and privacy (this is patient audio)

- Validate the upload type (mp4, mov, mkv, webm, wav, mp3, m4a, flac) by sniffing the content, not only the file extension. Set a configurable size limit (default 200 MB).
- Store uploads outside the web root with UUID file names. Users can access only their own jobs; admins can access everything.
- Rate-limit login attempts. Configure CORS for the Vite dev origin only. Keep secrets in `.env` and commit `.env.example` instead.
- Add a setting to auto-delete uploaded media after N days (default 30). Never log transcript text or reference text.
- Show a visible disclaimer in the UI: research prototype; not for diagnosis or clinical decisions.

## 9. Frontend pages (React, JavaScript)

Login and Register → **Transcribe** (drag-and-drop upload, model selector with 4 choices: Pretrained / Full FT / LoRA / Compare all 3, an optional reference textarea, a progress state, then results with a copy button and exports) → **Compare view** (three columns, diff highlighting, metric cards) → **Benchmark** (charts from Section 5a) → **History**. Use a font that renders Devanagari properly (for example Noto Sans Devanagari). Make the layout responsive, with protected routes and automatic token refresh in an Axios interceptor.

## 10. Phases (stop after each one for my review)

1. Scaffold the monorepo in the current folder (`backend/`, `frontend/`, `docker-compose.yml` for Postgres, `.gitignore` covering `.env`, `uploads/`, `node_modules/`, `.venv/`), config, the DB models, Alembic, auth, and auth tests.
2. `TranscriptionService` and the audio pipeline. Write a CLI smoke test, `python -m app.scripts.smoke --model all --file sample.mp4`, that prints the three transcripts and asserts that the generation-config hash is equal across models. Write unit tests for normalization and WER/CER.
3. Jobs, transcription endpoints, the comparison logic, and the benchmark seeding and endpoints.
4. The React frontend.
5. README (Windows-friendly setup: Docker Desktop for Postgres, ffmpeg install, Python venv, `npm install`, the env vars and model paths, and the CPU vs GPU note), a final end-to-end check, and a summary of anything that is not done.

## 11. Constraints

- Do not train, fine-tune, or modify any weights. This app is inference only.
- Do not change the decoding settings to "improve" scores.
- Never modify, copy, move or delete anything in `<RESEARCH_ARCHIVE>`. Read only the two model folders and the three benchmark CSVs named in Section 3.
- Never run destructive commands (`rm -rf`, `del /s`, `git push --force`, dropping databases other than this app's own) outside the project folder.
- If no GPU is present, run on CPU and say in the UI that it will be slower.
