"""Transcription job endpoints end-to-end (sections 5b, 7, 8).

The model layer is stubbed: these tests are about the job lifecycle, ownership,
comparison wiring and exports, and loading ~3 GB of real weights would make
them useless as a fast feedback loop. The real weights are exercised in
test_models_real.py and app/scripts/verify_against_research.py.
"""

from __future__ import annotations

import io
import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.models import JobStatus, MediaFile, Transcription, TranscriptionJob
from app.services.transcription import ChunkResult, TranscriptionResult

TRANSCRIPTS = "/api/v1/transcriptions"

#: Stub outputs, deliberately different so comparison has something to show.
STUB_TEXT = {
    "pretrained": "यह गाँव है",
    "full_ft": "यह घर है",
    "lora": "यह घर था",
}
REFERENCE = "यह घर है"


@pytest.fixture(scope="module")
def ffmpeg() -> str:
    binary = Settings().ffmpeg_binary
    try:
        result = subprocess.run([binary, "-version"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("ffmpeg not available")
    if result.returncode != 0:
        pytest.skip("ffmpeg not runnable")
    return binary


@pytest.fixture(scope="module")
def wav_bytes(ffmpeg, tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("media") / "tone.wav"
    result = subprocess.run(
        [
            ffmpeg, "-nostdin", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]
    return path.read_bytes()


@pytest.fixture(autouse=True)
def stub_models(monkeypatch, tmp_path, session_factory):
    """Replace the torch layer with deterministic fake transcripts.

    Also points the worker at the test session factory: ``_process`` opens its
    own session (request sessions are gone by then), which would otherwise hit
    the real database.
    """
    from app.core import config as config_module
    from app.services import jobs, transcription

    # Patch the live settings instance only. Setting the attribute on the
    # Settings *class* leaks across tests even after monkeypatch unwinds.
    monkeypatch.setattr(config_module.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(config_module.settings, "inference_enabled", True)

    class StubService:
        def prepare_audio(self, media_path: Path, work_dir=None):
            from app.services.audio import AudioInfo

            return media_path, AudioInfo(
                path=media_path,
                duration_sec=1.0,
                sample_rate=16000,
                channels=1,
                frames=16000,
            )

        def transcribe_wav(self, model_key: str, wav_path: Path):
            from app.services.normalization import normalize_text

            raw = STUB_TEXT[model_key]
            return TranscriptionResult(
                model_key=model_key,
                text_raw=raw,
                text_normalized=normalize_text(raw),
                chunks=(ChunkResult(0, 0.0, 1.0, raw),),
                checkpoint=f"stub://{model_key}",
                gen_config_hash="c" * 64,
                device="cpu",
                audio_duration_sec=1.0,
                inference_ms=120,
                rtf=0.12,
            )

        def loaded_keys(self):
            return []

        @property
        def shared_gen_config_hash(self):
            return None

        @staticmethod
        def peak_vram():
            return {"allocated_mib": 0.0, "reserved_mib": 0.0, "total_mib": 0.0}

    stub = StubService()
    monkeypatch.setattr(jobs, "get_transcription_service", lambda: stub)
    monkeypatch.setattr(transcription, "get_transcription_service", lambda c=None: stub)
    monkeypatch.setattr(jobs, "SessionLocal", session_factory)
    return stub


async def _upload(client, tokens, wav_bytes, **form) -> dict:
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    payload = {"model_key": form.pop("model_key", "all"), **form}
    response = await client.post(
        TRANSCRIPTS,
        headers=headers,
        files={"file": ("clip.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data=payload,
    )
    return response


# --- create -----------------------------------------------------------------
async def test_upload_requires_authentication(client, wav_bytes) -> None:
    response = await client.post(
        TRANSCRIPTS,
        files={"file": ("clip.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"model_key": "all"},
    )
    assert response.status_code == 401


async def test_compare_all_queues_one_job_for_three_models(
    client, tokens, wav_bytes, db
) -> None:
    response = await _upload(client, tokens, wav_bytes, model_key="all")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["mode"] == "compare"
    assert body["status"] == "queued"
    assert sorted(body["model_keys"]) == ["full_ft", "lora", "pretrained"]

    job = (await db.execute(select(TranscriptionJob))).scalar_one()
    assert job.requested_model_key is None


async def test_single_model_job_records_the_requested_key(
    client, tokens, wav_bytes, db
) -> None:
    response = await _upload(client, tokens, wav_bytes, model_key="full_ft")

    assert response.json()["mode"] == "single"
    job = (await db.execute(select(TranscriptionJob))).scalar_one()
    assert job.requested_model_key == "full_ft"


async def test_unknown_model_key_is_rejected(client, tokens, wav_bytes) -> None:
    response = await _upload(client, tokens, wav_bytes, model_key="whisper-large")
    assert response.status_code == 422


async def test_non_media_upload_is_refused_with_415(client, tokens) -> None:
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.post(
        TRANSCRIPTS,
        headers=headers,
        files={"file": ("notes.wav", io.BytesIO(b"plain text" * 20), "audio/wav")},
        data={"model_key": "all"},
    )
    assert response.status_code == 415


async def test_disallowed_extension_is_refused(client, tokens) -> None:
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.post(
        TRANSCRIPTS,
        headers=headers,
        files={"file": ("x.txt", io.BytesIO(b"hello"), "text/plain")},
        data={"model_key": "all"},
    )
    assert response.status_code == 415


async def test_overlong_reference_is_refused(client, tokens, wav_bytes) -> None:
    response = await _upload(
        client, tokens, wav_bytes, model_key="all", reference_text="क" * 20_001
    )
    assert response.status_code == 422


async def test_upload_is_refused_when_inference_is_disabled(
    client, tokens, wav_bytes, monkeypatch
) -> None:
    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "inference_enabled", False)
    response = await _upload(client, tokens, wav_bytes)
    assert response.status_code == 503


# --- worker -----------------------------------------------------------------
async def test_worker_completes_a_compare_job(client, tokens, wav_bytes, db) -> None:
    from app.services.jobs import TranscriptionWorker

    response = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = uuid.UUID(response.json()["job_id"])

    worker = TranscriptionWorker()
    await worker._process(job_id)

    rows = (
        await db.execute(
            select(Transcription).where(Transcription.job_id == job_id)
        )
    ).scalars().all()
    assert sorted(row.model_key for row in rows) == ["full_ft", "lora", "pretrained"]

    job = await db.get(TranscriptionJob, job_id)
    await db.refresh(job)
    assert job.status == JobStatus.DONE
    assert job.finished_at is not None
    assert job.error is None


async def test_worker_runs_only_the_requested_model_for_single_mode(
    client, tokens, wav_bytes, db
) -> None:
    from app.services.jobs import TranscriptionWorker

    response = await _upload(client, tokens, wav_bytes, model_key="lora")
    job_id = uuid.UUID(response.json()["job_id"])
    await TranscriptionWorker()._process(job_id)

    rows = (
        await db.execute(
            select(Transcription).where(Transcription.job_id == job_id)
        )
    ).scalars().all()
    assert [row.model_key for row in rows] == ["lora"]


async def test_worker_records_timings_and_config_hash(
    client, tokens, wav_bytes, db
) -> None:
    from app.services.jobs import TranscriptionWorker

    response = await _upload(client, tokens, wav_bytes, model_key="full_ft")
    job_id = uuid.UUID(response.json()["job_id"])
    await TranscriptionWorker()._process(job_id)

    row = (
        await db.execute(
            select(Transcription).where(Transcription.job_id == job_id)
        )
    ).scalar_one()
    assert row.inference_ms == 120
    assert row.rtf == pytest.approx(0.12)
    assert row.device == "cpu"
    assert row.gen_config_hash == "c" * 64
    assert row.chunks and row.chunks[0]["index"] == 0


async def test_worker_scores_against_a_reference(client, tokens, wav_bytes, db) -> None:
    from app.services.jobs import TranscriptionWorker

    response = await _upload(
        client, tokens, wav_bytes, model_key="all", reference_text=REFERENCE
    )
    job_id = uuid.UUID(response.json()["job_id"])
    await TranscriptionWorker()._process(job_id)

    rows = {
        row.model_key: row
        for row in (
            await db.execute(
                select(Transcription).where(Transcription.job_id == job_id)
            )
        ).scalars()
    }
    assert rows["full_ft"].wer == 0.0
    assert rows["full_ft"].cer == 0.0
    assert rows["pretrained"].wer == pytest.approx(100 / 3)
    assert rows["pretrained"].substitutions == 1


async def test_worker_leaves_metrics_null_without_a_reference(
    client, tokens, wav_bytes, db
) -> None:
    from app.services.jobs import TranscriptionWorker

    response = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = uuid.UUID(response.json()["job_id"])
    await TranscriptionWorker()._process(job_id)

    rows = (
        await db.execute(
            select(Transcription).where(Transcription.job_id == job_id)
        )
    ).scalars().all()
    assert all(row.wer is None and row.cer is None for row in rows)


async def test_worker_fails_the_job_when_media_is_gone(
    client, tokens, wav_bytes, db
) -> None:
    from app.services.jobs import TranscriptionWorker

    response = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = uuid.UUID(response.json()["job_id"])

    media = (await db.execute(select(MediaFile))).scalar_one()
    Path(media.stored_path).unlink()

    await TranscriptionWorker()._process(job_id)

    job = await db.get(TranscriptionJob, job_id)
    await db.refresh(job)
    assert job.status == JobStatus.FAILED
    assert "missing" in (job.error or "").lower()


async def test_recover_orphans_fails_running_and_requeues_queued(
    client, tokens, wav_bytes, db
) -> None:
    from app.services.jobs import TranscriptionWorker

    first = await _upload(client, tokens, wav_bytes, model_key="all")
    second = await _upload(client, tokens, wav_bytes, model_key="all")
    stuck = await db.get(TranscriptionJob, uuid.UUID(first.json()["job_id"]))
    stuck.status = JobStatus.RUNNING
    await db.commit()

    worker = TranscriptionWorker()
    await worker.start()
    try:
        result = await worker.recover_orphans()

        assert result == {"failed": 1, "requeued": 1}
        assert worker.depth >= 1
    finally:
        await worker.stop()

    await db.refresh(stuck)
    assert stuck.status == JobStatus.FAILED
    assert "restart" in (stuck.error or "").lower()
    assert uuid.UUID(second.json()["job_id"])


async def test_an_unstarted_worker_does_not_lose_the_job(client, tokens, wav_bytes, db):
    """The job row is already `queued`, so startup recovery will find it."""
    from app.services.jobs import TranscriptionWorker

    worker = TranscriptionWorker()
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    await worker.enqueue(uuid.UUID(created.json()["job_id"]))

    assert worker.depth == 0  # nothing queued in memory
    job = await db.get(TranscriptionJob, uuid.UUID(created.json()["job_id"]))
    assert job.status == JobStatus.QUEUED


def test_worker_queue_survives_a_second_event_loop() -> None:
    """Regression: the queue must not bind to the loop that imported it.

    Sync on purpose -- asyncio.run() cannot nest inside pytest-asyncio's loop,
    and the whole point is to drive two independent loops.
    """
    import asyncio as aio

    from app.services.jobs import worker as module_worker

    async def cycle() -> int:
        await module_worker.start()
        await module_worker.enqueue(uuid.uuid4())
        depth = module_worker.depth
        await module_worker.stop()
        return depth

    # Two separate loops, as a uvicorn reload or two test cases would create.
    assert aio.run(cycle()) == 1
    assert aio.run(cycle()) == 1


# --- read -------------------------------------------------------------------
async def test_detail_includes_the_comparison_block(
    client, tokens, wav_bytes
) -> None:
    from app.services.jobs import TranscriptionWorker

    created = await _upload(
        client, tokens, wav_bytes, model_key="all", reference_text=REFERENCE
    )
    job_id = created.json()["job_id"]
    await TranscriptionWorker()._process(uuid.UUID(job_id))

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    body = (await client.get(f"{TRANSCRIPTS}/{job_id}", headers=headers)).json()

    assert body["status"] == "done"
    assert body["has_reference"] is True
    assert body["comparison"]["lowest_wer_model"] == "full_ft"
    assert body["comparison"]["scores"]["pretrained"]["diff"]
    assert len(body["transcriptions"]) == 3


async def test_detail_reports_agreement_without_a_reference(
    client, tokens, wav_bytes
) -> None:
    from app.services.jobs import TranscriptionWorker

    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]
    await TranscriptionWorker()._process(uuid.UUID(job_id))

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    body = (await client.get(f"{TRANSCRIPTS}/{job_id}", headers=headers)).json()

    assert body["has_reference"] is False
    assert body["comparison"]["scores"] == {}
    assert len(body["comparison"]["agreement"]) == 6
    assert body["comparison"]["agreement_note"] == "agreement, not accuracy"


async def test_history_lists_the_users_jobs_newest_first(
    client, tokens, wav_bytes
) -> None:
    for key in ("full_ft", "lora"):
        await _upload(client, tokens, wav_bytes, model_key=key)

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    body = (await client.get(TRANSCRIPTS, headers=headers)).json()

    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["created_at"] >= body["items"][1]["created_at"]


async def test_history_paginates(client, tokens, wav_bytes) -> None:
    for _ in range(3):
        await _upload(client, tokens, wav_bytes, model_key="full_ft")

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    page = (
        await client.get(TRANSCRIPTS, params={"limit": 2}, headers=headers)
    ).json()

    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["limit"] == 2


async def test_history_filters_by_status(client, tokens, wav_bytes) -> None:
    await _upload(client, tokens, wav_bytes, model_key="full_ft")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    queued = (
        await client.get(TRANSCRIPTS, params={"status": "queued"}, headers=headers)
    ).json()
    done = (
        await client.get(TRANSCRIPTS, params={"status": "done"}, headers=headers)
    ).json()

    assert queued["total"] == 1
    assert done["total"] == 0


async def test_history_filters_by_model(client, tokens, wav_bytes) -> None:
    from app.services.jobs import TranscriptionWorker

    created = await _upload(client, tokens, wav_bytes, model_key="lora")
    await TranscriptionWorker()._process(uuid.UUID(created.json()["job_id"]))

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    hit = (
        await client.get(TRANSCRIPTS, params={"model_key": "lora"}, headers=headers)
    ).json()
    miss = (
        await client.get(
            TRANSCRIPTS, params={"model_key": "pretrained"}, headers=headers
        )
    ).json()

    assert hit["total"] == 1
    assert miss["total"] == 0


# --- ownership --------------------------------------------------------------
async def test_another_user_cannot_read_a_job(
    client, tokens, wav_bytes, db
) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]

    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "other@example.com",
            "full_name": "Other Clinician",
            "password": "another-password-1",
        },
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "other@example.com", "password": "another-password-1"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    # 404, not 403: whether the id exists is not the stranger's business.
    assert (await client.get(f"{TRANSCRIPTS}/{job_id}", headers=headers)).status_code == 404
    assert (await client.get(TRANSCRIPTS, headers=headers)).json()["total"] == 0


async def test_admin_can_read_any_job(client, tokens, wav_bytes, admin_user) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "admin-password-123"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    assert (await client.get(f"{TRANSCRIPTS}/{job_id}", headers=headers)).status_code == 200


async def test_unknown_job_id_is_404(client, tokens) -> None:
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.get(f"{TRANSCRIPTS}/{uuid.uuid4()}", headers=headers)
    assert response.status_code == 404


# --- delete -----------------------------------------------------------------
async def test_delete_removes_the_job_and_the_stored_file(
    client, tokens, wav_bytes, db
) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]
    media = (await db.execute(select(MediaFile))).scalar_one()
    stored = Path(media.stored_path)
    assert stored.is_file()

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.delete(f"{TRANSCRIPTS}/{job_id}", headers=headers)

    assert response.status_code == 204
    assert not stored.exists()
    assert (await client.get(f"{TRANSCRIPTS}/{job_id}", headers=headers)).status_code == 404


async def test_delete_cascades_to_transcriptions(
    client, tokens, wav_bytes, db
) -> None:
    from app.services.jobs import TranscriptionWorker

    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]
    await TranscriptionWorker()._process(uuid.UUID(job_id))

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    await client.delete(f"{TRANSCRIPTS}/{job_id}", headers=headers)

    remaining = (await db.execute(select(Transcription))).scalars().all()
    assert remaining == []


async def test_another_user_cannot_delete_a_job(client, tokens, wav_bytes) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]

    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "thief@example.com",
            "full_name": "Thief",
            "password": "another-password-1",
        },
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "thief@example.com", "password": "another-password-1"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    assert (
        await client.delete(f"{TRANSCRIPTS}/{job_id}", headers=headers)
    ).status_code == 404


# --- export -----------------------------------------------------------------
@pytest.mark.parametrize(
    ("fmt", "content_type"),
    [
        ("txt", "text/plain"),
        ("json", "application/json"),
        ("csv", "text/csv"),
    ],
)
async def test_export_formats(client, tokens, wav_bytes, fmt, content_type) -> None:
    from app.services.jobs import TranscriptionWorker

    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]
    await TranscriptionWorker()._process(uuid.UUID(job_id))

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.get(
        f"{TRANSCRIPTS}/{job_id}/export", params={"format": fmt}, headers=headers
    )

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    assert STUB_TEXT["full_ft"] in response.text


async def test_csv_export_starts_with_a_bom_for_excel(
    client, tokens, wav_bytes
) -> None:
    from app.services.jobs import TranscriptionWorker

    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]
    await TranscriptionWorker()._process(uuid.UUID(job_id))

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.get(
        f"{TRANSCRIPTS}/{job_id}/export", params={"format": "csv"}, headers=headers
    )
    assert response.text.startswith("﻿")


async def test_export_of_an_unfinished_job_is_409(client, tokens, wav_bytes) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.get(
        f"{TRANSCRIPTS}/{job_id}/export", params={"format": "txt"}, headers=headers
    )
    assert response.status_code == 409


async def test_unknown_export_format_is_422(client, tokens, wav_bytes) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.get(
        f"{TRANSCRIPTS}/{job_id}/export", params={"format": "pdf"}, headers=headers
    )
    assert response.status_code == 422


# --- retry ------------------------------------------------------------------
async def test_retry_requeues_a_failed_job(client, tokens, wav_bytes, db) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]

    job = await db.get(TranscriptionJob, uuid.UUID(job_id))
    job.status = JobStatus.FAILED
    job.error = "boom"
    await db.commit()

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.post(f"{TRANSCRIPTS}/{job_id}/retry", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    await db.refresh(job)
    assert job.error is None


async def test_retry_of_a_queued_job_is_409(client, tokens, wav_bytes) -> None:
    created = await _upload(client, tokens, wav_bytes, model_key="all")
    job_id = created.json()["job_id"]

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = await client.post(f"{TRANSCRIPTS}/{job_id}/retry", headers=headers)
    assert response.status_code == 409


# --- aggregate --------------------------------------------------------------
async def test_aggregate_covers_only_jobs_with_a_reference(
    client, tokens, wav_bytes
) -> None:
    from app.services.jobs import TranscriptionWorker

    scored = await _upload(
        client, tokens, wav_bytes, model_key="all", reference_text=REFERENCE
    )
    unscored = await _upload(client, tokens, wav_bytes, model_key="all")
    for created in (scored, unscored):
        await TranscriptionWorker()._process(uuid.UUID(created.json()["job_id"]))

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = (await client.get(f"{TRANSCRIPTS}/aggregate", headers=headers)).json()

    by_model = {row["model_key"]: row for row in rows}
    assert by_model["full_ft"]["scored_jobs"] == 1
    assert by_model["full_ft"]["mean_wer"] == 0.0
    assert by_model["pretrained"]["mean_wer"] == pytest.approx(100 / 3)


async def test_aggregate_is_empty_before_any_scored_job(client, tokens) -> None:
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    assert (await client.get(f"{TRANSCRIPTS}/aggregate", headers=headers)).json() == []


# --- models -----------------------------------------------------------------
async def test_models_endpoint_lists_three_models_in_order(client, tokens) -> None:
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = (await client.get("/api/v1/models", headers=headers)).json()

    assert [row["key"] for row in rows] == ["pretrained", "full_ft", "lora"]
    assert all(row["loaded"] is False for row in rows)
    assert "973afd24" in rows[0]["path_or_revision"]


async def test_models_endpoint_requires_authentication(client) -> None:
    assert (await client.get("/api/v1/models")).status_code == 401
