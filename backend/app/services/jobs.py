"""Background transcription worker (PROMPT.md section 4.8).

A single asyncio worker draining an ``asyncio.Queue`` -- no Celery, no broker.
One worker is the right number, not a limitation: GPU access has to be
serialized anyway, so a second worker would only queue behind the same lock
while holding a second database session open.

The blocking torch work runs in a thread via ``asyncio.to_thread`` so a
multi-second decode never stalls the event loop that is serving status polls.

Each job gets its own database session. Request-scoped sessions are gone by
the time the worker picks the job up.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger, redact
from app.db.session import SessionLocal
from app.models.enums import JobMode, JobStatus, ModelKey
from app.models.media import MediaFile
from app.models.transcription import Transcription, TranscriptionJob
from app.services import comparison
from app.services.audio import AudioError
from app.services.normalization import normalize_text
from app.services.transcription import (
    InferenceDisabled,
    ModelLoadError,
    get_transcription_service,
)

logger = get_logger(__name__)

ALL_MODEL_KEYS = (
    ModelKey.PRETRAINED.value,
    ModelKey.FULL_FT.value,
    ModelKey.LORA.value,
)


def models_for(mode: JobMode, model_key: str | None) -> list[str]:
    """Which models a job should run."""
    if mode == JobMode.COMPARE:
        return list(ALL_MODEL_KEYS)
    if model_key not in ALL_MODEL_KEYS:
        raise ModelLoadError(f"Unknown model key {model_key!r}")
    return [model_key]


class TranscriptionWorker:
    """Owns the queue and the single consumer task."""

    def __init__(self) -> None:
        # Allocated in start(), never here: an asyncio.Queue binds itself to
        # the first running loop that touches it, and this object is a module
        # global created at import time. Binding it then would make every
        # later loop (a uvicorn reload, each test's loop) fail with
        # "bound to a different event loop".
        self._queue: asyncio.Queue[uuid.UUID] | None = None
        self._task: asyncio.Task | None = None
        self._stopping = False

    # --- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run(), name="transcription-worker")
        logger.info("Transcription worker started")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._queue = None
        logger.info("Transcription worker stopped")

    @property
    def depth(self) -> int:
        return 0 if self._queue is None else self._queue.qsize()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # --- enqueue ----------------------------------------------------------
    async def enqueue(self, job_id: uuid.UUID) -> None:
        """Hand a job to the worker.

        A job is already persisted as ``queued`` before this is called, so if
        the worker is not running the job is not lost -- the next startup picks
        it up via :meth:`recover_orphans`.
        """
        if self._queue is None:
            logger.warning(
                "Worker not running; job %s stays queued for startup recovery",
                job_id,
            )
            return
        await self._queue.put(job_id)
        logger.info("Queued job %s (depth %d)", job_id, self._queue.qsize())

    async def recover_orphans(self) -> dict[str, int]:
        """Reconcile jobs left behind by a previous process.

        ``queued`` jobs are simply re-queued. ``running`` jobs are failed: the
        process that owned them is gone, so their real state is unknowable and
        silently retrying could double-write results.
        """
        async with SessionLocal() as db:
            orphaned_running = (
                await db.execute(
                    select(TranscriptionJob.id).where(
                        TranscriptionJob.status == JobStatus.RUNNING
                    )
                )
            ).scalars().all()
            if orphaned_running:
                await db.execute(
                    update(TranscriptionJob)
                    .where(TranscriptionJob.id.in_(orphaned_running))
                    .values(
                        status=JobStatus.FAILED,
                        error="Interrupted by a server restart. Please retry.",
                        finished_at=datetime.now(UTC),
                    )
                )

            pending = (
                await db.execute(
                    select(TranscriptionJob.id)
                    .where(TranscriptionJob.status == JobStatus.QUEUED)
                    .order_by(TranscriptionJob.created_at)
                )
            ).scalars().all()
            await db.commit()

        for job_id in pending:
            await self.enqueue(job_id)

        if orphaned_running or pending:
            logger.info(
                "Recovered jobs: %d failed as interrupted, %d re-queued",
                len(orphaned_running),
                len(pending),
            )
        return {"failed": len(orphaned_running), "requeued": len(pending)}

    # --- consumer ---------------------------------------------------------
    async def _run(self) -> None:
        queue = self._queue
        assert queue is not None, "start() must allocate the queue first"
        while not self._stopping:
            job_id = await queue.get()
            try:
                await self._process(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - last-resort guard
                logger.exception("Worker crashed on job %s", job_id)
            finally:
                queue.task_done()

    async def _process(self, job_id: uuid.UUID) -> None:
        async with SessionLocal() as db:
            job = await db.get(TranscriptionJob, job_id)
            if job is None:
                logger.warning("Job %s vanished before processing", job_id)
                return
            if job.status != JobStatus.QUEUED:
                logger.info("Job %s is %s, not queued; skipping", job_id, job.status)
                return

            media = await db.get(MediaFile, job.media_id)
            if media is None:
                await self._fail(db, job, "The uploaded media no longer exists.")
                return

            job.status = JobStatus.RUNNING
            await db.commit()

            try:
                await self._transcribe(db, job, media)
            except (AudioError, InferenceDisabled, ModelLoadError) as exc:
                await self._fail(db, job, str(exc))
            except Exception as exc:  # pragma: no cover - unexpected
                logger.exception("Job %s failed", job_id)
                await self._fail(db, job, f"{type(exc).__name__}: {exc}")

    async def _transcribe(
        self, db: AsyncSession, job: TranscriptionJob, media: MediaFile
    ) -> None:
        service = get_transcription_service()
        keys = models_for(job.mode, job.requested_model_key)
        source = Path(media.stored_path)
        if not source.is_file():
            raise AudioError(f"Stored media is missing: {media.original_filename}")

        work = Path(tempfile.mkdtemp(prefix="hdsd-job-"))
        try:
            # ffmpeg conversion and inference both block; keep them off the loop.
            wav_path, info = await asyncio.to_thread(
                service.prepare_audio, source, work
            )
            if media.duration_sec is None:
                media.duration_sec = info.duration_sec

            transcripts: dict[str, str] = {}
            results = {}
            for key in keys:
                result = await asyncio.to_thread(service.transcribe_wav, key, wav_path)
                results[key] = result
                transcripts[key] = result.text_normalized

            reference = normalize_text(job.reference_text)
            scored = comparison.score_against_reference(reference, transcripts)

            for key, result in results.items():
                entry = scored.get(key)
                db.add(
                    Transcription(
                        job_id=job.id,
                        model_key=key,
                        text_raw=result.text_raw,
                        text_normalized=result.text_normalized,
                        chunks=result.chunks_as_json(),
                        inference_ms=result.inference_ms,
                        rtf=result.rtf,
                        device=result.device,
                        gen_config_hash=result.gen_config_hash,
                        wer=entry.scores.wer if entry else None,
                        cer=entry.scores.cer if entry else None,
                        substitutions=entry.scores.substitutions if entry else None,
                        deletions=entry.scores.deletions if entry else None,
                        insertions=entry.scores.insertions if entry else None,
                    )
                )

            job.status = JobStatus.DONE
            job.error = None
            job.finished_at = datetime.now(UTC)
            await db.commit()
            logger.info(
                "Job %s done: %d model(s), reference=%s",
                job.id,
                len(results),
                redact(job.reference_text),
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    async def _fail(
        self, db: AsyncSession, job: TranscriptionJob, message: str
    ) -> None:
        job.status = JobStatus.FAILED
        job.error = message
        job.finished_at = datetime.now(UTC)
        await db.commit()
        logger.warning("Job %s failed: %s", job.id, message)


#: Process-wide worker. Started and stopped by the FastAPI lifespan.
worker = TranscriptionWorker()


def get_worker() -> TranscriptionWorker:
    return worker
