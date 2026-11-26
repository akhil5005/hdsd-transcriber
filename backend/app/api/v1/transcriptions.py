"""Transcription job endpoints (sections 5b and 7).

Ownership is enforced on every read and delete: a clinician sees only their own
jobs, an admin sees everything. That check lives in one helper
(:func:`_load_job`) so no route can forget it.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import JobMode, JobStatus
from app.models.media import MediaFile
from app.models.transcription import Transcription, TranscriptionJob
from app.models.user import User
from app.schemas.transcription import (
    JobCreated,
    JobDetail,
    JobPage,
    JobSummary,
    LiveMetricsAggregate,
    MediaOut,
    RequestedModel,
    TranscriptionOut,
)
from app.services import comparison
from app.services import media as media_service
from app.services.jobs import ALL_MODEL_KEYS, get_worker
from app.services.normalization import normalize_text

logger = get_logger(__name__)

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])

#: A reference longer than this is almost certainly a paste error.
MAX_REFERENCE_CHARS = 20_000


def _job_statement():
    return select(TranscriptionJob).options(
        selectinload(TranscriptionJob.transcriptions),
        selectinload(TranscriptionJob.media),
    )


async def _load_job(
    db: AsyncSession, job_id: uuid.UUID, user: User
) -> TranscriptionJob:
    """Fetch a job the user is allowed to see, or 404.

    A job belonging to someone else returns 404 rather than 403: whether an id
    exists is itself information we do not owe the caller.
    """
    job = (
        await db.execute(_job_statement().where(TranscriptionJob.id == job_id))
    ).scalar_one_or_none()
    if job is None or (job.user_id != user.id and not user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )
    return job


def _summary(job: TranscriptionJob) -> JobSummary:
    return JobSummary(
        id=job.id,
        mode=job.mode,
        status=job.status,
        requested_model_key=job.requested_model_key,
        has_reference=bool(normalize_text(job.reference_text)),
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
        media=MediaOut.model_validate(job.media) if job.media else None,
        model_keys=sorted(t.model_key for t in job.transcriptions),
    )


def _detail(job: TranscriptionJob) -> JobDetail:
    transcripts = {t.model_key: t.text_normalized for t in job.transcriptions}
    block: dict[str, Any] | None = None
    if job.status == JobStatus.DONE and transcripts:
        block = comparison.summarize(transcripts, job.reference_text)

    return JobDetail(
        **_summary(job).model_dump(),
        transcriptions=[
            TranscriptionOut.model_validate(t) for t in job.transcriptions
        ],
        comparison=block,
    )


# --- create -----------------------------------------------------------------
@router.post("", response_model=JobCreated, status_code=status.HTTP_201_CREATED)
async def create_transcription(
    user: CurrentUser,
    db: DbSession,
    file: Annotated[UploadFile, File(description="Audio or video file.")],
    model_key: Annotated[RequestedModel, Form()] = "all",
    reference_text: Annotated[str | None, Form()] = None,
) -> JobCreated:
    """Upload a file and queue transcription.

    Returns immediately with a job id; the frontend polls
    ``GET /transcriptions/{job_id}``.
    """
    if not settings.inference_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Transcription is disabled on this host. Run the backend on a "
                "machine that has the model weights."
            ),
        )

    if reference_text is not None and len(reference_text) > MAX_REFERENCE_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Reference text exceeds {MAX_REFERENCE_CHARS} characters.",
        )

    try:
        stored = media_service.store_upload(file)
    except media_service.MediaTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except media_service.MediaError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc

    media = MediaFile(
        user_id=user.id,
        original_filename=stored.original_filename,
        stored_path=str(stored.path),
        mime_type=stored.mime_type,
        size_bytes=stored.size_bytes,
        duration_sec=stored.duration_sec,
        sha256=stored.sha256,
    )
    db.add(media)
    await db.flush()

    mode = JobMode.COMPARE if model_key == "all" else JobMode.SINGLE
    job = TranscriptionJob(
        user_id=user.id,
        media_id=media.id,
        mode=mode,
        status=JobStatus.QUEUED,
        requested_model_key=None if mode == JobMode.COMPARE else model_key,
        # Stored for scoring; never logged (see app/core/logging.py).
        reference_text=reference_text or None,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    worker = get_worker()
    await worker.enqueue(job.id)

    keys = list(ALL_MODEL_KEYS) if mode == JobMode.COMPARE else [model_key]
    return JobCreated(
        job_id=job.id,
        status=job.status,
        mode=job.mode,
        model_keys=keys,
        queue_depth=worker.depth,
    )


# --- read -------------------------------------------------------------------
@router.get("", response_model=JobPage)
async def list_transcriptions(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    model_key: str | None = Query(None, description="Filter by model."),
    job_status: JobStatus | None = Query(None, alias="status"),
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> JobPage:
    """The user's job history, newest first, with filters (section 5b)."""
    conditions = []
    if not user.is_admin:
        conditions.append(TranscriptionJob.user_id == user.id)
    if job_status is not None:
        conditions.append(TranscriptionJob.status == job_status)
    if created_after is not None:
        conditions.append(TranscriptionJob.created_at >= created_after)
    if created_before is not None:
        conditions.append(TranscriptionJob.created_at <= created_before)
    if model_key is not None:
        conditions.append(
            TranscriptionJob.id.in_(
                select(Transcription.job_id).where(
                    Transcription.model_key == model_key
                )
            )
        )

    total = (
        await db.execute(
            select(func.count()).select_from(TranscriptionJob).where(*conditions)
        )
    ).scalar_one()

    rows = (
        await db.execute(
            _job_statement()
            .where(*conditions)
            .order_by(TranscriptionJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()

    return JobPage(
        items=[_summary(job) for job in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/aggregate", response_model=list[LiveMetricsAggregate])
async def aggregate_live_metrics(
    user: CurrentUser, db: DbSession
) -> list[LiveMetricsAggregate]:
    """Mean live scores per model over every job that had a reference.

    Declared before ``/{job_id}`` so the literal path wins over the UUID
    converter.
    """
    conditions = [Transcription.wer.is_not(None)]
    if not user.is_admin:
        conditions.append(
            Transcription.job_id.in_(
                select(TranscriptionJob.id).where(
                    TranscriptionJob.user_id == user.id
                )
            )
        )

    rows = (
        await db.execute(
            select(
                Transcription.model_key,
                func.count().label("scored"),
                func.avg(Transcription.wer),
                func.avg(Transcription.cer),
                func.avg(Transcription.rtf),
                func.sum(Transcription.substitutions),
                func.sum(Transcription.deletions),
                func.sum(Transcription.insertions),
            )
            .where(*conditions)
            .group_by(Transcription.model_key)
            .order_by(Transcription.model_key)
        )
    ).all()

    return [
        LiveMetricsAggregate(
            model_key=key,
            scored_jobs=scored,
            mean_wer=float(mean_wer) if mean_wer is not None else None,
            mean_cer=float(mean_cer) if mean_cer is not None else None,
            mean_rtf=float(mean_rtf) if mean_rtf is not None else None,
            total_substitutions=int(subs or 0),
            total_deletions=int(dels or 0),
            total_insertions=int(ins or 0),
        )
        for key, scored, mean_wer, mean_cer, mean_rtf, subs, dels, ins in rows
    ]


@router.get("/{job_id}", response_model=JobDetail)
async def get_transcription(
    job_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> JobDetail:
    return _detail(await _load_job(db, job_id, user))


# --- export -----------------------------------------------------------------
@router.get("/{job_id}/export")
async def export_transcription(
    job_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    export_format: Annotated[
        Literal["txt", "json", "csv"], Query(alias="format")
    ] = "txt",
) -> Response:
    job = await _load_job(db, job_id, user)
    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is {job.status}; nothing to export yet.",
        )

    stem = f"transcript-{job.id}"
    rows = sorted(job.transcriptions, key=lambda t: t.model_key)

    if export_format == "txt":
        parts = [f"# {job.media.original_filename if job.media else stem}"]
        for row in rows:
            parts.append("")
            parts.append(f"## {row.model_key}")
            parts.append(row.text_raw or "(empty)")
        body = "\n".join(parts)
        return Response(
            content=body,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}.txt"'
            },
        )

    if export_format == "json":
        payload = {
            "job_id": str(job.id),
            "mode": str(job.mode),
            "created_at": job.created_at.isoformat(),
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "media": {
                "original_filename": job.media.original_filename if job.media else None,
                "duration_sec": job.media.duration_sec if job.media else None,
            },
            "transcriptions": [
                {
                    "model_key": row.model_key,
                    "text_raw": row.text_raw,
                    "text_normalized": row.text_normalized,
                    "chunks": row.chunks,
                    "inference_ms": row.inference_ms,
                    "rtf": row.rtf,
                    "device": row.device,
                    "gen_config_hash": row.gen_config_hash,
                    "wer": row.wer,
                    "cer": row.cer,
                    "substitutions": row.substitutions,
                    "deletions": row.deletions,
                    "insertions": row.insertions,
                }
                for row in rows
            ],
        }
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}.json"'
            },
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "model_key", "text_raw", "text_normalized", "inference_ms", "rtf",
            "device", "gen_config_hash", "wer", "cer",
            "substitutions", "deletions", "insertions",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.model_key, row.text_raw, row.text_normalized,
                row.inference_ms, row.rtf, row.device, row.gen_config_hash,
                row.wer, row.cer, row.substitutions, row.deletions, row.insertions,
            ]
        )
    # BOM so Excel opens Devanagari correctly.
    return Response(
        content="﻿" + buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'},
    )


# --- delete -----------------------------------------------------------------
@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transcription(
    job_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> Response:
    """Delete the job, its transcripts and the stored media file."""
    job = await _load_job(db, job_id, user)
    media = job.media

    await db.delete(job)
    await db.commit()

    if media is not None:
        # The DB row is gone either way; a missing file is not an error.
        media_service.delete_media(media.stored_path)
        remaining = (
            await db.execute(
                select(func.count())
                .select_from(TranscriptionJob)
                .where(TranscriptionJob.media_id == media.id)
            )
        ).scalar_one()
        if remaining == 0:
            stale = await db.get(MediaFile, media.id)
            if stale is not None:
                await db.delete(stale)
                await db.commit()

    logger.info("Deleted job %s for user %s", job_id, user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- retry ------------------------------------------------------------------
@router.post("/{job_id}/retry", response_model=JobCreated)
async def retry_transcription(
    job_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> JobCreated:
    """Re-queue a failed job against the same media."""
    job = await _load_job(db, job_id, user)
    if job.status != JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Only failed jobs can be retried; this one is {job.status}.",
        )
    if not settings.inference_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transcription is disabled on this host.",
        )

    for row in list(job.transcriptions):
        await db.delete(row)
    job.status = JobStatus.QUEUED
    job.error = None
    job.finished_at = None
    await db.commit()

    worker = get_worker()
    await worker.enqueue(job.id)
    keys = (
        list(ALL_MODEL_KEYS)
        if job.mode == JobMode.COMPARE
        else [job.requested_model_key or ""]
    )
    return JobCreated(
        job_id=job.id,
        status=job.status,
        mode=job.mode,
        model_keys=keys,
        queue_depth=worker.depth,
    )

