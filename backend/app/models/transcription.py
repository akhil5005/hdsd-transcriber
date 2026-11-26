"""Transcription jobs and their per-model results."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import (
    JSONB_OR_JSON,
    Base,
    CreatedAtMixin,
    StrEnumType,
    TimestampType,
    UUIDPrimaryKeyMixin,
    UUIDType,
)
from app.models.enums import JobMode, JobStatus
from app.models.media import MediaFile
from app.models.user import User


class TranscriptionJob(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "transcription_jobs"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, sa.ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    media_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, sa.ForeignKey("media_files.id", ondelete="CASCADE"), index=True, nullable=False
    )
    mode: Mapped[JobMode] = mapped_column(StrEnumType(JobMode, 16), nullable=False)
    #: Which model a `single`-mode job asked for. NULL for `compare`, which
    #: always runs all three. PROMPT.md section 6 does not list this column,
    #: but without it a single-model job cannot survive a restart: the worker
    #: would have no way to recover its target after re-queueing.
    requested_model_key: Mapped[str | None] = mapped_column(
        sa.String(32), nullable=True
    )
    status: Mapped[JobStatus] = mapped_column(
        StrEnumType(JobStatus, 16), nullable=False, default=JobStatus.QUEUED, index=True
    )
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    #: Optional Devanagari ground truth supplied by the user. Never logged.
    reference_text: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TimestampType, nullable=True)

    user: Mapped[User] = relationship()
    media: Mapped[MediaFile] = relationship()
    transcriptions: Mapped[list[Transcription]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Transcription.model_key",
    )


class Transcription(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """One model's output for one job.

    The metric columns are populated only when the job carried a reference
    transcript; otherwise they stay NULL and the UI shows timings only.
    """

    __tablename__ = "transcriptions"
    __table_args__ = (
        sa.UniqueConstraint("job_id", "model_key", name="uq_transcription_job_model"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType,
        sa.ForeignKey("transcription_jobs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    model_key: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    text_raw: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    text_normalized: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    #: Per-30s-window text and offsets, so long files stay auditable.
    chunks: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB_OR_JSON, nullable=True)

    inference_ms: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    rtf: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    device: Mapped[str | None] = mapped_column(sa.String(16), nullable=True)
    #: SHA-256 of the resolved generation config; must match across models.
    gen_config_hash: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)

    wer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    cer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    substitutions: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    deletions: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    insertions: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)

    job: Mapped[TranscriptionJob] = relationship(back_populates="transcriptions")
