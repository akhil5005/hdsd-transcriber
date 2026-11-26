"""Request/response models for models, jobs and transcriptions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import JobMode, JobStatus, ModelKind

#: What `POST /transcriptions` accepts for `model_key`.
RequestedModel = Literal["pretrained", "full_ft", "lora", "all"]


class ModelInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    key: str
    display_name: str
    kind: ModelKind
    learning_rate: float | None = None
    seed: int | None = None
    path_or_revision: str
    trainable_params: int
    checkpoint_bytes: int | None = None
    is_active: bool
    #: Live state, not stored: whether the weights are currently resident.
    loaded: bool = False


class ChunkOut(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    text: str


class DiffTokenOut(BaseModel):
    kind: str
    reference: str | None = None
    hypothesis: str | None = None


class TranscriptionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    model_key: str
    text_raw: str
    text_normalized: str
    chunks: list[ChunkOut] | None = None

    inference_ms: int | None = None
    rtf: float | None = None
    device: str | None = None
    gen_config_hash: str | None = None

    wer: float | None = None
    cer: float | None = None
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None


class AgreementRow(BaseModel):
    reference_model: str
    hypothesis_model: str
    wer: float | None = None
    substitutions: int
    deletions: int
    insertions: int
    note: str


class MediaOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    original_filename: str
    mime_type: str
    size_bytes: int
    duration_sec: float | None = None


class JobSummary(BaseModel):
    """Row shape for the history list."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    mode: JobMode
    status: JobStatus
    requested_model_key: str | None = None
    has_reference: bool
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
    media: MediaOut | None = None
    model_keys: list[str] = Field(default_factory=list)


class JobDetail(JobSummary):
    """Full job with results and the comparison block."""

    transcriptions: list[TranscriptionOut] = Field(default_factory=list)
    comparison: dict[str, Any] | None = None


class JobCreated(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    mode: JobMode
    model_keys: list[str]
    queue_depth: int


class JobPage(BaseModel):
    items: list[JobSummary]
    total: int
    limit: int
    offset: int


class LiveMetricsAggregate(BaseModel):
    """Aggregate live scores across every job that carried a reference."""

    model_key: str
    scored_jobs: int
    mean_wer: float | None = None
    mean_cer: float | None = None
    mean_rtf: float | None = None
    total_substitutions: int = 0
    total_deletions: int = 0
    total_insertions: int = 0
