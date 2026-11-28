"""ORM models. Importing this package registers every table on ``Base``."""

from app.db.base import Base
from app.models.benchmark import (
    BenchmarkEfficiency,
    BenchmarkMetric,
    BenchmarkSeedSpread,
)
from app.models.enums import (
    JobMode,
    JobStatus,
    ModelKey,
    ModelKind,
    UserRole,
)
from app.models.media import MediaFile
from app.models.prompt import DatasetPrompt
from app.models.registry import ModelRegistry
from app.models.transcription import Transcription, TranscriptionJob
from app.models.user import RefreshToken, User

__all__ = [
    "Base",
    "BenchmarkEfficiency",
    "BenchmarkMetric",
    "BenchmarkSeedSpread",
    "DatasetPrompt",
    "JobMode",
    "JobStatus",
    "MediaFile",
    "ModelKey",
    "ModelKind",
    "ModelRegistry",
    "RefreshToken",
    "Transcription",
    "TranscriptionJob",
    "User",
    "UserRole",
]
