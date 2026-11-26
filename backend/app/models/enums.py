"""Enumerations shared by the ORM models and the API schemas.

Stored as short strings rather than native PostgreSQL enums so that adding a
value later is a code change, not a migration with a type rewrite. The
columns use :class:`app.db.base.StrEnumType`, which persists ``value`` and
reads back a real enum member.
"""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "admin"
    CLINICIAN = "clinician"


class ModelKind(StrEnum):
    PRETRAINED = "pretrained"
    FULL = "full"
    LORA = "lora"


class ModelKey(StrEnum):
    """Public identifiers used by the API and the frontend."""

    PRETRAINED = "pretrained"
    FULL_FT = "full_ft"
    LORA = "lora"


class JobMode(StrEnum):
    SINGLE = "single"
    COMPARE = "compare"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
