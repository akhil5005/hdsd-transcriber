"""Declarative base and shared column types.

Types are chosen so the same models run on PostgreSQL (production) and on
SQLite (fast unit tests): ``sa.Uuid`` maps to native ``uuid`` on PostgreSQL,
and ``JSONB_OR_JSON`` degrades to plain ``JSON`` off PostgreSQL.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: JSONB on PostgreSQL, generic JSON elsewhere (e.g. SQLite under pytest).
JSONB_OR_JSON = sa.JSON().with_variant(JSONB, "postgresql")

#: Timezone-aware UUID primary key / timestamp helpers.
UUIDType = sa.Uuid(as_uuid=True)
TimestampType = sa.DateTime(timezone=True)


EnumT = TypeVar("EnumT", bound=enum.Enum)


class StrEnumType(sa.types.TypeDecorator):
    """Persist a string-valued Enum by its ``value``, load it back as the Enum.

    Keeps column contents stable ("clinician", not "UserRole.CLINICIAN") and
    hands ORM attributes back as real enum members regardless of the Python
    version's ``Enum.__str__`` behaviour.
    """

    impl = sa.String
    cache_ok = True

    def __init__(self, enum_cls: type[EnumT], length: int = 32, **kwargs: Any) -> None:
        super().__init__(length=length, **kwargs)
        self.enum_cls = enum_cls

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return self.enum_cls(value).value

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return self.enum_cls(value)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, primary_key=True, default=uuid.uuid4
    )


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, server_default=sa.func.now()
    )
