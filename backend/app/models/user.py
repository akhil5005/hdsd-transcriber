"""Users and refresh tokens."""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import (
    Base,
    CreatedAtMixin,
    StrEnumType,
    TimestampType,
    UUIDPrimaryKeyMixin,
    UUIDType,
)
from app.models.enums import UserRole


class User(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(sa.String(320), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    hashed_password: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        StrEnumType(UserRole, 20), nullable=False, default=UserRole.CLINICIAN
    )
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=True, server_default=sa.true()
    )

    refresh_tokens: Mapped[list[RefreshToken]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email} role={self.role}>"


class RefreshToken(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """A single issued refresh token.

    Only the SHA-256 hash of the token is stored, so a database leak does not
    hand out usable sessions.
    """

    __tablename__ = "refresh_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, sa.ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    token_hash: Mapped[str] = mapped_column(sa.String(64), unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(TimestampType, nullable=True)

    user: Mapped[User] = relationship(back_populates="refresh_tokens")
