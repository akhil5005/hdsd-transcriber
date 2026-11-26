"""Uploaded media files.

``stored_path`` points at a UUID-named file under ``UPLOAD_DIR``, outside any
web root. The original filename is kept only for display.
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin, UUIDType
from app.models.user import User


class MediaFile(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "media_files"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, sa.ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    original_filename: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    stored_path: Mapped[str] = mapped_column(sa.String(1024), nullable=False)
    mime_type: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    duration_sec: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    sha256: Mapped[str] = mapped_column(sa.String(64), index=True, nullable=False)

    user: Mapped[User] = relationship()
