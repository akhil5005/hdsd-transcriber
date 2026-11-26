"""The three-model registry.

Rows describe *which* checkpoint each API ``model_key`` resolves to, plus the
efficiency figures reported in the paper. Seeded, not user-editable.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, StrEnumType
from app.models.enums import ModelKind


class ModelRegistry(Base):
    __tablename__ = "models_registry"

    key: Mapped[str] = mapped_column(sa.String(32), primary_key=True)
    display_name: Mapped[str] = mapped_column(sa.String(120), nullable=False)
    kind: Mapped[ModelKind] = mapped_column(StrEnumType(ModelKind, 20), nullable=False)
    learning_rate: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    #: Absolute checkpoint directory, or ``id@revision`` for the pretrained model.
    path_or_revision: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    trainable_params: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, default=0)
    checkpoint_bytes: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=True, server_default=sa.true()
    )
