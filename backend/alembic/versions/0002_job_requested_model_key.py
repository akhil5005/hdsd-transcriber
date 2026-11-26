"""add transcription_jobs.requested_model_key

A `single`-mode job has to remember which of the three models it was asked
for, or the worker cannot resume it after a restart. PROMPT.md section 6 does
not list this column; it is a deliberate, additive extension. NULL for
`compare` jobs, which always run all three.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "transcription_jobs",
        sa.Column("requested_model_key", sa.String(32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("transcription_jobs", "requested_model_key")
