"""Record whether a dataset reference was human-verified when the job ran.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "transcription_jobs",
        sa.Column("reference_verified", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("transcription_jobs", "reference_verified")
