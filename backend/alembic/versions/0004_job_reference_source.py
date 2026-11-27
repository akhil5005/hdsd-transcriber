"""Record where a job's reference came from: typed, or matched from the corpus.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "transcription_jobs",
        sa.Column("reference_source", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "transcription_jobs",
        sa.Column("reference_prompt_id", sa.String(length=16), nullable=True),
    )
    # Existing rows with a reference were all typed by hand; there was no
    # other way to supply one before this migration.
    op.execute(
        "UPDATE transcription_jobs SET reference_source = 'manual' "
        "WHERE reference_text IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("transcription_jobs", "reference_prompt_id")
    op.drop_column("transcription_jobs", "reference_source")
