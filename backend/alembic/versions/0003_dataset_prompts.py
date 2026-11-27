"""Add dataset_prompts: the 30 fixed HDSD sentences used as ground truth.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_prompts",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("prompt_id", sa.String(length=16), nullable=False),
        sa.Column("romanized", sa.Text(), nullable=False),
        sa.Column("devanagari", sa.Text(), nullable=False),
        sa.Column(
            "verified", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )
    op.create_index(
        "ix_dataset_prompts_prompt_id", "dataset_prompts", ["prompt_id"], unique=True
    )


def downgrade() -> None:
    op.drop_index("ix_dataset_prompts_prompt_id", table_name="dataset_prompts")
    op.drop_table("dataset_prompts")
