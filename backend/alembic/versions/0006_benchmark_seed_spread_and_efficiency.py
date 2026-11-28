"""Persist model_summary.csv and efficiency.csv so /summary is DB-backed.

Both datasets used to be read from the research archive on every request to
``/api/v1/benchmark/summary``, which fails on any host without that archive.
They get a table each, matching the shape the readers already produce.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_seed_spread",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("model_key", sa.String(length=32), nullable=False),
        sa.Column("split", sa.String(length=64), nullable=False),
        sa.Column("seed_count", sa.Integer(), nullable=True),
        sa.Column("mean_wer", sa.Float(), nullable=True),
        sa.Column("sd_wer", sa.Float(), nullable=True),
        sa.Column("mean_cer", sa.Float(), nullable=True),
        sa.Column("sd_cer", sa.Float(), nullable=True),
        sa.Column("mean_substitutions", sa.Float(), nullable=True),
        sa.Column("mean_deletions", sa.Float(), nullable=True),
        sa.Column("mean_insertions", sa.Float(), nullable=True),
        sa.Column("reference_words", sa.Integer(), nullable=True),
        sa.UniqueConstraint("model_key", "split", name="uq_seed_spread_model_split"),
    )
    op.create_index(
        op.f("ix_benchmark_seed_spread_model_key"),
        "benchmark_seed_spread",
        ["model_key"],
    )
    op.create_index(
        op.f("ix_benchmark_seed_spread_split"), "benchmark_seed_spread", ["split"]
    )

    op.create_table(
        "benchmark_efficiency",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("model_key", sa.String(length=32), nullable=False),
        sa.Column("run", sa.String(length=128), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column(
            "is_selected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("trainable_parameters", sa.BigInteger(), nullable=False),
        sa.Column("checkpoint_bytes", sa.BigInteger(), nullable=True),
        sa.Column("adaptation_minutes", sa.Float(), nullable=True),
        sa.Column("adaptation_peak_gib", sa.Float(), nullable=True),
        sa.Column("inference_seconds_447", sa.Float(), nullable=True),
        sa.Column("checkpoint_scope", sa.String(length=64), nullable=False),
        sa.UniqueConstraint("model_key", "run", name="uq_efficiency_model_run"),
    )
    op.create_index(
        op.f("ix_benchmark_efficiency_model_key"),
        "benchmark_efficiency",
        ["model_key"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_benchmark_efficiency_model_key"), "benchmark_efficiency")
    op.drop_table("benchmark_efficiency")
    op.drop_index(op.f("ix_benchmark_seed_spread_split"), "benchmark_seed_spread")
    op.drop_index(op.f("ix_benchmark_seed_spread_model_key"), "benchmark_seed_spread")
    op.drop_table("benchmark_seed_spread")
