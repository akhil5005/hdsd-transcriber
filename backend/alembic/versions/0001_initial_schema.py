"""initial schema

Creates the seven tables described in PROMPT.md section 6: users,
refresh_tokens, models_registry, media_files, transcription_jobs,
transcriptions and benchmark_metrics.

Revision ID: 0001
Revises: None
Create Date: 2026-10-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Mirrors app.db.base: JSONB on PostgreSQL, plain JSON elsewhere.
JSON_COL = sa.JSON().with_variant(JSONB, "postgresql")
UUID_COL = sa.Uuid(as_uuid=True)
TS_COL = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID_COL, primary_key=True, nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", TS_COL, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "refresh_tokens",
        sa.Column("id", UUID_COL, primary_key=True, nullable=False),
        sa.Column("user_id", UUID_COL, nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", TS_COL, nullable=False),
        sa.Column("revoked_at", TS_COL, nullable=True),
        sa.Column("created_at", TS_COL, nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index(
        "ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True
    )

    op.create_table(
        "models_registry",
        sa.Column("key", sa.String(32), primary_key=True, nullable=False),
        sa.Column("display_name", sa.String(120), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("path_or_revision", sa.String(512), nullable=False),
        sa.Column("trainable_params", sa.BigInteger(), nullable=False),
        sa.Column("checkpoint_bytes", sa.BigInteger(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )

    op.create_table(
        "media_files",
        sa.Column("id", UUID_COL, primary_key=True, nullable=False),
        sa.Column("user_id", UUID_COL, nullable=False),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("stored_path", sa.String(1024), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", TS_COL, nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_media_files_user_id", "media_files", ["user_id"])
    op.create_index("ix_media_files_sha256", "media_files", ["sha256"])

    op.create_table(
        "transcription_jobs",
        sa.Column("id", UUID_COL, primary_key=True, nullable=False),
        sa.Column("user_id", UUID_COL, nullable=False),
        sa.Column("media_id", UUID_COL, nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("reference_text", sa.Text(), nullable=True),
        sa.Column("finished_at", TS_COL, nullable=True),
        sa.Column("created_at", TS_COL, nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["media_id"], ["media_files.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_transcription_jobs_user_id", "transcription_jobs", ["user_id"])
    op.create_index("ix_transcription_jobs_media_id", "transcription_jobs", ["media_id"])
    op.create_index("ix_transcription_jobs_status", "transcription_jobs", ["status"])

    op.create_table(
        "transcriptions",
        sa.Column("id", UUID_COL, primary_key=True, nullable=False),
        sa.Column("job_id", UUID_COL, nullable=False),
        sa.Column("model_key", sa.String(32), nullable=False),
        sa.Column("text_raw", sa.Text(), nullable=False),
        sa.Column("text_normalized", sa.Text(), nullable=False),
        sa.Column("chunks", JSON_COL, nullable=True),
        sa.Column("inference_ms", sa.Integer(), nullable=True),
        sa.Column("rtf", sa.Float(), nullable=True),
        sa.Column("device", sa.String(16), nullable=True),
        sa.Column("gen_config_hash", sa.String(64), nullable=True),
        sa.Column("wer", sa.Float(), nullable=True),
        sa.Column("cer", sa.Float(), nullable=True),
        sa.Column("substitutions", sa.Integer(), nullable=True),
        sa.Column("deletions", sa.Integer(), nullable=True),
        sa.Column("insertions", sa.Integer(), nullable=True),
        sa.Column("created_at", TS_COL, nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["job_id"], ["transcription_jobs.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("job_id", "model_key", name="uq_transcription_job_model"),
    )
    op.create_index("ix_transcriptions_job_id", "transcriptions", ["job_id"])

    op.create_table(
        "benchmark_metrics",
        sa.Column("id", UUID_COL, primary_key=True, nullable=False),
        sa.Column("model_key", sa.String(32), nullable=False),
        sa.Column("run", sa.String(128), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column(
            "is_selected", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("split", sa.String(64), nullable=False),
        sa.Column("utterances", sa.Integer(), nullable=True),
        sa.Column("wer", sa.Float(), nullable=True),
        sa.Column("cer", sa.Float(), nullable=True),
        sa.Column("macro_speaker_wer", sa.Float(), nullable=True),
        sa.Column("substitutions", sa.Integer(), nullable=True),
        sa.Column("deletions", sa.Integer(), nullable=True),
        sa.Column("insertions", sa.Integer(), nullable=True),
        sa.Column("reference_units", sa.Integer(), nullable=True),
        sa.UniqueConstraint(
            "model_key", "run", "split", name="uq_benchmark_model_run_split"
        ),
    )
    op.create_index("ix_benchmark_metrics_model_key", "benchmark_metrics", ["model_key"])
    op.create_index("ix_benchmark_metrics_split", "benchmark_metrics", ["split"])


def downgrade() -> None:
    op.drop_table("benchmark_metrics")
    op.drop_table("transcriptions")
    op.drop_table("transcription_jobs")
    op.drop_table("media_files")
    op.drop_table("models_registry")
    op.drop_table("refresh_tokens")
    op.drop_table("users")
