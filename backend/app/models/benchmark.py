"""Static research benchmark figures, seeded from the final_v3 CSVs.

Three tables, one per source CSV, so that serving the benchmark never reads
the research archive: ``per_run.csv`` -> :class:`BenchmarkMetric`,
``model_summary.csv`` -> :class:`BenchmarkSeedSpread`, ``efficiency.csv`` ->
:class:`BenchmarkEfficiency`. Seeding is the only thing that touches those
files; a request is served entirely from the database, on hosts that have no
archive at all.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class BenchmarkMetric(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "benchmark_metrics"
    __table_args__ = (
        sa.UniqueConstraint("model_key", "run", "split", name="uq_benchmark_model_run_split"),
    )

    model_key: Mapped[str] = mapped_column(sa.String(32), index=True, nullable=False)
    #: Research run directory name, e.g. "full_lr5e-06_seed123".
    run: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    seed: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    is_selected: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )
    #: Evaluation case, e.g. "unfamiliar_speaker_unfamiliar_sentence". Cases
    #: differ in speakers and sentences and must never be pooled.
    split: Mapped[str] = mapped_column(sa.String(64), index=True, nullable=False)
    utterances: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)

    wer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    cer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    macro_speaker_wer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    substitutions: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    deletions: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    insertions: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    reference_units: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)


class BenchmarkSeedSpread(Base, UUIDPrimaryKeyMixin):
    """Mean +/- sample SD across training seeds, from ``model_summary.csv``.

    One row per (model_key, split): the spread is already aggregated over
    seeds, so unlike :class:`BenchmarkMetric` there is no ``run``.
    """

    __tablename__ = "benchmark_seed_spread"
    __table_args__ = (
        sa.UniqueConstraint("model_key", "split", name="uq_seed_spread_model_split"),
    )

    model_key: Mapped[str] = mapped_column(sa.String(32), index=True, nullable=False)
    split: Mapped[str] = mapped_column(sa.String(64), index=True, nullable=False)
    seed_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)

    mean_wer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    sd_wer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    mean_cer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    sd_cer: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    mean_substitutions: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    mean_deletions: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    mean_insertions: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    reference_words: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)


class BenchmarkEfficiency(Base, UUIDPrimaryKeyMixin):
    """Cost of each run, from ``efficiency.csv``. One row per (model_key, run)."""

    __tablename__ = "benchmark_efficiency"
    __table_args__ = (
        sa.UniqueConstraint("model_key", "run", name="uq_efficiency_model_run"),
    )

    model_key: Mapped[str] = mapped_column(sa.String(32), index=True, nullable=False)
    run: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    seed: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    is_selected: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )

    trainable_parameters: Mapped[int] = mapped_column(
        sa.BigInteger, nullable=False, default=0
    )
    checkpoint_bytes: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    adaptation_minutes: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    adaptation_peak_gib: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    inference_seconds_447: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    #: "full checkpoint" vs "adapter only" -- why the two sizes differ so much.
    checkpoint_scope: Mapped[str] = mapped_column(
        sa.String(64), nullable=False, default=""
    )
