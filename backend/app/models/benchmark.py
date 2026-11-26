"""Static research benchmark figures, seeded from the final_v3 CSVs.

One row per (model_key, run, split). ``is_selected`` marks the seed chosen on
development-set WER, which is the one the live app actually runs.
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
