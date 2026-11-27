"""The fixed HDSD sentence prompts, used as ground truth for dataset uploads.

Every speaker in the corpus reads the same 30 sentences (H01-H30), so a
recording's filename is enough to know what was *supposed* to be said. That
makes WER/CER available on a dataset upload without the clinician typing a
reference by hand.

Seeded from ``data/hdsd_prompts.csv`` by ``app.scripts.seed_prompts``. The
Devanagari was transcribed from the corpus's own romanised word labels, not
from any patient transcript: ``verified`` records whether a human has since
checked it, and the UI says so when it has not.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class DatasetPrompt(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "dataset_prompts"

    #: Sentence id as it appears in the filename, e.g. "H15".
    prompt_id: Mapped[str] = mapped_column(
        sa.String(16), unique=True, index=True, nullable=False
    )
    #: The corpus's own romanised label, kept so the Devanagari can be audited.
    romanized: Mapped[str] = mapped_column(sa.Text, nullable=False)
    #: Ground truth the models are scored against.
    devanagari: Mapped[str] = mapped_column(sa.Text, nullable=False)
    #: False until a human has confirmed the Devanagari matches the prompt.
    verified: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )
