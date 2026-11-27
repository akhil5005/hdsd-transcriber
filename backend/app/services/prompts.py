"""Ground truth for HDSD dataset recordings, looked up from the filename.

Every speaker reads the same 30 sentences, and the corpus encodes which one a
file holds in its name::

    M22_S1_H15_M1.wav
    ^^^    ^^^ ^^
    |      |   mic (M1 / M2)
    |      sentence id -> the prompt, and therefore the ground truth
    speaker

So an upload straight out of the dataset can be scored without the clinician
typing anything. A recording made in the browser has no such id and falls back
to whatever reference the user types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prompt import DatasetPrompt

#: Sentence id inside an HDSD filename. Anchored on the underscore-delimited
#: segment so an unrelated file that merely contains "h15" does not match.
_PROMPT_IN_FILENAME = re.compile(r"(?:^|_)(H\d{2})(?:_|\.)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PromptMatch:
    """A dataset prompt recognised from a filename."""

    prompt_id: str
    devanagari: str
    verified: bool


def prompt_id_from_filename(filename: str | None) -> str | None:
    """Extract the HDSD sentence id, or None if this is not a corpus file."""
    if not filename:
        return None
    match = _PROMPT_IN_FILENAME.search(filename)
    return match.group(1).upper() if match else None


async def lookup_ground_truth(
    db: AsyncSession, filename: str | None
) -> PromptMatch | None:
    """Ground truth for ``filename``, or None when it is not a dataset file."""
    prompt_id = prompt_id_from_filename(filename)
    if prompt_id is None:
        return None
    row = (
        await db.execute(
            select(DatasetPrompt).where(DatasetPrompt.prompt_id == prompt_id)
        )
    ).scalar_one_or_none()
    if row is None or not row.devanagari.strip():
        return None
    return PromptMatch(
        prompt_id=row.prompt_id,
        devanagari=row.devanagari,
        verified=row.verified,
    )
