"""The fixed corpus sentences, so the UI can show ground truth before a run.

Every speaker reads the same 30 sentences and the filename says which one, so
once a corpus file is chosen the frontend can display exactly what the models
will be scored against -- rather than leaving the clinician to wonder what
filled the reference in for them.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.models.prompt import DatasetPrompt

router = APIRouter(prefix="/prompts", tags=["prompts"])


class PromptOut(BaseModel):
    prompt_id: str
    romanized: str
    devanagari: str
    #: False until a human has confirmed the Devanagari; the UI says so.
    verified: bool


@router.get("/{prompt_id}", response_model=PromptOut)
async def get_prompt(prompt_id: str, user: CurrentUser, db: DbSession) -> PromptOut:
    """One corpus sentence by its id, e.g. ``H15``."""
    row = (
        await db.execute(
            select(DatasetPrompt).where(
                DatasetPrompt.prompt_id == prompt_id.upper()
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No corpus sentence with id {prompt_id!r}.",
        )
    return PromptOut.model_validate(row, from_attributes=True)
