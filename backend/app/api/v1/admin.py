"""Admin-only user management."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentAdmin, DbSession
from app.models.user import User
from app.schemas.auth import UserOut
from app.services import auth_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserOut])
async def list_users(_: CurrentAdmin, db: DbSession) -> list[UserOut]:
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return [UserOut.model_validate(user) for user in result.scalars()]


@router.post("/users/{user_id}/deactivate", response_model=UserOut)
async def deactivate_user(
    user_id: uuid.UUID, admin: CurrentAdmin, db: DbSession
) -> UserOut:
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own account",
        )

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.is_active = False
    await db.commit()
    # Existing sessions must not outlive the deactivation.
    await auth_service.revoke_all_refresh_tokens(db, user.id)
    await db.refresh(user)
    return UserOut.model_validate(user)
