"""Authentication use-cases: register, login, refresh, logout.

Routes stay thin; all token bookkeeping lives here.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security
from app.core.logging import get_logger
from app.models.enums import UserRole
from app.models.user import RefreshToken, User

logger = get_logger(__name__)


class AuthError(Exception):
    """Base class for authentication failures."""


class EmailAlreadyRegistered(AuthError):
    pass


class InvalidCredentials(AuthError):
    pass


class InactiveUser(AuthError):
    pass


class InvalidRefreshToken(AuthError):
    pass


def normalize_email(email: str) -> str:
    """Emails are matched case-insensitively on the local part too.

    Lower-casing the whole address is the pragmatic choice here: it keeps the
    unique index meaningful and matches what every mail provider our users
    have actually does.
    """
    return email.strip().lower()


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == normalize_email(email)))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await db.get(User, user_id)


async def register_user(
    db: AsyncSession,
    *,
    email: str,
    full_name: str,
    password: str,
    role: UserRole = UserRole.CLINICIAN,
) -> User:
    email = normalize_email(email)
    if await get_user_by_email(db, email) is not None:
        raise EmailAlreadyRegistered(email)

    user = User(
        email=email,
        full_name=full_name.strip(),
        hashed_password=security.hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("Registered user %s (role=%s)", user.id, user.role)
    return user


async def authenticate(db: AsyncSession, *, email: str, password: str) -> User:
    user = await get_user_by_email(db, email)
    if user is None:
        # Spend a comparable amount of time so a missing account is not
        # distinguishable from a wrong password by timing.
        security.verify_password(password, security.hash_password("timing-equalizer"))
        raise InvalidCredentials()
    if not security.verify_password(password, user.hashed_password):
        raise InvalidCredentials()
    if not user.is_active:
        raise InactiveUser()
    return user


async def issue_token_pair(db: AsyncSession, user: User) -> tuple[str, str, datetime]:
    """Create an access token plus a fresh stored refresh token."""
    access_token, access_expires_at = security.create_access_token(
        user.id, role=user.role.value
    )
    raw_refresh, token_hash, refresh_expires_at = security.create_refresh_token()
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=refresh_expires_at,
        )
    )
    await db.commit()
    return access_token, raw_refresh, access_expires_at


async def _load_active_refresh_token(db: AsyncSession, raw_token: str) -> RefreshToken:
    token_hash = security.hash_refresh_token(raw_token)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    stored = result.scalar_one_or_none()
    if stored is None or stored.revoked_at is not None:
        raise InvalidRefreshToken()

    expires_at = stored.expires_at
    if expires_at.tzinfo is None:  # SQLite round-trips naive datetimes
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        raise InvalidRefreshToken()
    return stored


async def rotate_refresh_token(
    db: AsyncSession, raw_token: str
) -> tuple[User, str, str, datetime]:
    """Exchange a refresh token for a new pair, revoking the old token.

    Rotation means a stolen refresh token is usable at most once, and the
    legitimate client's next refresh fails loudly instead of silently sharing
    a session.
    """
    stored = await _load_active_refresh_token(db, raw_token)

    user = await get_user_by_id(db, stored.user_id)
    if user is None or not user.is_active:
        raise InactiveUser()

    stored.revoked_at = datetime.now(UTC)
    await db.flush()

    access_token, raw_refresh, access_expires_at = await issue_token_pair(db, user)
    return user, access_token, raw_refresh, access_expires_at


async def revoke_refresh_token(db: AsyncSession, raw_token: str) -> None:
    """Revoke one refresh token. Idempotent: unknown tokens are a no-op."""
    token_hash = security.hash_refresh_token(raw_token)
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.token_hash == token_hash, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    await db.commit()


async def revoke_all_refresh_tokens(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Used when an admin deactivates a user."""
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    await db.commit()
