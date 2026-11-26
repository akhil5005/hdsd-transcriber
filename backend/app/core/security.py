"""Password hashing and JWT issuing/verification.

Access tokens are short-lived bearer JWTs. Refresh tokens are opaque random
strings; only their SHA-256 hash reaches the database.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import settings

_hasher = PasswordHasher()

TokenType = Literal["access", "refresh"]


# --- Passwords --------------------------------------------------------------
def hash_password(plain_password: str) -> str:
    return _hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return _hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(hashed_password: str) -> bool:
    try:
        return _hasher.check_needs_rehash(hashed_password)
    except InvalidHashError:
        return True


# --- Access tokens ----------------------------------------------------------
class TokenError(Exception):
    """The presented token is missing, malformed, expired or of the wrong type."""


def create_access_token(
    subject: str | uuid.UUID,
    *,
    role: str,
    expires_delta: timedelta | None = None,
) -> tuple[str, datetime]:
    """Return ``(token, expires_at)`` for a signed access JWT."""
    now = datetime.now(UTC)
    expires_at = now + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    return token, expires_at


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate an access token, or raise :class:`TokenError`."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Token is invalid") from exc

    if payload.get("type") != "access":
        raise TokenError("Expected an access token")
    return payload


# --- Refresh tokens ---------------------------------------------------------
def create_refresh_token() -> tuple[str, str, datetime]:
    """Return ``(raw_token, token_hash, expires_at)``.

    The raw token is handed to the client exactly once; only the hash is
    persisted.
    """
    raw = secrets.token_urlsafe(48)
    expires_at = datetime.now(UTC) + timedelta(
        days=settings.refresh_token_expire_days
    )
    return raw, hash_refresh_token(raw), expires_at


def hash_refresh_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
