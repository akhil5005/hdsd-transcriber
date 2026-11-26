"""Unit tests for password hashing and token primitives."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


def test_password_hash_is_salted_and_verifiable() -> None:
    first = hash_password("same-password")
    second = hash_password("same-password")

    assert first != second, "argon2 must salt each hash"
    assert first.startswith("$argon2")
    assert verify_password("same-password", first)
    assert verify_password("same-password", second)


def test_verify_password_rejects_wrong_password_and_garbage_hash() -> None:
    hashed = hash_password("right")
    assert not verify_password("wrong", hashed)
    assert not verify_password("right", "not-a-hash")


def test_access_token_round_trip() -> None:
    token, expires_at = create_access_token(
        "11111111-1111-1111-1111-111111111111", role="clinician"
    )
    payload = decode_access_token(token)

    assert payload["sub"] == "11111111-1111-1111-1111-111111111111"
    assert payload["role"] == "clinician"
    assert payload["type"] == "access"
    assert payload["exp"] == int(expires_at.timestamp())


def test_expired_access_token_is_rejected() -> None:
    token, _ = create_access_token(
        "11111111-1111-1111-1111-111111111111",
        role="clinician",
        expires_delta=timedelta(seconds=-10),
    )
    with pytest.raises(TokenError):
        decode_access_token(token)


def test_tampered_access_token_is_rejected() -> None:
    token, _ = create_access_token("11111111-1111-1111-1111-111111111111", role="admin")
    head, payload, signature = token.split(".")
    tampered = f"{head}.{payload}.{signature[:-3]}aaa"

    with pytest.raises(TokenError):
        decode_access_token(tampered)


def test_refresh_token_is_random_and_only_the_hash_is_storable() -> None:
    raw_a, hash_a, expires_a = create_refresh_token()
    raw_b, hash_b, _ = create_refresh_token()

    assert raw_a != raw_b
    assert hash_a != hash_b
    assert hash_a == hash_refresh_token(raw_a)
    assert len(hash_a) == 64
    assert raw_a not in hash_a
    assert expires_a.tzinfo is not None
