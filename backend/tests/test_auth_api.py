"""End-to-end tests for the auth routes."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.core.rate_limit import login_limiter
from app.models import RefreshToken, User

AUTH = "/api/v1/auth"


# --- register ---------------------------------------------------------------
async def test_register_creates_clinician_and_normalizes_email(
    client, db, register_payload
) -> None:
    response = await client.post(f"{AUTH}/register", json=register_payload)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["email"] == "clinician@example.com"
    assert body["role"] == "clinician"
    assert body["is_active"] is True
    assert "hashed_password" not in body and "password" not in body

    stored = (await db.execute(select(User))).scalar_one()
    assert stored.hashed_password != register_payload["password"]


async def test_register_rejects_duplicate_email_case_insensitively(
    client, registered_user
) -> None:
    response = await client.post(
        f"{AUTH}/register",
        json={
            "email": "CLINICIAN@example.com",
            "full_name": "Someone Else",
            "password": "another-password-1",
        },
    )
    assert response.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "full_name": "A", "password": "longenough1"},
        {"email": "a@b.com", "full_name": "A", "password": "short"},
        {"email": "a@b.com", "full_name": "", "password": "longenough1"},
    ],
)
async def test_register_validation_errors(client, payload) -> None:
    response = await client.post(f"{AUTH}/register", json=payload)
    assert response.status_code == 422


# --- login ------------------------------------------------------------------
async def test_login_returns_token_pair_and_stores_only_the_hash(
    client, db, registered_user
) -> None:
    response = await client.post(
        f"{AUTH}/login",
        json={
            "email": registered_user["email"],
            "password": registered_user["password"],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"] and body["refresh_token"]
    assert body["user"]["email"] == "clinician@example.com"

    stored = (await db.execute(select(RefreshToken))).scalar_one()
    assert stored.token_hash != body["refresh_token"]
    assert stored.revoked_at is None


async def test_login_with_wrong_password_is_401(client, registered_user) -> None:
    response = await client.post(
        f"{AUTH}/login",
        json={"email": registered_user["email"], "password": "wrong-password"},
    )
    assert response.status_code == 401


async def test_login_for_unknown_email_is_401_with_the_same_detail(
    client, registered_user
) -> None:
    unknown = await client.post(
        f"{AUTH}/login", json={"email": "nobody@example.com", "password": "whatever1"}
    )
    wrong = await client.post(
        f"{AUTH}/login",
        json={"email": registered_user["email"], "password": "wrong-password"},
    )
    assert unknown.status_code == wrong.status_code == 401
    # Identical message: the response must not reveal whether the account exists.
    assert unknown.json()["detail"] == wrong.json()["detail"]


async def test_login_rate_limit_triggers_429_then_clears_on_success(
    client, registered_user
) -> None:
    for _ in range(login_limiter.max_attempts):
        bad = await client.post(
            f"{AUTH}/login",
            json={"email": registered_user["email"], "password": "wrong-password"},
        )
        assert bad.status_code == 401

    blocked = await client.post(
        f"{AUTH}/login",
        json={
            "email": registered_user["email"],
            "password": registered_user["password"],
        },
    )
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers

    login_limiter.reset()
    allowed = await client.post(
        f"{AUTH}/login",
        json={
            "email": registered_user["email"],
            "password": registered_user["password"],
        },
    )
    assert allowed.status_code == 200


async def test_deactivated_user_cannot_log_in(client, db, registered_user) -> None:
    user = (await db.execute(select(User))).scalar_one()
    user.is_active = False
    await db.commit()

    response = await client.post(
        f"{AUTH}/login",
        json={
            "email": registered_user["email"],
            "password": registered_user["password"],
        },
    )
    assert response.status_code == 403


# --- me ---------------------------------------------------------------------
async def test_me_requires_a_token(client) -> None:
    assert (await client.get(f"{AUTH}/me")).status_code == 401


async def test_me_rejects_a_garbage_token(client) -> None:
    response = await client.get(
        f"{AUTH}/me", headers={"Authorization": "Bearer not.a.jwt"}
    )
    assert response.status_code == 401


async def test_me_returns_the_current_user(client, tokens) -> None:
    access = tokens["access_token"]
    response = await client.get(
        f"{AUTH}/me", headers={"Authorization": f"Bearer {access}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "clinician@example.com"


async def test_me_rejects_a_refresh_token_used_as_a_bearer_token(client, tokens) -> None:
    refresh = tokens["refresh_token"]
    response = await client.get(
        f"{AUTH}/me", headers={"Authorization": f"Bearer {refresh}"}
    )
    assert response.status_code == 401


# --- refresh ----------------------------------------------------------------
async def test_refresh_rotates_the_token_and_revokes_the_old_one(
    client, db, tokens
) -> None:
    response = await client.post(
        f"{AUTH}/refresh", json={"refresh_token": tokens["refresh_token"]}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["refresh_token"] != tokens["refresh_token"]

    stored = (await db.execute(select(RefreshToken))).scalars().all()
    assert len(stored) == 2
    assert sum(1 for token in stored if token.revoked_at is not None) == 1


async def test_a_rotated_refresh_token_cannot_be_reused(client, tokens) -> None:
    first = await client.post(
        f"{AUTH}/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert first.status_code == 200

    replay = await client.post(
        f"{AUTH}/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert replay.status_code == 401


async def test_refresh_rejects_an_unknown_token(client) -> None:
    response = await client.post(
        f"{AUTH}/refresh", json={"refresh_token": "totally-made-up"}
    )
    assert response.status_code == 401


# --- logout -----------------------------------------------------------------
async def test_logout_revokes_the_refresh_token(client, tokens) -> None:
    response = await client.post(
        f"{AUTH}/logout", json={"refresh_token": tokens["refresh_token"]}
    )
    assert response.status_code == 204

    after = await client.post(
        f"{AUTH}/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert after.status_code == 401


async def test_logout_is_idempotent(client, tokens) -> None:
    for _ in range(2):
        response = await client.post(
            f"{AUTH}/logout", json={"refresh_token": tokens["refresh_token"]}
        )
        assert response.status_code == 204


# --- admin ------------------------------------------------------------------
async def test_clinician_cannot_reach_admin_routes(client, tokens) -> None:
    access = tokens["access_token"]
    response = await client.get(
        "/api/v1/admin/users", headers={"Authorization": f"Bearer {access}"}
    )
    assert response.status_code == 403


async def test_admin_can_list_and_deactivate_users(client, admin_user, tokens) -> None:
    login = await client.post(
        f"{AUTH}/login",
        json={"email": "admin@example.com", "password": "admin-password-123"},
    )
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    listed = await client.get("/api/v1/admin/users", headers=headers)
    assert listed.status_code == 200
    emails = {user["email"] for user in listed.json()}
    assert {"admin@example.com", "clinician@example.com"} <= emails

    target = next(
        user for user in listed.json() if user["email"] == "clinician@example.com"
    )
    deactivated = await client.post(
        f"/api/v1/admin/users/{target['id']}/deactivate", headers=headers
    )
    assert deactivated.status_code == 200
    assert deactivated.json()["is_active"] is False

    # The clinician's existing session is revoked along with the account.
    orphaned = await client.post(
        f"{AUTH}/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert orphaned.status_code in (401, 403)


async def test_admin_cannot_deactivate_itself(client, admin_user) -> None:
    login = await client.post(
        f"{AUTH}/login",
        json={"email": "admin@example.com", "password": "admin-password-123"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = await client.post(
        f"/api/v1/admin/users/{admin_user.id}/deactivate", headers=headers
    )
    assert response.status_code == 400
