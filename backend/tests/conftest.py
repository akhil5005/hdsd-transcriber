"""Test fixtures.

Tests run against an in-memory SQLite database by default so `pytest` needs no
Docker. Point ``TEST_DATABASE_URL`` at a throwaway PostgreSQL database to
exercise the real dialect:

    TEST_DATABASE_URL=postgresql+asyncpg://hdsd:hdsd@localhost:5432/hdsd_test
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

# Must be set before app.core.config is imported anywhere.
os.environ.setdefault(
    "DATABASE_URL", os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
)
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-used-in-production")
os.environ.setdefault("ENVIRONMENT", "development")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.api.deps import get_db  # noqa: E402
from app.core.rate_limit import login_limiter  # noqa: E402
from app.db.session import get_db as real_get_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base, User, UserRole  # noqa: E402

TEST_DATABASE_URL = os.environ["DATABASE_URL"]
_IS_SQLITE = TEST_DATABASE_URL.startswith("sqlite")


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def engine() -> AsyncIterator:
    kwargs: dict[str, object] = {}
    if _IS_SQLITE:
        # One shared in-memory database for the whole test, across connections.
        kwargs.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
    eng = create_async_engine(TEST_DATABASE_URL, **kwargs)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


@pytest.fixture(autouse=True)
def clear_rate_limiter() -> AsyncIterator[None]:
    login_limiter.reset()
    yield
    login_limiter.reset()


@pytest.fixture
async def client(session_factory) -> AsyncIterator[AsyncClient]:
    """An HTTP client bound to the app, with the DB dependency overridden.

    ``create_app`` is used without its lifespan so tests do not require the
    read-only research checkpoints to be present on the machine.
    """
    app = create_app()

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[real_get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


@pytest.fixture
def register_payload() -> dict[str, str]:
    return {
        "email": "Clinician@Example.COM",
        "full_name": "Test Clinician",
        "password": "correct-horse-battery",
    }


@pytest.fixture
async def registered_user(client, register_payload) -> dict[str, str]:
    response = await client.post("/api/v1/auth/register", json=register_payload)
    assert response.status_code == 201, response.text
    return register_payload


@pytest.fixture
async def tokens(client, registered_user) -> dict[str, str]:
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": registered_user["email"], "password": registered_user["password"]},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
async def admin_user(db) -> User:
    from app.core.security import hash_password

    user = User(
        email="admin@example.com",
        full_name="Test Admin",
        hashed_password=hash_password("admin-password-123"),
        role=UserRole.ADMIN,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
