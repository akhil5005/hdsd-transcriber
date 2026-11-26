"""Async engine, session factory and the FastAPI session dependency.

Pool sizing is deliberately small. The backend targets a free/shared host and
a serverless Postgres (Neon) that scales to zero, so we want few, short-lived
connections and ``pool_pre_ping`` to survive the provider dropping idle ones.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.db.url import is_sqlite, normalize_database_url


def _make_engine() -> AsyncEngine:
    url, connect_args = normalize_database_url(settings.database_url)

    kwargs: dict[str, object] = {
        "echo": settings.db_echo,
        "future": True,
        "connect_args": connect_args,
    }
    if not is_sqlite(url):
        kwargs.update(
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            # Serverless Postgres recycles idle connections; beat it to it.
            pool_recycle=settings.db_pool_recycle_seconds,
        )
    return create_async_engine(url, **kwargs)


engine: AsyncEngine = _make_engine()

SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a session per request and always close it."""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
