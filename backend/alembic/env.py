"""Alembic environment, wired to the async engine and the app's metadata."""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Make `app` importable when alembic runs from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.db.url import normalize_database_url  # noqa: E402
from app.models import Base  # noqa: E402  (imports every model -> full metadata)

config = context.config

# A URL set by the caller (e.g. the migration test) wins; otherwise fall back
# to the application settings so `alembic upgrade head` needs no arguments.
_RAW_URL = config.get_main_option("sqlalchemy.url", None) or settings.database_url

# Migrations must normalize exactly like the app does (app/db/session.py).
# A managed provider hands out a libpq URL -- `postgres://`, a bare
# `postgresql://`, `?sslmode=require` -- and a bare scheme makes SQLAlchemy
# pick its default psycopg2 dialect, which is not installed on a host that
# only ships asyncpg. The start command runs `alembic upgrade head` before
# uvicorn, so getting this wrong fails the deploy before the app ever boots.
DATABASE_URL, CONNECT_ARGS = normalize_database_url(_RAW_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # The URL goes in here rather than through `set_main_option` so that a
    # password containing `%` is never handed to ConfigParser interpolation.
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = DATABASE_URL

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=CONNECT_ARGS,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
