"""Managed-provider connection strings must be usable verbatim.

These cover the exact shapes Neon, Supabase and Render hand out, because
pasting one in unmodified is the normal path and asyncpg rejects libpq-only
query parameters with an opaque TypeError.
"""

from __future__ import annotations

import pytest

from app.db.url import is_sqlite, normalize_database_url


def test_neon_url_is_translated_for_asyncpg() -> None:
    url, connect_args = normalize_database_url(
        "postgresql://u:p@ep-cool-1.eu-central-1.aws.neon.tech/hdsd"
        "?sslmode=require&channel_binding=require"
    )

    assert url == "postgresql+asyncpg://u:p@ep-cool-1.eu-central-1.aws.neon.tech/hdsd"
    assert connect_args == {"ssl": True}


def test_render_heroku_postgres_scheme_is_upgraded() -> None:
    url, connect_args = normalize_database_url("postgres://u:p@host:5432/db")

    assert url == "postgresql+asyncpg://u:p@host:5432/db"
    assert connect_args == {}


def test_supabase_sslmode_require_sets_ssl() -> None:
    url, connect_args = normalize_database_url(
        "postgresql://postgres:p@db.abcdef.supabase.co:5432/postgres?sslmode=require"
    )

    assert url.startswith("postgresql+asyncpg://")
    assert "sslmode" not in url
    assert connect_args == {"ssl": True}


def test_sslmode_disable_turns_ssl_off_explicitly() -> None:
    _, connect_args = normalize_database_url(
        "postgresql://u:p@localhost:5432/db?sslmode=disable"
    )
    assert connect_args == {"ssl": False}


@pytest.mark.parametrize(
    "param",
    ["channel_binding=require", "sslrootcert=/x/ca.pem", "target_session_attrs=rw"],
)
def test_libpq_only_parameters_are_dropped(param) -> None:
    url, _ = normalize_database_url(f"postgresql://u:p@host/db?{param}")
    assert "?" not in url


def test_unknown_query_parameters_are_preserved() -> None:
    url, _ = normalize_database_url(
        "postgresql://u:p@host/db?application_name=hdsd&sslmode=require"
    )
    assert "application_name=hdsd" in url
    assert "sslmode" not in url


def test_an_explicit_driver_is_respected() -> None:
    url, connect_args = normalize_database_url("postgresql+psycopg://u:p@host/db")
    assert url == "postgresql+psycopg://u:p@host/db"
    # ssl translation is asyncpg-specific, so psycopg keeps libpq semantics.
    assert connect_args == {}


def test_explicit_asyncpg_driver_still_gets_ssl_translation() -> None:
    url, connect_args = normalize_database_url(
        "postgresql+asyncpg://u:p@host/db?sslmode=require"
    )
    assert url == "postgresql+asyncpg://u:p@host/db"
    assert connect_args == {"ssl": True}


def test_sqlite_urls_pass_through_untouched() -> None:
    original = "sqlite+aiosqlite:///:memory:"
    url, connect_args = normalize_database_url(original)

    assert url == original
    assert connect_args == {}
    assert is_sqlite(url)


def test_password_special_characters_survive_normalization() -> None:
    url, _ = normalize_database_url(
        "postgresql://u:p%40ss%2Fword@host/db?sslmode=require"
    )
    assert "p%40ss%2Fword" in url
