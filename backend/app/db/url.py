"""Normalize a hosted-Postgres connection string for SQLAlchemy + asyncpg.

Managed providers hand out libpq-style URLs that asyncpg does not accept
verbatim:

* Render/Heroku use the ``postgres://`` scheme, which SQLAlchemy dropped.
* Neon and Supabase append ``?sslmode=require`` (and Neon also
  ``&channel_binding=require``). Those are libpq parameters; asyncpg rejects
  them as unexpected keyword arguments.

:func:`normalize_database_url` turns any of those into a URL SQLAlchemy can
use plus the ``connect_args`` that carry the TLS intent across.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: libpq query parameters asyncpg does not understand. ``sslmode`` is
#: translated into ``connect_args["ssl"]``; the rest are simply dropped.
_LIBPQ_ONLY_PARAMS = frozenset(
    {"sslmode", "channel_binding", "sslrootcert", "sslcert", "sslkey", "target_session_attrs"}
)

#: sslmode values that mean "use TLS".
_TLS_MODES = frozenset({"require", "verify-ca", "verify-full", "prefer", "allow"})

_ASYNC_DRIVER = "postgresql+asyncpg"


def normalize_database_url(url: str) -> tuple[str, dict[str, object]]:
    """Return ``(sqlalchemy_url, connect_args)`` for ``url``.

    Non-PostgreSQL URLs (notably ``sqlite+aiosqlite://`` used by the tests)
    pass through untouched.
    """
    parts = urlsplit(url)
    scheme = parts.scheme

    if not scheme.startswith(("postgres", "postgresql")):
        return url, {}

    # postgres:// and bare postgresql:// -> the async driver.
    # An explicitly chosen driver (e.g. postgresql+psycopg) is respected.
    if "+" not in scheme:
        scheme = _ASYNC_DRIVER

    connect_args: dict[str, object] = {}
    kept_params: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key not in _LIBPQ_ONLY_PARAMS:
            kept_params.append((key, value))
            continue
        if key == "sslmode" and scheme.endswith("asyncpg"):
            if value in _TLS_MODES:
                connect_args["ssl"] = True
            elif value == "disable":
                connect_args["ssl"] = False

    normalized = urlunsplit(
        (scheme, parts.netloc, parts.path, urlencode(kept_params), parts.fragment)
    )
    return normalized, connect_args


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")
