"""The Alembic migration must stay in step with the ORM metadata.

The initial migration is hand-written, so this test is the thing that catches
drift: it runs `alembic upgrade head` against a throwaway SQLite file and
compares the resulting tables, columns, nullability and constraints with
``Base.metadata``. Run against PostgreSQL too by setting
``TEST_DATABASE_URL`` (see tests/conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from app.models import Base

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _alembic_config(url: str) -> Config:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def _upgraded_engine(tmp_path: Path) -> sa.Engine:
    db_file = tmp_path / "migration_check.sqlite3"
    command.upgrade(_alembic_config(f"sqlite+aiosqlite:///{db_file.as_posix()}"), "head")
    # Inspect over the plain sync driver; the file is already migrated.
    return sa.create_engine(f"sqlite:///{db_file.as_posix()}")


def test_migration_creates_every_orm_table(tmp_path) -> None:
    engine = _upgraded_engine(tmp_path)
    inspector = sa.inspect(engine)

    assert set(inspector.get_table_names()) - {"alembic_version"} == set(
        Base.metadata.tables
    )
    engine.dispose()


def test_migration_columns_match_the_orm(tmp_path) -> None:
    engine = _upgraded_engine(tmp_path)
    inspector = sa.inspect(engine)

    mismatches: list[str] = []
    for table_name, table in Base.metadata.tables.items():
        migrated = {col["name"]: col for col in inspector.get_columns(table_name)}

        missing = set(table.columns.keys()) - set(migrated)
        extra = set(migrated) - set(table.columns.keys())
        if missing:
            mismatches.append(f"{table_name}: missing columns {sorted(missing)}")
        if extra:
            mismatches.append(f"{table_name}: unexpected columns {sorted(extra)}")

        for name, column in table.columns.items():
            if name not in migrated:
                continue
            if bool(migrated[name]["nullable"]) != bool(column.nullable):
                mismatches.append(
                    f"{table_name}.{name}: nullable "
                    f"{migrated[name]['nullable']} != ORM {column.nullable}"
                )

    assert not mismatches, "\n".join(mismatches)
    engine.dispose()


def test_migration_primary_keys_and_unique_constraints_match(tmp_path) -> None:
    engine = _upgraded_engine(tmp_path)
    inspector = sa.inspect(engine)

    for table_name, table in Base.metadata.tables.items():
        migrated_pk = set(inspector.get_pk_constraint(table_name)["constrained_columns"])
        assert migrated_pk == {col.name for col in table.primary_key}, table_name

        # Unique constraints declared on the model (not the unique indexes).
        expected = {
            tuple(sorted(col.name for col in constraint.columns))
            for constraint in table.constraints
            if isinstance(constraint, sa.UniqueConstraint)
        }
        actual = {
            tuple(sorted(constraint["column_names"]))
            for constraint in inspector.get_unique_constraints(table_name)
        }
        assert expected <= actual, f"{table_name}: missing unique constraints"

    engine.dispose()


def test_migration_foreign_keys_cascade_on_delete(tmp_path) -> None:
    """Deleting a user must take their media, jobs and transcripts with it."""
    engine = _upgraded_engine(tmp_path)
    inspector = sa.inspect(engine)

    for table_name in (
        "refresh_tokens",
        "media_files",
        "transcription_jobs",
        "transcriptions",
    ):
        foreign_keys = inspector.get_foreign_keys(table_name)
        assert foreign_keys, f"{table_name} has no foreign keys"
        for key in foreign_keys:
            assert key["options"].get("ondelete") == "CASCADE", (
                f"{table_name}.{key['constrained_columns']} -> "
                f"{key['referred_table']} is not ON DELETE CASCADE"
            )

    engine.dispose()


def test_downgrade_removes_every_table(tmp_path) -> None:
    db_file = tmp_path / "downgrade_check.sqlite3"
    config = _alembic_config(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = sa.create_engine(f"sqlite:///{db_file.as_posix()}")
    remaining = set(sa.inspect(engine).get_table_names()) - {"alembic_version"}
    assert remaining == set()
    engine.dispose()
