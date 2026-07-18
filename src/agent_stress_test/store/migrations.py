"""Schema-version guard + one-time migration for ``runs.sqlite``.

Every entity is stored as an opaque ``model_dump_json()`` blob (see
``sqlite_store.py``), so a "schema migration" here means: can every existing
blob still be parsed by *today's* Pydantic models? Phase A1's v2 contract
(``Message.content`` widened to a union, ``Node``/``AgentResponse`` gaining
``tool_calls``) was additive, so any pre-A1 ("v1") row already parses under
the current ("v2") models untouched — but the mechanism here is built to
matter for the next change that *isn't* additive, not just this one.

Two entry points:
  - ``ensure_current_or_raise`` — the startup guard (``cli.py``/``server.py``
    call this once, before doing any real work). A DB whose rows all still
    parse under the current models is stamped current and allowed through
    (this covers both a brand-new DB and one written entirely by today's
    code, e.g. in tests, that never happened to go through this stamping
    before). A DB with a row that *doesn't* parse is refused with an
    actionable message instead of a raw ``ValidationError`` — that's the
    signal an operator actually needs ``migrate()``.
  - ``migrate`` — the explicit, operator-run upgrade: back up the file, then
    for every row, validate under the current models (where a future
    genuinely-breaking change would plug in real field-by-field rewriting)
    and write the re-dumped JSON back. Idempotent: a DB already at the
    current version is a no-op.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agent_stress_test.models import (
    Cluster,
    Node,
    RegressionCase,
    Run,
    StressProfile,
    SystemPromptVersion,
    Verdict,
)

CURRENT_SCHEMA_VERSION = 2

_TABLES_TO_MODELS: dict[str, type[BaseModel]] = {
    "runs": Run,
    "nodes": Node,
    "verdicts": Verdict,
    "clusters": Cluster,
    "regression_cases": RegressionCase,
    "system_prompt_versions": SystemPromptVersion,
    "stress_profiles": StressProfile,
}

_VERSION_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_version "
    "(id INTEGER PRIMARY KEY CHECK (id = 0), version INTEGER NOT NULL)"
)


class MigrationError(ValueError):
    """A DB can't be safely used as-is — an actionable message, not a raw ValidationError."""


def get_schema_version(conn: sqlite3.Connection) -> int:
    """The DB's stamped version, or 1 if it predates the marker itself."""
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "schema_version" not in tables:
        return 1
    row = conn.execute("SELECT version FROM schema_version WHERE id = 0").fetchone()
    return row[0] if row is not None else 1


def _stamp_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(_VERSION_TABLE_DDL)
    conn.execute(
        "INSERT INTO schema_version (id, version) VALUES (0, ?) "
        "ON CONFLICT (id) DO UPDATE SET version = excluded.version",
        (version,),
    )
    conn.commit()


def _existing_tables(conn: sqlite3.Connection) -> dict[str, type[BaseModel]]:
    present = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return {table: model for table, model in _TABLES_TO_MODELS.items() if table in present}


def _rows(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    return conn.execute(f"SELECT id, data FROM {table}").fetchall()


def _all_rows_parse(conn: sqlite3.Connection) -> bool:
    for table, model in _existing_tables(conn).items():
        for _row_id, data in _rows(conn, table):
            try:
                model.model_validate_json(data)
            except ValidationError:
                return False
    return True


def ensure_current_or_raise(db_path: str | Path) -> None:
    """Startup guard: call once before any real work in ``cli.py``/``server.py``.

    Raises ``MigrationError`` (a ``ValueError``) if the DB has a row that
    doesn't parse under the current models — never a raw ``ValidationError``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        if get_schema_version(conn) == CURRENT_SCHEMA_VERSION:
            return
        if not _all_rows_parse(conn):
            raise MigrationError(
                f"'{db_path}' is an older schema version and has data that no longer "
                f"parses — run the migration script first: "
                f"python -m agent_stress_test.store.migrations '{db_path}'"
            )
        _stamp_schema_version(conn, CURRENT_SCHEMA_VERSION)
    finally:
        conn.close()


def migrate(db_path: str | Path) -> None:
    """Upgrade ``db_path`` to ``CURRENT_SCHEMA_VERSION`` in place.

    Backs up the file first (``<name>.bak-v<old_version>``). A DB already at
    the current version is a no-op — safe to run more than once.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        version = get_schema_version(conn)
        if version == CURRENT_SCHEMA_VERSION:
            return
        if version > CURRENT_SCHEMA_VERSION:
            raise MigrationError(
                f"'{db_path}' is schema version {version}, newer than this build's "
                f"{CURRENT_SCHEMA_VERSION} — refusing to downgrade."
            )

        conn.close()
        backup_path = db_path.with_name(f"{db_path.name}.bak-v{version}")
        shutil.copy2(db_path, backup_path)
        conn = sqlite3.connect(str(db_path))

        for table, model in _existing_tables(conn).items():
            for row_id, data in _rows(conn, table):
                upgraded = model.model_validate_json(data)
                conn.execute(
                    f"UPDATE {table} SET data = ? WHERE id = ?",
                    (upgraded.model_dump_json(), row_id),
                )
        conn.commit()
        _stamp_schema_version(conn, CURRENT_SCHEMA_VERSION)
    finally:
        conn.close()


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m agent_stress_test.store.migrations <path-to-runs.sqlite>")
        return 1
    migrate(argv[0])
    print(f"'{argv[0]}' is now at schema version {CURRENT_SCHEMA_VERSION}.")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
