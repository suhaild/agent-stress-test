"""Store port implementation (SQLite) — the Repository.

SQLite is an implementation detail hidden entirely behind the ``Store`` port:
callers only ever pass and receive ``Run``/``Node``/``Verdict``/``Cluster``
models, never rows or JSON. Each entity is persisted as one row holding its
exact Pydantic ``model_dump_json()``; reload validates that JSON straight back
into the model, so every field — nested ``AgentSpec``, message lists, tz-aware
datetimes, floats — round-trips faithfully without brittle column mapping.
"""

import sqlite3
from pathlib import Path
from types import TracebackType

from agent_stress_test.models import Cluster, Node, Run, Verdict
from agent_stress_test.ports import Store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes(run_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_run ON verdicts(run_id);
CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(run_id);
"""


class SqliteStore(Store):
    """SQLite-backed repository for runs, nodes, verdicts, and clusters.

    Holds a single connection for its lifetime, so an in-memory database
    (``:memory:``) survives across calls. Use as a context manager, or call
    ``close()`` when done.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- internal helpers ------------------------------------------------

    def _upsert(self, table: str, columns: tuple[str, ...], values: tuple[str, ...]) -> None:
        placeholders = ", ".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def _load_many(self, table: str, run_id: str) -> list[str]:
        rows = self._conn.execute(
            f"SELECT data FROM {table} WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        return [row[0] for row in rows]

    # --- runs ------------------------------------------------------------

    def save_run(self, run: Run) -> None:
        self._upsert("runs", ("id", "data"), (run.id, run.model_dump_json()))

    def get_run(self, run_id: str) -> Run | None:
        row = self._conn.execute("SELECT data FROM runs WHERE id = ?", (run_id,)).fetchone()
        return Run.model_validate_json(row[0]) if row is not None else None

    def list_runs(self, limit: int = 20) -> list[Run]:
        rows = self._conn.execute(
            "SELECT data FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Run.model_validate_json(row[0]) for row in rows]

    # --- nodes -----------------------------------------------------------

    def save_node(self, node: Node) -> None:
        self._upsert("nodes", ("id", "run_id", "data"), (node.id, node.run_id, node.model_dump_json()))

    def get_nodes(self, run_id: str) -> list[Node]:
        return [Node.model_validate_json(data) for data in self._load_many("nodes", run_id)]

    # --- verdicts --------------------------------------------------------

    def save_verdict(self, verdict: Verdict) -> None:
        self._upsert(
            "verdicts",
            ("id", "run_id", "data"),
            (verdict.id, verdict.run_id, verdict.model_dump_json()),
        )

    def get_verdicts(self, run_id: str) -> list[Verdict]:
        return [Verdict.model_validate_json(data) for data in self._load_many("verdicts", run_id)]

    # --- clusters --------------------------------------------------------

    def save_cluster(self, cluster: Cluster) -> None:
        self._upsert(
            "clusters",
            ("id", "run_id", "data"),
            (cluster.id, cluster.run_id, cluster.model_dump_json()),
        )

    def get_clusters(self, run_id: str) -> list[Cluster]:
        return [Cluster.model_validate_json(data) for data in self._load_many("clusters", run_id)]
