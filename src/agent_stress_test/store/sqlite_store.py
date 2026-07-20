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

from agent_stress_test.models import (
    Cluster,
    Node,
    RegressionCase,
    Run,
    StressProfile,
    SystemPromptVersion,
    Verdict,
)
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
CREATE TABLE IF NOT EXISTS regression_cases (
    id TEXT PRIMARY KEY, agent_spec_name TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS system_prompt_versions (
    id TEXT PRIMARY KEY, agent_spec_name TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stress_profiles (
    id TEXT PRIMARY KEY, agent_spec_name TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes(run_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_run ON verdicts(run_id);
CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_regression_cases_agent ON regression_cases(agent_spec_name);
CREATE INDEX IF NOT EXISTS idx_system_prompt_versions_agent
    ON system_prompt_versions(agent_spec_name);
CREATE INDEX IF NOT EXISTS idx_stress_profiles_agent ON stress_profiles(agent_spec_name);
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
        # table/columns are always literals from this class's own call sites below,
        # never external input.
        placeholders = ", ".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",  # nosec B608
            values,
        )
        self._conn.commit()

    def _load_many(self, table: str, run_id: str) -> list[str]:
        # table is always a literal from this class's own call sites, never external input.
        rows = self._conn.execute(
            f"SELECT data FROM {table} WHERE run_id = ? ORDER BY rowid",  # nosec B608
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

    def list_runs_for_agent(self, agent_spec_name: str, limit: int = 50) -> list[Run]:
        """Most-recent-first runs for one agent spec — Phase RE1's cross-run
        intelligence (trend/diff/pass-rate). ``agent_spec_name`` isn't its own
        indexed column (unlike ``regression_cases``/``system_prompt_versions``
        — those are keyed on it from the start); ``runs`` only ever stored
        ``(id, data)`` (see ``_SCHEMA``), and at this project's scale a full
        scan + filter in Python is simpler than a migration to add and
        backfill a column for it.
        """
        rows = self._conn.execute("SELECT data FROM runs ORDER BY rowid DESC").fetchall()
        matching = (Run.model_validate_json(row[0]) for row in rows)
        return [run for run in matching if run.agent_spec.name == agent_spec_name][:limit]

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

    # --- regression cases --------------------------------------------------

    def save_regression_case(self, case: RegressionCase) -> None:
        self._upsert(
            "regression_cases",
            ("id", "agent_spec_name", "data"),
            (case.id, case.agent_spec_name, case.model_dump_json()),
        )

    def get_regression_case(self, case_id: str) -> RegressionCase | None:
        row = self._conn.execute(
            "SELECT data FROM regression_cases WHERE id = ?", (case_id,)
        ).fetchone()
        return RegressionCase.model_validate_json(row[0]) if row is not None else None

    def get_regression_cases(self, agent_spec_name: str) -> list[RegressionCase]:
        rows = self._conn.execute(
            "SELECT data FROM regression_cases WHERE agent_spec_name = ? ORDER BY rowid",
            (agent_spec_name,),
        ).fetchall()
        return [RegressionCase.model_validate_json(row[0]) for row in rows]

    # --- system prompt versions --------------------------------------------

    def save_system_prompt_version(self, version: SystemPromptVersion) -> None:
        self._upsert(
            "system_prompt_versions",
            ("id", "agent_spec_name", "data"),
            (version.id, version.agent_spec_name, version.model_dump_json()),
        )

    def get_system_prompt_versions(self, agent_spec_name: str) -> list[SystemPromptVersion]:
        # Most-recent-first: rowid grows with insertion order, and versions
        # are only ever inserted, never updated in place.
        rows = self._conn.execute(
            "SELECT data FROM system_prompt_versions WHERE agent_spec_name = ? ORDER BY rowid DESC",
            (agent_spec_name,),
        ).fetchall()
        return [SystemPromptVersion.model_validate_json(row[0]) for row in rows]

    # --- stress profiles -----------------------------------------------------

    def save_stress_profile(self, profile: StressProfile) -> None:
        self._upsert(
            "stress_profiles",
            ("id", "agent_spec_name", "data"),
            (profile.id, profile.agent_spec_name, profile.model_dump_json()),
        )

    def get_stress_profile(self, agent_spec_name: str) -> StressProfile | None:
        # One live profile per agent: the most recently saved row. Editing
        # saves back to the same id (in place); regenerating mints a new one
        # and this always surfaces whichever is newest.
        row = self._conn.execute(
            "SELECT data FROM stress_profiles WHERE agent_spec_name = ? ORDER BY rowid DESC LIMIT 1",
            (agent_spec_name,),
        ).fetchone()
        return StressProfile.model_validate_json(row[0]) if row is not None else None
