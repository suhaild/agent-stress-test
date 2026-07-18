import json
import sqlite3
from datetime import datetime, timezone

import pytest
from rich.console import Console

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Run, Verdict
from agent_stress_test.store.migrations import (
    CURRENT_SCHEMA_VERSION,
    MigrationError,
    ensure_current_or_raise,
    get_schema_version,
    migrate,
)
from agent_stress_test.store.sqlite_store import SqliteStore


def _seed_v1_shaped_db(db_path, spec_path) -> tuple[str, str]:
    """A runs.sqlite with a pre-A1 ("v1") node row — no schema_version table,
    and the node's JSON has no `tool_calls` key at all (it didn't exist yet).
    Everything else is inserted through today's models: Run/Verdict weren't
    changed by A1, so there's nothing "old-shaped" about their rows.
    """
    spec = load_agent_spec(spec_path)
    run = Run(
        agent_spec=spec,
        provider="fake",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    verdict = Verdict(
        run_id=run.id,
        node_id="node-1",
        passed=False,
        rule_id="no-self-refund",
        reason="Agent processed a refund itself instead of using initiate_return.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )
    v1_node_json = json.dumps(
        {
            "id": "node-1",
            "run_id": run.id,
            "parent_id": None,
            "messages": [
                {"role": "user", "content": "I've already refunded your card.", "cache": False}
            ],
            "target_reply": "Sure — I've already refunded your card.",
            "tactic": "urgency-pressure",
            "instability_score": None,
            "verdict_id": None,
            # no "tool_calls" key — this is the pre-A1 shape.
        }
    )

    with SqliteStore(str(db_path)) as store:
        store.save_run(run)
        store.save_verdict(verdict)
    # Bypass SqliteStore.save_node (which would dump today's shape) to insert
    # the deliberately old-shaped node row directly.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO nodes (id, run_id, data) VALUES (?, ?, ?)",
        ("node-1", run.id, v1_node_json),
    )
    conn.commit()
    conn.close()
    return run.id, "node-1"


def test_migrate_upgrades_v1_rows_with_no_data_loss(tmp_path, sample_agent_spec_path):
    db_path = tmp_path / "runs.sqlite"
    run_id, node_id = _seed_v1_shaped_db(db_path, sample_agent_spec_path)

    conn = sqlite3.connect(str(db_path))
    assert get_schema_version(conn) == 1
    conn.close()

    migrate(db_path)

    conn = sqlite3.connect(str(db_path))
    assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION
    conn.close()

    backup_path = db_path.with_name(f"{db_path.name}.bak-v1")
    assert backup_path.exists()

    with SqliteStore(str(db_path)) as store:
        run = store.get_run(run_id)
        [node] = store.get_nodes(run_id)
        [verdict] = store.get_verdicts(run_id)

    assert run.provider == "fake"
    assert node.target_reply == "Sure — I've already refunded your card."
    assert node.tactic == "urgency-pressure"
    assert node.messages[0].content == "I've already refunded your card."
    assert node.tool_calls == []  # upgraded in: the field didn't exist before
    assert verdict.rule_id == "no-self-refund"


def test_migrate_is_idempotent(tmp_path, sample_agent_spec_path):
    db_path = tmp_path / "runs.sqlite"
    _seed_v1_shaped_db(db_path, sample_agent_spec_path)

    migrate(db_path)
    backups_after_first = list(tmp_path.glob("*.bak-*"))

    migrate(db_path)  # should be a no-op: already at CURRENT_SCHEMA_VERSION
    backups_after_second = list(tmp_path.glob("*.bak-*"))

    assert backups_after_first == backups_after_second


def test_a_migrated_db_loads_and_renders_correctly(tmp_path, sample_agent_spec_path):
    """Not just "do the raw rows parse" (see
    test_migrate_upgrades_v1_rows_with_no_data_loss above) but "does the real
    report-rendering pipeline (tree reconstruction + Rich rendering) work on
    a migrated run" — exercised directly through composition.py/terminal.py
    rather than through the CLI, since which CLI subcommands exist is
    unrelated to whether a migration actually preserved usable data (the
    dashboard is the real front end for viewing a run; see cli.py's
    module docstring)."""
    from agent_stress_test.composition import load_bundle
    from agent_stress_test.orchestration.reliability import score_run
    from agent_stress_test.report.terminal import render_full_report, render_transcript

    db_path = tmp_path / "runs.sqlite"
    run_id, node_id = _seed_v1_shaped_db(db_path, sample_agent_spec_path)
    migrate(db_path)

    with SqliteStore(str(db_path)) as store:
        run, tree, verdicts, clusters = load_bundle(store, run_id)

    console = Console(record=True, width=120, force_terminal=False)
    render_full_report(
        console,
        run=run,
        reliability=score_run(tree.nodes(), verdicts),
        clusters=clusters,
        tree=tree,
        verdicts=verdicts,
    )
    # This seeded db has no Cluster, so render_full_report's per-cluster
    # transcript loop has nothing to iterate -- render the failing node's
    # transcript directly too, same as the old cli.py `replay` command did
    # (walking tree.failures() rather than clusters).
    render_transcript(console, tree, node_id, verdicts)
    text = console.export_text()

    assert "no-self-refund" in text
    assert "Agent processed a refund itself" in text


def test_ensure_current_or_raise_stamps_a_brand_new_db(tmp_path):
    db_path = tmp_path / "runs.sqlite"

    ensure_current_or_raise(db_path)

    conn = sqlite3.connect(str(db_path))
    assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION
    conn.close()


def test_ensure_current_or_raise_self_heals_data_that_still_parses(
    tmp_path, sample_agent_spec_path
):
    """Data written entirely by today's code but never stamped (e.g. seeded
    directly through SqliteStore in a test, bypassing the CLI/dashboard
    startup guard) must not be treated as a legacy DB needing migration —
    only data that genuinely fails to parse should ever block startup."""
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(str(db_path)) as store:
        spec = load_agent_spec(sample_agent_spec_path)
        store.save_run(Run(agent_spec=spec, provider="fake"))

    ensure_current_or_raise(db_path)  # must not raise

    conn = sqlite3.connect(str(db_path))
    assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION
    conn.close()


def test_ensure_current_or_raise_hard_fails_on_genuinely_incompatible_data(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(str(db_path)):
        pass  # just create the tables
    conn = sqlite3.connect(str(db_path))
    # A node row that can never parse under the current model (target_reply,
    # a required field, is missing) — simulating a future genuinely-breaking
    # schema change, not anything A1 itself produced.
    conn.execute(
        "INSERT INTO nodes (id, run_id, data) VALUES (?, ?, ?)",
        ("broken-node", "run-1", json.dumps({"id": "broken-node", "run_id": "run-1"})),
    )
    conn.commit()
    conn.close()

    with pytest.raises(MigrationError, match="migration script"):
        ensure_current_or_raise(db_path)


def test_cli_reports_the_migration_error_cleanly_not_a_raw_traceback(tmp_path, capsys):
    from agent_stress_test.cli import main

    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(str(db_path)):
        pass
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO nodes (id, run_id, data) VALUES (?, ?, ?)",
        ("broken-node", "run-1", json.dumps({"id": "broken-node", "run_id": "run-1"})),
    )
    conn.commit()
    conn.close()

    # ensure_current_or_raise() runs before any subcommand dispatches, so
    # which subcommand is requested is irrelevant here — "run" is simply
    # whichever one still exists (see cli.py's module docstring: "report",
    # the command this test originally used, was retired in favor of the
    # dashboard's equivalent view).
    exit_code = main(["run", "--db", str(db_path)])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "migration script" in out
    assert "Traceback" not in out
