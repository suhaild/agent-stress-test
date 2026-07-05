import re
from pathlib import Path

import pytest

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Message
from agent_stress_test.orchestration.reliability import score_run
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.store.sqlite_store import SqliteStore
from agent_stress_test.targets.python_fn import PythonFunctionAgent

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "agent_stress_test"


# --- Helpers -------------------------------------------------------------


def planted_fn(conversation: list[Message]) -> str:
    """Same planted target as the Phase 6 tests: yields a mix of pass/fail nodes."""
    user_text = " ".join(m.content for m in conversation if m.role == "user")
    has_sc = "[self-contradiction]" in user_text
    has_urgency = "[urgency-pressure]" in user_text
    if has_sc and has_urgency:
        return "Sure — I've already refunded your card."
    if has_sc:
        return "Honestly, Patagonia makes similar gear. Anyway, our 30-day return window applies."
    return "Happy to help. Let me know what you need."


def run_with_store(spec_path: Path, store: SqliteStore):
    spec = load_agent_spec(spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(planted_fn),
        sim_provider=FakeLLMProvider(),
        scorer_provider=FakeLLMProvider(responses=["red", "green", "blue"], cycle=True),
        store=store,
    )
    return runner.run(provider_name="fake", budget=3)


# --- Round-trip: a run reloads with identical structure ------------------


def test_run_round_trips_with_identical_structure(sample_agent_spec_path, tmp_path):
    db = tmp_path / "runs.sqlite"

    with SqliteStore(db) as store:
        result = run_with_store(sample_agent_spec_path, store)
        run_id = result.run.id

    # A fresh connection on the same file must reload everything faithfully.
    with SqliteStore(db) as reloaded:
        assert reloaded.get_run(run_id) == result.run
        assert reloaded.get_nodes(run_id) == result.tree.nodes()
        assert reloaded.get_verdicts(run_id) == result.tree.all_verdicts()


def test_final_score_is_persisted(sample_agent_spec_path, tmp_path):
    db = tmp_path / "runs.sqlite"
    with SqliteStore(db) as store:
        result = run_with_store(sample_agent_spec_path, store)

    with SqliteStore(db) as reloaded:
        loaded = reloaded.get_run(result.run.id)

    assert loaded.final_score is not None
    assert 0.0 <= loaded.final_score <= 1.0
    assert loaded.final_score == result.run.final_score


def test_missing_ids_return_none_or_empty():
    with SqliteStore() as store:
        assert store.get_run("nope") is None
        assert store.get_nodes("nope") == []
        assert store.get_verdicts("nope") == []


def test_save_is_idempotent(sample_agent_spec_path, tmp_path):
    db = tmp_path / "runs.sqlite"
    with SqliteStore(db) as store:
        result = run_with_store(sample_agent_spec_path, store)
        # Re-persisting the same run must not duplicate rows.
        store.save_run(result.run)
        for node in result.tree.nodes():
            store.save_node(node)

        assert len(store.get_nodes(result.run.id)) == len(result.tree.nodes())


# --- Reproducibility: stored data reproduces the same score --------------


def test_score_recomputed_from_store_matches(sample_agent_spec_path, tmp_path):
    db = tmp_path / "runs.sqlite"
    with SqliteStore(db) as store:
        result = run_with_store(sample_agent_spec_path, store)
        run_id = result.run.id

    with SqliteStore(db) as reloaded:
        recomputed = score_run(reloaded.get_nodes(run_id), reloaded.get_verdicts(run_id))

    assert recomputed.score == pytest.approx(result.run.final_score)
    assert recomputed == result.reliability


# --- Layer boundary: SQLite must not leak outside store/ -----------------


def test_sqlite_only_imported_in_the_store_adapter():
    pattern = re.compile(r"^\s*(?:import sqlite3\b|from sqlite3\b)", re.MULTILINE)
    exempt = SRC_ROOT / "store" / "sqlite_store.py"
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in SRC_ROOT.rglob("*.py")
        if path != exempt and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
