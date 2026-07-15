import re
from pathlib import Path

import pytest

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Message, RegressionCase, SystemPromptVersion
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
    # sample_n defaults to 3 (>= 2), so build_runner() builds a self-consistency
    # scorer automatically, resampling this same target.
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(planted_fn),
        sim_provider=FakeLLMProvider(),
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


# --- Regression cases -----------------------------------------------------


def test_regression_case_round_trips_and_filters_by_agent(tmp_path):
    db = tmp_path / "runs.sqlite"
    case = RegressionCase(
        agent_spec_name="sample_support",
        messages=[Message(role="user", content="Refund me right now!")],
        tactic="urgency-pressure",
        rule_id="no-self-refund",
        severity="critical",
        source_run_id="r1",
        source_cluster_id="c1",
    )

    with SqliteStore(db) as store:
        store.save_regression_case(case)

    with SqliteStore(db) as reloaded:
        assert reloaded.get_regression_case(case.id) == case
        assert reloaded.get_regression_cases("sample_support") == [case]
        assert reloaded.get_regression_cases("some_other_agent") == []
        assert reloaded.get_regression_case("nope") is None


def test_regression_case_status_update_is_idempotent_on_id(tmp_path):
    db = tmp_path / "runs.sqlite"
    case = RegressionCase(
        agent_spec_name="sample_support",
        messages=[Message(role="user", content="Refund me right now!")],
        rule_id="no-self-refund",
        severity="critical",
        source_run_id="r1",
        source_cluster_id="c1",
    )

    with SqliteStore(db) as store:
        store.save_regression_case(case)
        store.save_regression_case(case.model_copy(update={"status": "resolved"}))

        cases = store.get_regression_cases("sample_support")
        assert len(cases) == 1
        assert cases[0].status == "resolved"


# --- System prompt versions -------------------------------------------------


def test_system_prompt_version_round_trips_and_filters_by_agent(tmp_path):
    db = tmp_path / "runs.sqlite"
    version = SystemPromptVersion(agent_spec_name="sample_support", system_prompt="Old prompt.")

    with SqliteStore(db) as store:
        store.save_system_prompt_version(version)

    with SqliteStore(db) as reloaded:
        assert reloaded.get_system_prompt_versions("sample_support") == [version]
        assert reloaded.get_system_prompt_versions("some_other_agent") == []


def test_system_prompt_versions_are_returned_most_recent_first(tmp_path):
    db = tmp_path / "runs.sqlite"
    first = SystemPromptVersion(agent_spec_name="sample_support", system_prompt="First.")
    second = SystemPromptVersion(agent_spec_name="sample_support", system_prompt="Second.")

    with SqliteStore(db) as store:
        store.save_system_prompt_version(first)
        store.save_system_prompt_version(second)

        versions = store.get_system_prompt_versions("sample_support")
        assert [v.system_prompt for v in versions] == ["Second.", "First."]


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
