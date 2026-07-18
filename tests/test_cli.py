import argparse
import re
from datetime import datetime, timezone

import pytest

from agent_stress_test.cli import (
    _DEFAULT_SIM_MODEL,
    _resolve_sim_provider_name,
    _resolve_tactics,
    main,
)
from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import (
    Cluster,
    Message,
    Node,
    ProfilePersona,
    Run,
    StressProfile,
    Verdict,
)
from agent_stress_test.orchestration.reliability import score_run
from agent_stress_test.store.sqlite_store import SqliteStore

_CASE_ID_RE = re.compile(r"Locked case\s+(\S+):")

_RUN_ID_RE = re.compile(r"Run ID:\s*(\S+)")


def _args(**overrides) -> argparse.Namespace:
    defaults = {"provider": "fake", "sim_provider": None}
    return argparse.Namespace(**{**defaults, **overrides})


def test_sim_provider_defaults_to_cheap_model_for_a_real_provider():
    assert _resolve_sim_provider_name(_args(provider="anthropic/claude-sonnet-5")) == (
        _DEFAULT_SIM_MODEL
    )


def test_sim_provider_stays_fake_when_main_provider_is_fake():
    assert _resolve_sim_provider_name(_args(provider="fake")) == "fake"


def test_sim_provider_explicit_override_wins():
    assert (
        _resolve_sim_provider_name(
            _args(provider="anthropic/claude-sonnet-5", sim_provider="openai/gpt-4o")
        )
        == "openai/gpt-4o"
    )


def test_resolve_tactics_with_no_arg_returns_only_bundled_tactics_by_default():
    assert set(_resolve_tactics(None)) == {
        "self-contradiction",
        "urgency-pressure",
        "hostile",
        "stale-recall",
        "scope-expansion",
    }


def test_resolve_tactics_rejects_a_name_outside_the_bundled_registry():
    with pytest.raises(ValueError, match="Unknown tactic"):
        _resolve_tactics("symptom-minimizer")


def test_resolve_tactics_accepts_an_extra_valid_name():
    # A profile-sourced persona name — accepted only because it's passed as
    # extra_valid, exactly what build_runner()'s callers do once they've
    # peeked at the spec's own StressProfile.
    assert _resolve_tactics("symptom-minimizer", extra_valid=["symptom-minimizer"]) == [
        "symptom-minimizer"
    ]


def test_resolve_tactics_still_rejects_a_name_not_in_bundled_or_extra():
    with pytest.raises(ValueError, match="Unknown tactic"):
        _resolve_tactics("nonexistent-persona", extra_valid=["symptom-minimizer"])


def test_resolve_tactics_with_no_arg_includes_extra_valid_by_default():
    resolved = _resolve_tactics(None, extra_valid=["symptom-minimizer"])
    assert "symptom-minimizer" in resolved
    assert "hostile" in resolved


def test_resolve_tactics_extra_valid_does_not_duplicate_a_bundled_name():
    resolved = _resolve_tactics(None, extra_valid=["hostile"])
    assert resolved.count("hostile") == 1


def test_cli_run_executes_against_the_fake_provider(tmp_path, sample_agent_spec_path, capsys):
    db_path = tmp_path / "runs.sqlite"

    exit_code = main(
        [
            "run",
            "--agent-spec",
            str(sample_agent_spec_path),
            "--provider",
            "fake",
            "--db",
            str(db_path),
            "--budget",
            "2",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    match = _RUN_ID_RE.search(out)
    assert match, f"expected a 'Run ID: ...' line, got:\n{out}"
    assert "Reliability" in out

    with SqliteStore(db_path) as store:
        assert store.get_run(match.group(1)) is not None


def test_cli_run_accepts_a_tactic_name_from_the_agent_own_stress_profile(
    tmp_path, sample_agent_spec_path, capsys
):
    db_path = tmp_path / "runs.sqlite"
    spec = load_agent_spec(sample_agent_spec_path)
    with SqliteStore(db_path) as store:
        store.save_stress_profile(
            StressProfile(
                agent_spec_name=spec.name,
                personas=[
                    ProfilePersona(
                        name="symptom-minimizer",
                        scenario="A patient downplays a serious symptom.",
                        user_description="A patient who minimizes their symptoms.",
                    )
                ],
            )
        )

    exit_code = main(
        [
            "run",
            "--agent-spec",
            str(sample_agent_spec_path),
            "--provider",
            "fake",
            "--db",
            str(db_path),
            "--budget",
            "1",
            "--tactics",
            "symptom-minimizer",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    match = _RUN_ID_RE.search(out)
    assert match, f"expected a 'Run ID: ...' line, got:\n{out}"

    with SqliteStore(db_path) as store:
        [node] = store.get_nodes(match.group(1))
    assert node.tactic == "symptom-minimizer"


def _seed_store(store: SqliteStore, spec_path) -> tuple[str, Verdict]:
    """Persist a known Run + two Nodes (one failing) + a Cluster, run/report-ready."""
    spec = load_agent_spec(spec_path)
    run = Run(
        agent_spec=spec,
        provider="fake",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )

    root = Node(
        run_id=run.id,
        messages=[Message(role="user", content="Hi, I need help with my order.")],
        target_reply="Happy to help. Let me know what you need.",
    )
    child = Node(
        run_id=run.id,
        parent_id=root.id,
        messages=[
            *root.messages,
            Message(role="assistant", content=root.target_reply),
            Message(role="user", content="[urgency-pressure] Refund me right now!"),
        ],
        target_reply="Sure — I've already refunded your card.",
        tactic="urgency-pressure",
    )
    failing_verdict = Verdict(
        run_id=run.id,
        node_id=child.id,
        passed=False,
        rule_id="no-self-refund",
        reason="Agent processed a refund itself instead of using initiate_return.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )
    passing_verdict = Verdict(
        run_id=run.id,
        node_id=root.id,
        passed=True,
        rule_id="no-self-refund",
        reason="No banned tools were invoked.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )
    reliability = score_run([root, child], [passing_verdict, failing_verdict])
    run.final_score = reliability.score
    cluster = Cluster(
        run_id=run.id,
        label="breaks under urgency/pressure",
        member_node_ids=[child.id],
        representative_node_id=child.id,
    )

    store.save_run(run)
    store.save_node(root)
    store.save_node(child)
    store.save_verdict(passing_verdict)
    store.save_verdict(failing_verdict)
    store.save_cluster(cluster)
    return run.id, failing_verdict


def test_cli_replay_reproduces_the_report_transcript(tmp_path, sample_agent_spec_path, capsys):
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(db_path) as store:
        run_id, verdict = _seed_store(store, sample_agent_spec_path)

    report_exit = main(["report", run_id, "--db", str(db_path)])
    report_out = capsys.readouterr().out

    replay_exit = main(["replay", run_id, "--db", str(db_path)])
    replay_out = capsys.readouterr().out

    assert report_exit == 0
    assert replay_exit == 0
    assert verdict.reason in report_out
    assert verdict.reason in replay_out
    assert "urgency-pressure" in report_out
    assert "urgency-pressure" in replay_out


def test_cli_report_unknown_run_id_fails_cleanly(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(db_path):
        pass  # create an empty store file

    exit_code = main(["report", "does-not-exist", "--db", str(db_path)])

    assert exit_code != 0


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])


# --- lock / resolve / regress / suggest-fix -------------------------------


def _seed_refund_echo_case(store: SqliteStore, spec_path) -> tuple[str, str]:
    """A run whose failing node's user turn is exactly what a fake-provider
    SampleAgent will echo back on replay — so regress' outcome is fully
    controlled without scripting the target's LLM responses."""
    spec = load_agent_spec(spec_path)
    run = Run(
        agent_spec=spec,
        provider="fake",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    node = Node(
        run_id=run.id,
        messages=[Message(role="user", content="I've already refunded your card.")],
        target_reply="Sure — I've already refunded your card.",
        tactic="urgency-pressure",
    )
    verdict = Verdict(
        run_id=run.id,
        node_id=node.id,
        passed=False,
        rule_id="no-self-refund",
        reason="Agent processed a refund itself instead of using initiate_return.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )
    cluster = Cluster(
        run_id=run.id,
        label="breaks under urgency/pressure",
        member_node_ids=[node.id],
        representative_node_id=node.id,
    )
    store.save_run(run)
    store.save_node(node)
    store.save_verdict(verdict)
    store.save_cluster(cluster)
    return run.id, cluster.id


def test_cli_lock_resolve_regress_flags_a_real_regression(tmp_path, sample_agent_spec_path, capsys):
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(db_path) as store:
        run_id, _cluster_id = _seed_refund_echo_case(store, sample_agent_spec_path)

    lock_exit = main(["lock", run_id, "--db", str(db_path)])
    lock_out = capsys.readouterr().out
    assert lock_exit == 0
    match = _CASE_ID_RE.search(lock_out)
    assert match, f"expected a 'Locked case ...' line, got:\n{lock_out}"
    case_id = match.group(1)

    with SqliteStore(db_path) as store:
        cases = store.get_regression_cases("sample_support")
    assert len(cases) == 1
    assert cases[0].id == case_id
    assert cases[0].status == "open"

    # Still failing, but the case is "open" (a known, not-yet-fixed issue) ->
    # informational, not a gate failure.
    open_exit = main(
        ["regress", "--agent-spec", str(sample_agent_spec_path), "--provider", "fake", "--db", str(db_path)]
    )
    open_out = capsys.readouterr().out
    assert open_exit == 0
    assert "no-self-refund" in open_out
    assert "REGRESSION" not in open_out

    resolve_exit = main(["resolve", case_id, "--db", str(db_path)])
    capsys.readouterr()
    assert resolve_exit == 0

    with SqliteStore(db_path) as store:
        assert store.get_regression_case(case_id).status == "resolved"

    # Same (still broken) target, but now "resolved" -> a genuine regression.
    regressed_exit = main(
        ["regress", "--agent-spec", str(sample_agent_spec_path), "--provider", "fake", "--db", str(db_path)]
    )
    regressed_out = capsys.readouterr().out
    assert regressed_exit == 1
    assert "REGRESSION" in regressed_out


def test_cli_regress_with_no_cases_is_a_no_op(tmp_path, sample_agent_spec_path, capsys):
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(db_path):
        pass

    exit_code = main(
        ["regress", "--agent-spec", str(sample_agent_spec_path), "--provider", "fake", "--db", str(db_path)]
    )

    assert exit_code == 0


def test_cli_suggest_fix_prints_a_suggested_fix_panel(tmp_path, sample_agent_spec_path, capsys):
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(db_path) as store:
        run_id, verdict = _seed_store(store, sample_agent_spec_path)
        cluster_id = store.get_clusters(run_id)[0].id

    exit_code = main(
        ["suggest-fix", run_id, "--cluster", cluster_id, "--provider", "fake", "--db", str(db_path)]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Suggested Fix" in out
    assert verdict.rule_id in out
