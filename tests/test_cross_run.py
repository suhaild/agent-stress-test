from datetime import datetime, timedelta, timezone

from agent_stress_test.models import Cluster, Verdict
from agent_stress_test.orchestration.cross_run import (
    RunDiff,
    diff_against_previous,
    previous_completed_run,
    reliability_trend,
    rule_pass_rate_history,
)
from tests.conftest import make_agent_spec

_SPEC = make_agent_spec()
_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _run(offset_days: int, *, run_id: str, score: float | None, status: str = "completed"):
    from agent_stress_test.models import Run

    return Run(
        id=run_id,
        agent_spec=_SPEC,
        provider="fake",
        status=status,
        started_at=_BASE + timedelta(days=offset_days),
        final_score=score,
    )


def _verdict(rule_id: str, *, passed: bool, node_id: str = "n1") -> Verdict:
    return Verdict(
        run_id="r",
        node_id=node_id,
        passed=passed,
        rule_id=rule_id,
        reason="x",
        tier="rules",
        confidence=1.0,
        severity="major",
    )


# --- reliability_trend -----------------------------------------------------


def test_reliability_trend_orders_oldest_to_newest():
    runs = [
        _run(2, run_id="c", score=0.9),
        _run(0, run_id="a", score=0.5),
        _run(1, run_id="b", score=0.7),
    ]
    trend = reliability_trend(runs)
    assert [point.run_id for point in trend] == ["a", "b", "c"]
    assert [point.score for point in trend] == [0.5, 0.7, 0.9]


def test_reliability_trend_skips_runs_with_no_score_yet():
    runs = [_run(0, run_id="a", score=None, status="running"), _run(1, run_id="b", score=0.8)]
    trend = reliability_trend(runs)
    assert [point.run_id for point in trend] == ["b"]


# --- previous_completed_run -------------------------------------------------


def test_previous_completed_run_picks_the_most_recent_earlier_one():
    current = _run(2, run_id="current", score=0.6)
    agent_runs = [
        _run(0, run_id="oldest", score=0.4),
        _run(1, run_id="middle", score=0.5),
        current,
    ]
    previous = previous_completed_run(current, agent_runs)
    assert previous.id == "middle"


def test_previous_completed_run_ignores_non_completed_runs():
    current = _run(1, run_id="current", score=0.6)
    agent_runs = [_run(0, run_id="failed-one", score=None, status="failed"), current]
    assert previous_completed_run(current, agent_runs) is None


def test_previous_completed_run_none_when_this_is_the_first_run():
    current = _run(0, run_id="only", score=0.6)
    assert previous_completed_run(current, [current]) is None


# --- diff_against_previous ---------------------------------------------------


def _cluster(label: str) -> Cluster:
    return Cluster(run_id="r", label=label, member_node_ids=["n1"])


def test_diff_reports_new_and_resolved_clusters_and_score_delta():
    current_run = _run(1, run_id="current", score=0.8)
    previous_run = _run(0, run_id="previous", score=0.5)
    diff = diff_against_previous(
        current_run,
        [_cluster("still-here"), _cluster("brand-new")],
        previous_run,
        [_cluster("still-here"), _cluster("now-resolved")],
    )
    assert diff.new_cluster_labels == ["brand-new"]
    assert diff.resolved_cluster_labels == ["now-resolved"]
    assert diff.score_delta == 0.8 - 0.5


def test_diff_with_no_previous_run_reports_everything_as_new():
    current_run = _run(0, run_id="only", score=0.7)
    diff = diff_against_previous(current_run, [_cluster("a"), _cluster("b")], None, [])
    assert diff == RunDiff(previous_run_id=None, score_delta=None, new_cluster_labels=["a", "b"])


# --- rule_pass_rate_history ---------------------------------------------------


def test_rule_pass_rate_history_computes_both_sides_independently():
    current = [_verdict("no-invent", passed=True), _verdict("no-invent", passed=False)]
    historical = [
        _verdict("no-invent", passed=True),
        _verdict("no-invent", passed=True),
        _verdict("be-polite", passed=False),
    ]
    rates = {r.rule_id: r for r in rule_pass_rate_history(current, historical)}

    assert rates["no-invent"].current_pass_rate == 0.5
    assert rates["no-invent"].historical_pass_rate == 1.0
    assert rates["be-polite"].current_pass_rate is None
    assert rates["be-polite"].historical_pass_rate == 0.0


def test_rule_pass_rate_history_ignores_non_rule_scoped_verdicts():
    tool_verdict = Verdict(
        run_id="r",
        node_id="n1",
        passed=False,
        reason="bad args",
        tier="llm",
        confidence=0.9,
        severity="minor",
        scope="tool",
    )
    rates = rule_pass_rate_history([tool_verdict], [])
    assert rates == []
