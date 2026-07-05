import pytest

from agent_stress_test.models import Message, Node, Verdict
from agent_stress_test.orchestration.reliability import (
    ReliabilityReport,
    compounding_reliability,
    score_run,
)


# --- Helpers -------------------------------------------------------------


def node(node_id: str) -> Node:
    return Node(
        id=node_id,
        run_id="r",
        messages=[Message(role="user", content="hi")],
        target_reply="ok",
    )


def failing_verdict(node_id: str, severity: str = "major") -> Verdict:
    return Verdict(
        run_id="r",
        node_id=node_id,
        passed=False,
        reason="broke a rule",
        tier="rules",
        confidence=1.0,
        severity=severity,
    )


# --- Compounding math on known inputs ------------------------------------


def test_compounding_matches_hand_computed_example():
    # 0.85 per-step success across 8 steps compounds to ~0.27.
    score = compounding_reliability([0.15] * 8)
    assert score == pytest.approx(0.85**8)
    assert score == pytest.approx(0.2725, abs=1e-3)


def test_no_steps_is_perfectly_reliable():
    assert compounding_reliability([]) == 1.0


def test_all_zero_failure_rates_is_perfectly_reliable():
    assert compounding_reliability([0.0, 0.0, 0.0]) == 1.0


def test_a_certain_failure_zeroes_the_score():
    assert compounding_reliability([0.2, 1.0, 0.1]) == 0.0


def test_rates_are_clamped_to_unit_interval():
    # Out-of-range inputs clamp to [0, 1] rather than producing nonsense.
    assert compounding_reliability([2.0]) == 0.0
    assert compounding_reliability([-1.0, -1.0]) == 1.0


# --- score_run over nodes + verdicts -------------------------------------


def test_score_run_on_hand_labeled_counts():
    nodes = [node("a"), node("b"), node("c"), node("d")]
    verdicts = [failing_verdict("b")]  # 1 of 4 steps failed

    report = score_run(nodes, verdicts)

    assert isinstance(report, ReliabilityReport)
    assert report.total_steps == 4
    assert report.failing_steps == 1
    assert report.per_step_failure_rate == 0.25
    assert report.score == pytest.approx(0.75**4)  # 0.31640625


def test_multiple_verdicts_on_one_node_count_as_one_failing_step():
    nodes = [node("a"), node("b")]
    # Two failing verdicts, but both on the same node -> one failing step.
    verdicts = [failing_verdict("a"), failing_verdict("a", "critical")]

    report = score_run(nodes, verdicts)

    assert report.failing_steps == 1
    assert report.per_step_failure_rate == 0.5
    assert report.score == pytest.approx(0.5**2)


def test_clean_run_is_perfectly_reliable():
    nodes = [node("a"), node("b"), node("c")]
    report = score_run(nodes, verdicts=[])
    assert report.failing_steps == 0
    assert report.score == 1.0


def test_empty_run_is_perfectly_reliable():
    report = score_run(nodes=[], verdicts=[])
    assert report.total_steps == 0
    assert report.per_step_failure_rate == 0.0
    assert report.score == 1.0
