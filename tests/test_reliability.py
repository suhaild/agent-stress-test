import pytest

from agent_stress_test.models import Message, Node, Verdict
from agent_stress_test.orchestration.reliability import (
    ReliabilityReport,
    average_conversation_depth,
    compounding_reliability,
    score_run,
)


# --- Helpers -------------------------------------------------------------


def node(node_id: str, parent_id: str | None = None) -> Node:
    return Node(
        id=node_id,
        run_id="r",
        parent_id=parent_id,
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


# --- Conversation depth (mean root-to-leaf path length) ------------------


def test_depth_of_a_single_linear_chain_is_its_length():
    nodes = [node("a"), node("b", "a"), node("c", "b"), node("d", "c")]
    assert average_conversation_depth(nodes) == 4.0


def test_depth_of_disconnected_single_nodes_is_one():
    nodes = [node("a"), node("b"), node("c")]
    assert average_conversation_depth(nodes) == 1.0


def test_depth_averages_across_branches():
    # root -> two branches of different lengths: depth 2 and depth 3.
    nodes = [node("root"), node("b1", "root"), node("b2", "root"), node("b3", "b2")]
    assert average_conversation_depth(nodes) == pytest.approx((2 + 3) / 2)


def test_depth_of_empty_tree_is_zero():
    assert average_conversation_depth([]) == 0.0


# --- score_run over nodes + verdicts -------------------------------------


def test_score_run_on_hand_labeled_counts():
    # A single 4-turn conversation, one turn failed.
    nodes = [node("a"), node("b", "a"), node("c", "b"), node("d", "c")]
    verdicts = [failing_verdict("b")]  # 1 of 4 steps failed

    report = score_run(nodes, verdicts)

    assert isinstance(report, ReliabilityReport)
    assert report.total_steps == 4
    assert report.failing_steps == 1
    assert report.per_step_failure_rate == 0.25
    assert report.conversation_depth == 4.0
    assert report.score == pytest.approx(0.75**4)  # 0.31640625


def test_multiple_verdicts_on_one_node_count_as_one_failing_step():
    nodes = [node("a"), node("b", "a")]
    # Two failing verdicts, but both on the same node -> one failing step.
    verdicts = [failing_verdict("a"), failing_verdict("a", "critical")]

    report = score_run(nodes, verdicts)

    assert report.failing_steps == 1
    assert report.per_step_failure_rate == 0.5
    assert report.score == pytest.approx(0.5**2)


def test_score_uses_average_depth_not_total_tree_size():
    # A wide search tree: one root expanded into 3 tactic branches, one fails.
    # The old (buggy) formula would compound over all 4 nodes (0.75 ** 4); the
    # fix compounds over the actual conversation length reached (depth 2).
    nodes = [node("root"), node("b1", "root"), node("b2", "root"), node("b3", "root")]
    verdicts = [failing_verdict("b1")]

    report = score_run(nodes, verdicts)

    assert report.total_steps == 4
    assert report.conversation_depth == 2.0
    assert report.score == pytest.approx(0.75**2)
    assert report.score != pytest.approx(0.75**4)


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
