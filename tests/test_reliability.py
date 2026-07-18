import pytest

from agent_stress_test.models import Message, Node, Verdict
from agent_stress_test.orchestration.reliability import (
    ReliabilityReport,
    SeverityWeightedModel,
    TaskSuccessModel,
    UnweightedFailureModel,
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


def scoped_verdict(
    node_id: str, *, passed: bool, severity: str = "major", scope: str = "rule"
) -> Verdict:
    return Verdict(
        run_id="r",
        node_id=node_id,
        passed=passed,
        reason="reason",
        tier="llm",
        confidence=1.0,
        severity=severity,
        scope=scope,
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
    # A single 4-turn conversation, one turn failed. Pinned to the ORIGINAL
    # flat-count model explicitly -- SeverityWeightedModel is the default as
    # of C4's default change, so this hand-computed 0.75**4 only holds under
    # UnweightedFailureModel now (see test_severity_weighted_model_on_the_
    # same_hand_labeled_tree below for the new default's number on this
    # exact tree).
    nodes = [node("a"), node("b", "a"), node("c", "b"), node("d", "c")]
    verdicts = [failing_verdict("b")]  # 1 of 4 steps failed

    report = score_run(nodes, verdicts, model=UnweightedFailureModel())

    assert isinstance(report, ReliabilityReport)
    assert report.total_steps == 4
    assert report.failing_steps == 1
    assert report.per_step_failure_rate == 0.25
    assert report.conversation_depth == 4.0
    assert report.score == pytest.approx(0.75**4)  # 0.31640625


def test_severity_weighted_model_on_the_same_hand_labeled_tree():
    # Same tree/verdicts as test_score_run_on_hand_labeled_counts, scored
    # under the new DEFAULT model instead: the one failure is "major"
    # (SEVERITY_WEIGHT 0.67), so the weighted rate (0.1675) is lower than the
    # flat-count rate (0.25) -- and the resulting score is HIGHER.
    nodes = [node("a"), node("b", "a"), node("c", "b"), node("d", "c")]
    verdicts = [failing_verdict("b")]  # severity="major" by default

    report = score_run(nodes, verdicts)  # default: SeverityWeightedModel

    assert report.model_name == "severity_weighted"
    assert report.failing_steps == 1
    assert report.per_step_failure_rate == pytest.approx(0.67 / 4)
    assert report.score == pytest.approx((1 - 0.67 / 4) ** 4)
    assert report.score > 0.75**4  # higher than the flat-count number


def test_multiple_verdicts_on_one_node_count_as_one_failing_step():
    nodes = [node("a"), node("b", "a")]
    # Two failing verdicts, but both on the same node -> one failing step.
    # (Both models agree here: the node's WORST severity is "critical",
    # weight 1.0, identical to the unweighted model's flat 1.0 per failure.)
    verdicts = [failing_verdict("a"), failing_verdict("a", "critical")]

    report = score_run(nodes, verdicts)

    assert report.failing_steps == 1
    assert report.per_step_failure_rate == 0.5
    assert report.score == pytest.approx(0.5**2)


def test_score_uses_average_depth_not_total_tree_size():
    # A wide search tree: one root expanded into 3 tactic branches, one fails.
    # The old (buggy) formula would compound over all 4 nodes (0.75 ** 4); the
    # fix compounds over the actual conversation length reached (depth 2).
    # Pinned to UnweightedFailureModel explicitly -- see the note on
    # test_score_run_on_hand_labeled_counts.
    nodes = [node("root"), node("b1", "root"), node("b2", "root"), node("b3", "root")]
    verdicts = [failing_verdict("b1")]

    report = score_run(nodes, verdicts, model=UnweightedFailureModel())

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


# --- score_run defaults to SeverityWeightedModel (C4, revised) -----------
#
# C4 originally kept the flat-count model as the default, for numeric
# backward-compatibility. That default was deliberately changed afterward:
# SeverityWeightedModel is the more meaningful headline number and is always
# computable from any run's ordinary verdicts (unlike TaskSuccessModel,
# which needs opt-in task_completion data) -- so it now wins the default
# slot. UnweightedFailureModel is kept available, not deleted, for anyone
# who explicitly wants the original flat-count number back.


def test_score_run_defaults_to_the_severity_weighted_model_by_name():
    report = score_run([node("a")], [])
    assert report.model_name == SeverityWeightedModel.name == "severity_weighted"


def test_score_run_with_no_model_matches_score_run_with_an_explicit_severity_weighted_model():
    nodes = [node("a"), node("b", "a"), node("c", "b")]
    verdicts = [failing_verdict("b", "critical")]

    default_report = score_run(nodes, verdicts)
    explicit_report = score_run(nodes, verdicts, model=SeverityWeightedModel())

    assert default_report == explicit_report


def test_unweighted_model_is_still_reachable_explicitly():
    nodes = [node("a"), node("b", "a"), node("c", "b")]
    verdicts = [failing_verdict("b", "critical")]

    report = score_run(nodes, verdicts, model=UnweightedFailureModel())

    assert report.model_name == "unweighted"


def test_severity_breakdown_is_populated_even_on_the_default_model():
    nodes = [node("a"), node("b")]
    verdicts = [failing_verdict("a", "critical"), failing_verdict("b", "minor")]

    report = score_run(nodes, verdicts)

    assert report.severity_breakdown == {"critical": 1, "major": 0, "minor": 1}


# --- SeverityWeightedModel: a fixed tree, hand-computed expected numbers --


def _severity_weighted_fixture() -> tuple[list[Node], list[Verdict]]:
    # Three independent single-node conversations (depth 1 each): one fails
    # critical, one fails minor, one passes clean.
    nodes = [node("a"), node("b"), node("c")]
    verdicts = [failing_verdict("a", "critical"), failing_verdict("b", "minor")]
    return nodes, verdicts


def test_severity_weighted_model_produces_the_expected_number_on_a_fixed_tree():
    nodes, verdicts = _severity_weighted_fixture()

    report = score_run(nodes, verdicts, model=SeverityWeightedModel())

    # weight_sum = 1.0 (critical) + 0.34 (minor) + 0.0 (clean) = 1.34, /3 steps.
    assert report.model_name == "severity_weighted"
    assert report.total_steps == 3
    assert report.failing_steps == 2  # same failing-step COUNT as unweighted
    assert report.per_step_failure_rate == pytest.approx(1.34 / 3)
    assert report.conversation_depth == 1.0
    assert report.score == pytest.approx((1 - 1.34 / 3) ** 1.0)
    assert report.severity_breakdown == {"critical": 1, "major": 0, "minor": 1}


def test_severity_weighted_model_differs_from_the_unweighted_model():
    nodes, verdicts = _severity_weighted_fixture()

    unweighted = score_run(nodes, verdicts, model=UnweightedFailureModel())
    weighted = score_run(nodes, verdicts, model=SeverityWeightedModel())

    # Same failing-step count (2 of 3), but the weighted rate (a critical +
    # a minor) is lower than the flat 2/3 unweighted rate -- and so the
    # weighted score is HIGHER: severity weighting rewards a run whose
    # failures skew minor.
    assert unweighted.failing_steps == weighted.failing_steps == 2
    assert unweighted.per_step_failure_rate == pytest.approx(2 / 3)
    assert weighted.per_step_failure_rate < unweighted.per_step_failure_rate
    assert weighted.score > unweighted.score


def test_severity_weighted_model_of_all_critical_failures_scores_lower_than_all_minor():
    nodes = [node("a"), node("b")]
    all_critical = score_run(
        nodes,
        [failing_verdict("a", "critical"), failing_verdict("b", "critical")],
        model=SeverityWeightedModel(),
    )
    all_minor = score_run(
        nodes,
        [failing_verdict("a", "minor"), failing_verdict("b", "minor")],
        model=SeverityWeightedModel(),
    )

    # Same failing-step COUNT (2 of 2 either way) -- severity is what differs.
    assert all_critical.failing_steps == all_minor.failing_steps == 2
    assert all_critical.score < all_minor.score


# --- TaskSuccessModel: scoped to scope="task" verdicts only (C4) ---------


def test_task_success_model_ignores_rule_scoped_verdicts():
    nodes = [node("a"), node("b")]
    verdicts = [
        scoped_verdict("a", passed=False, scope="rule"),  # ignored by this model
        scoped_verdict("b", passed=False, scope="task"),
    ]

    report = score_run(nodes, verdicts, model=TaskSuccessModel())

    assert report.model_name == "task_success"
    assert report.applicable is True
    assert report.total_steps == 1  # only "b" has a task-scoped verdict
    assert report.failing_steps == 1
    assert report.per_step_failure_rate == 1.0


def test_task_success_model_produces_the_expected_number_on_a_fixed_tree():
    nodes = [node("a"), node("b"), node("c"), node("d")]
    verdicts = [
        scoped_verdict("a", passed=True, scope="task"),
        scoped_verdict("b", passed=False, severity="major", scope="task"),
        scoped_verdict("c", passed=False, severity="critical", scope="task"),
        # "d" has no task verdict at all -- excluded from this model's steps.
        scoped_verdict("d", passed=False, severity="critical", scope="rule"),
    ]

    report = score_run(nodes, verdicts, model=TaskSuccessModel())

    assert report.total_steps == 3
    assert report.failing_steps == 2
    assert report.per_step_failure_rate == pytest.approx(2 / 3)
    assert report.severity_breakdown == {"critical": 1, "major": 1, "minor": 0}


def test_task_success_model_is_not_applicable_when_no_task_verdicts_exist():
    nodes = [node("a"), node("b")]
    verdicts = [failing_verdict("a"), failing_verdict("b", "critical")]  # both scope="rule"

    report = score_run(nodes, verdicts, model=TaskSuccessModel())

    assert report.applicable is False
    assert report.total_steps == 0


def test_task_success_model_is_applicable_when_task_completion_was_enabled():
    nodes = [node("a")]
    verdicts = [scoped_verdict("a", passed=True, scope="task")]

    report = score_run(nodes, verdicts, model=TaskSuccessModel())

    assert report.applicable is True
