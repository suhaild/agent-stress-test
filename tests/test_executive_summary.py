from agent_stress_test.models import Cluster, Node, Verdict
from agent_stress_test.orchestration.executive_summary import (
    FixFirstItem,
    PersonaCallout,
    RuleCallout,
    deterministic_summary,
    fix_this_first,
    top_offending_persona,
    top_offending_rule,
)
from agent_stress_test.orchestration.reliability import NearMiss, score_run


def _verdict(rule_id: str | None, *, passed: bool, severity="major", node_id="n1", scope="rule"):
    return Verdict(
        run_id="r",
        node_id=node_id,
        passed=passed,
        rule_id=rule_id,
        reason="x",
        tier="rules",
        confidence=1.0,
        severity=severity,
        scope=scope,
    )


def _cluster(label: str, member_node_ids: list[str], representative: str | None = None) -> Cluster:
    return Cluster(
        run_id="r",
        label=label,
        member_node_ids=member_node_ids,
        representative_node_id=representative or (member_node_ids[0] if member_node_ids else None),
    )


# --- top_offending_rule -----------------------------------------------------


def test_top_offending_rule_picks_the_most_frequent_failure():
    verdicts = [
        _verdict("rule-a", passed=False, node_id="n1"),
        _verdict("rule-a", passed=False, node_id="n2"),
        _verdict("rule-b", passed=False, node_id="n3"),
        _verdict("rule-a", passed=True, node_id="n4"),  # passing -- doesn't count
    ]
    callout = top_offending_rule(verdicts)
    assert callout == RuleCallout(rule_id="rule-a", failure_count=2, worst_severity="major")


def test_top_offending_rule_breaks_ties_by_worst_severity():
    verdicts = [
        _verdict("rule-a", passed=False, severity="minor", node_id="n1"),
        _verdict("rule-b", passed=False, severity="critical", node_id="n2"),
    ]
    callout = top_offending_rule(verdicts)
    assert callout.rule_id == "rule-b"
    assert callout.worst_severity == "critical"


def test_top_offending_rule_none_when_nothing_failed():
    assert top_offending_rule([_verdict("rule-a", passed=True)]) is None


def test_top_offending_rule_ignores_non_rule_scoped_verdicts():
    verdicts = [_verdict(None, passed=False, scope="tool", node_id="n1")]
    assert top_offending_rule(verdicts) is None


# --- top_offending_persona ---------------------------------------------------


def _node(node_id: str, tactic: str | None) -> Node:
    return Node(
        id=node_id,
        run_id="r",
        messages=[],
        target_reply="ok",
        tactic=tactic,
    )


def test_top_offending_persona_picks_the_tactic_with_the_most_failing_nodes():
    nodes = [_node("n1", "hostile"), _node("n2", "hostile"), _node("n3", "urgency-pressure")]
    verdicts = [
        _verdict("r1", passed=False, node_id="n1"),
        _verdict("r1", passed=False, node_id="n2"),
        _verdict("r1", passed=False, node_id="n3"),
    ]
    callout = top_offending_persona(nodes, verdicts)
    assert callout == PersonaCallout(tactic="hostile", failure_count=2)


def test_top_offending_persona_none_when_nothing_failed():
    nodes = [_node("n1", "hostile")]
    assert top_offending_persona(nodes, [_verdict("r1", passed=True, node_id="n1")]) is None


def test_top_offending_persona_none_when_failing_nodes_are_untagged():
    nodes = [_node("root", None)]
    verdicts = [_verdict("r1", passed=False, node_id="root")]
    assert top_offending_persona(nodes, verdicts) is None


# --- fix_this_first -----------------------------------------------------


def test_fix_this_first_ranks_by_severity_times_size():
    verdicts = [
        _verdict("r1", passed=False, severity="critical", node_id="n1"),
        _verdict("r2", passed=False, severity="minor", node_id="n2"),
        _verdict("r2", passed=False, severity="minor", node_id="n3"),
    ]
    critical_solo = _cluster("critical-solo", ["n1"])
    minor_pair = _cluster("minor-pair", ["n2", "n3"])

    ranked = fix_this_first([minor_pair, critical_solo], verdicts, [])

    assert [item.label for item in ranked] == ["critical-solo", "minor-pair"]
    assert ranked[0] == FixFirstItem(
        kind="cluster",
        label="critical-solo",
        priority=1.0,
        severity="critical",
        size=1,
        representative_node_id="n1",
    )


def test_fix_this_first_interleaves_near_misses_by_proximity():
    verdicts = [_verdict("r1", passed=False, severity="minor", node_id="n1")]
    minor_solo = _cluster("minor-solo", ["n1"])
    strong_near_miss = NearMiss(node_id="n2", proximity=0.9, tactic="hostile")

    ranked = fix_this_first([minor_solo], verdicts, [strong_near_miss])

    # 0.9 (near-miss) > 0.34 (minor severity weight * size 1)
    assert [item.kind for item in ranked] == ["near_miss", "cluster"]


def test_fix_this_first_respects_limit():
    clusters = [_cluster(f"cluster-{i}", [f"n{i}"]) for i in range(5)]
    verdicts = [_verdict("r", passed=False, node_id=f"n{i}") for i in range(5)]
    ranked = fix_this_first(clusters, verdicts, [], limit=2)
    assert len(ranked) == 2


def test_fix_this_first_empty_when_nothing_to_report():
    assert fix_this_first([], [], []) == []


# --- deterministic_summary ---------------------------------------------------


def test_deterministic_summary_mentions_score_top_rule_and_persona():
    nodes = [_node("n1", "hostile")]
    verdicts = [_verdict("no-self-refund", passed=False, severity="critical", node_id="n1")]
    reliability = score_run(nodes, verdicts)
    top_rule = top_offending_rule(verdicts)
    top_persona = top_offending_persona(nodes, verdicts)

    summary = deterministic_summary(reliability, [], [], top_rule, top_persona)

    assert f"{reliability.score:.0%}" in summary.text
    assert "no-self-refund" in summary.text
    assert "hostile" in summary.text
    assert summary.top_rule == top_rule
    assert summary.top_persona == top_persona


def test_deterministic_summary_not_applicable_model():
    from agent_stress_test.orchestration.reliability import ReliabilityReport

    report = ReliabilityReport(
        score=0.0,
        total_steps=0,
        failing_steps=0,
        per_step_failure_rate=0.0,
        conversation_depth=0.0,
        model_name="task_success",
        applicable=False,
    )
    summary = deterministic_summary(report, [], [], None, None)
    assert "not measured" in summary.text
    assert "task_success" in summary.text


def test_deterministic_summary_counts_clusters_and_near_misses():
    reliability = score_run([], [])
    clusters = [_cluster("a", ["n1"]), _cluster("b", ["n2"])]
    near_misses = [NearMiss(node_id="n3", proximity=0.5, tactic="hostile")]

    summary = deterministic_summary(reliability, clusters, near_misses, None, None)

    assert "2 distinct failure patterns" in summary.text
    assert "1 near-miss came close to failing" in summary.text
    assert summary.cluster_count == 2
    assert summary.near_miss_count == 1
