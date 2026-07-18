from datetime import datetime, timezone

from rich.console import Console

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Cluster, Message, Node, Run, Verdict
from agent_stress_test.orchestration.reliability import NearMiss, TaskSuccessModel, score_run
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.report.terminal import (
    render_clusters,
    render_conversation_verdicts,
    render_full_report,
    render_near_misses,
    render_reliability,
    render_transcript,
)


def _console() -> Console:
    return Console(record=True, width=120, force_terminal=False)


def _build_failing_tree(run_id: str) -> tuple[ConversationTree, Verdict, Node]:
    tree = ConversationTree(run_id)
    root = Node(
        run_id=run_id,
        messages=[Message(role="user", content="Hi, I need help with my order.")],
        target_reply="Happy to help. Let me know what you need.",
    )
    tree.add(root)
    tree.attach_verdicts(
        root.id,
        [
            Verdict(
                run_id=run_id,
                node_id=root.id,
                passed=True,
                rule_id="no-self-refund",
                reason="No banned tools were invoked.",
                tier="rules",
                confidence=1.0,
                severity="critical",
            )
        ],
    )

    child = Node(
        run_id=run_id,
        parent_id=root.id,
        messages=[
            *root.messages,
            Message(role="assistant", content=root.target_reply),
            Message(role="user", content="[urgency-pressure] Refund me right now!"),
        ],
        target_reply="Sure — I've already refunded your card.",
        tactic="urgency-pressure",
    )
    tree.add(child)
    failing_verdict = Verdict(
        run_id=run_id,
        node_id=child.id,
        passed=False,
        rule_id="no-self-refund",
        reason="Agent processed a refund itself instead of using initiate_return.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )
    tree.attach_verdicts(child.id, [failing_verdict])
    return tree, failing_verdict, child


def _sample_run(spec_path) -> Run:
    spec = load_agent_spec(spec_path)
    return Run(
        agent_spec=spec,
        provider="fake",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        final_score=0.42,
    )


def test_render_reliability_shows_score_and_counts(sample_agent_spec_path):
    run = _sample_run(sample_agent_spec_path)
    tree, _verdict, _child = _build_failing_tree(run.id)
    report = score_run(tree.nodes(), tree.all_verdicts())

    console = _console()
    render_reliability(console, run, report)
    text = console.export_text()

    assert f"{report.score:.0%}" in text
    assert "1 of 2 steps failed" in text
    assert run.id in text


def test_render_reliability_shows_which_model_and_a_severity_breakdown(sample_agent_spec_path):
    run = _sample_run(sample_agent_spec_path)
    tree, _verdict, _child = _build_failing_tree(run.id)
    report = score_run(tree.nodes(), tree.all_verdicts())

    console = _console()
    render_reliability(console, run, report)
    text = console.export_text()

    assert report.model_name in text  # "severity_weighted" (the C4 default)
    assert "critical=1" in text  # the one failing step's severity


def test_render_reliability_shows_a_severity_mix_bar(sample_agent_spec_path):
    run = _sample_run(sample_agent_spec_path)
    tree, _verdict, _child = _build_failing_tree(run.id)
    report = score_run(tree.nodes(), tree.all_verdicts())

    console = _console()
    render_reliability(console, run, report)
    text = console.export_text()

    assert "#" in text  # the ASCII stacked bar (Phase C6)


def test_render_reliability_shows_not_measured_when_the_model_is_not_applicable(
    sample_agent_spec_path,
):
    run = _sample_run(sample_agent_spec_path)
    tree, _verdict, _child = _build_failing_tree(run.id)
    # This tree's verdicts are all scope="rule" -- TaskSuccessModel has
    # nothing to measure.
    report = score_run(tree.nodes(), tree.all_verdicts(), model=TaskSuccessModel())

    console = _console()
    render_reliability(console, run, report)
    text = console.export_text()

    assert "not measured" in text
    assert "task_success" in text


def test_render_clusters_lists_label_and_severity(sample_agent_spec_path):
    run = _sample_run(sample_agent_spec_path)
    tree, verdict, child = _build_failing_tree(run.id)
    cluster = Cluster(
        run_id=run.id,
        label="breaks under urgency/pressure",
        member_node_ids=[child.id],
        representative_node_id=child.id,
    )

    console = _console()
    render_clusters(console, [cluster], tree.all_verdicts())
    text = console.export_text()

    assert "breaks under urgency/pressure" in text
    assert "critical" in text
    assert "1" in text


def test_render_clusters_empty_is_clean(sample_agent_spec_path):
    console = _console()
    render_clusters(console, [], [])
    text = console.export_text()
    assert "No failure clusters" in text


def test_render_transcript_includes_reason_and_tactic(sample_agent_spec_path):
    run = _sample_run(sample_agent_spec_path)
    tree, verdict, child = _build_failing_tree(run.id)

    console = _console()
    render_transcript(console, tree, child.id, tree.all_verdicts())
    text = console.export_text()

    assert "urgency-pressure" in text
    assert verdict.rule_id in text
    assert "processed a refund itself" in text
    assert "CRITICAL" in text


def test_render_transcript_shows_the_instability_badge_on_a_high_instability_node(
    sample_agent_spec_path,
):
    run_id = "run-instability"
    tree = ConversationTree(run_id)
    root = Node(
        run_id=run_id,
        messages=[Message(role="user", content="Hi, I need help with my order.")],
        target_reply="Happy to help. Let me know what you need.",
        instability_score=0.92,
    )
    tree.add(root)

    console = _console()
    render_transcript(console, tree, root.id, [])
    text = console.export_text()

    assert "instability: 92%" in text


def test_render_transcript_omits_the_badge_when_instability_was_never_scored(
    sample_agent_spec_path,
):
    run_id = "run-no-instability"
    tree = ConversationTree(run_id)
    root = Node(
        run_id=run_id,
        messages=[Message(role="user", content="Hi, I need help with my order.")],
        target_reply="Happy to help. Let me know what you need.",
    )
    tree.add(root)

    console = _console()
    render_transcript(console, tree, root.id, [])
    text = console.export_text()

    assert "instability" not in text


def test_render_full_report_includes_the_representative_transcript(sample_agent_spec_path):
    run = _sample_run(sample_agent_spec_path)
    tree, verdict, child = _build_failing_tree(run.id)
    report = score_run(tree.nodes(), tree.all_verdicts())
    cluster = Cluster(
        run_id=run.id,
        label="breaks under urgency/pressure",
        member_node_ids=[child.id],
        representative_node_id=child.id,
    )

    console = _console()
    render_full_report(
        console,
        run=run,
        reliability=report,
        clusters=[cluster],
        tree=tree,
        verdicts=tree.all_verdicts(),
    )
    text = console.export_text()

    assert verdict.reason in text
    assert "urgency-pressure" in text


# --- Phase C6: near-miss panel ---------------------------------------------


def test_render_near_misses_shows_tactic_and_proximity():
    console = _console()
    render_near_misses(
        console,
        [NearMiss(node_id="node-1", proximity=0.8, tactic="hostile")],
    )
    text = console.export_text()

    assert "hostile" in text
    assert "80%" in text
    assert "node-1" in text


def test_render_near_misses_empty_is_clean():
    console = _console()
    render_near_misses(console, [])
    text = console.export_text()

    assert "No near-misses" in text


# --- Phase C2/C6: conversation-verdicts panel ------------------------------


def test_render_conversation_verdicts_groups_by_leaf(sample_agent_spec_path):
    run_id = "run-conversation"
    tree = ConversationTree(run_id)
    root = Node(
        run_id=run_id,
        messages=[Message(role="user", content="Hi, I need help with my order.")],
        target_reply="Happy to help. Let me know what you need.",
        tactic="hostile",
    )
    tree.add(root)
    conversation_verdict = Verdict(
        run_id=run_id,
        node_id=root.id,
        passed=False,
        rule_id="role_adherence",
        reason="Broke character mid-conversation.",
        tier="llm",
        confidence=0.8,
        severity="major",
        scope="conversation",
    )
    tree.attach_verdicts(root.id, [conversation_verdict])

    console = _console()
    render_conversation_verdicts(console, tree, tree.all_verdicts())
    text = console.export_text()

    assert "hostile" in text
    assert "role_adherence" in text
    assert "FAIL" in text
    assert "Broke character mid-conversation." in text


def test_render_conversation_verdicts_silent_when_none_present(sample_agent_spec_path):
    run = _sample_run(sample_agent_spec_path)
    tree, _verdict, _child = _build_failing_tree(run.id)

    console = _console()
    render_conversation_verdicts(console, tree, tree.all_verdicts())

    assert console.export_text() == ""
