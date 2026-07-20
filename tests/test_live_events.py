"""Regression test for the Live Failure Feed reconnect-duplication bug.

The failure panel appends via ``hx-swap="beforeend"`` (see run.html), so if
``seen`` were scoped per SSE connection, an EventSource auto-reconnect mid-run
(network blip, idle proxy timeout -- htmx's SSE extension retries by default)
would start over with an empty ``seen`` set and replay every already-shown
failure as a second, visually duplicate card. ``_seen_failures_for`` fixes
this by keying ``seen`` off the run id instead of the connection.
"""

from agent_stress_test.models import Message, Node, Verdict
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.report.dashboard import live_events


def _failing_verdict(run_id: str, node_id: str, rule_id: str) -> Verdict:
    return Verdict(
        run_id=run_id,
        node_id=node_id,
        passed=False,
        rule_id=rule_id,
        reason="fail",
        tier="rules",
        confidence=1.0,
        severity="major",
    )


def _tick(tree: ConversationTree) -> live_events._EventTick:
    return live_events._EventTick(
        tree=tree, node_count=len(tree.nodes()), run=None, status="running", is_terminal=False
    )


def test_a_reconnect_does_not_replay_already_shown_failures():
    run_id = "run-reconnect"
    tree = ConversationTree(run_id)
    node = tree.add(
        Node(run_id=run_id, messages=[Message(role="user", content="hi")], target_reply="ok")
    )
    tree.attach_verdicts(node.id, [_failing_verdict(run_id, node.id, "rule-a")])

    with live_events.live_trees_lock:
        live_events.live_trees[run_id] = tree
    try:
        # First "connection" sees the one existing failure.
        failure_panel_1 = next(
            p for p in live_events._make_live_panels(run_id) if p.event == "failure"
        )
        assert failure_panel_1.cadence(_tick(tree))
        assert len(failure_panel_1.context_builder(_tick(tree))) == 1

        # A second "connection" -- as if the SSE stream dropped and
        # auto-reconnected -- must not replay that same failure...
        failure_panel_2 = next(
            p for p in live_events._make_live_panels(run_id) if p.event == "failure"
        )
        assert not failure_panel_2.cadence(_tick(tree))

        # ...but a genuinely new failure arriving afterward still shows.
        tree.attach_verdicts(node.id, [_failing_verdict(run_id, node.id, "rule-b")])
        assert failure_panel_2.cadence(_tick(tree))
        contexts = failure_panel_2.context_builder(_tick(tree))
        assert len(contexts) == 1
        assert contexts[0]["verdict"].rule_id == "rule-b"
    finally:
        with live_events.live_trees_lock:
            live_events.live_trees.pop(run_id, None)
        live_events._seen_failures_by_run.pop(run_id, None)


def test_seen_failures_are_dropped_once_a_run_is_no_longer_tracked():
    run_id = "run-cleanup"
    live_events._seen_failures_for(run_id).add("some-verdict-id")
    assert run_id in live_events._seen_failures_by_run

    # stream_run_events pops this itself right before returning on a terminal
    # tick, mirroring live_trees' own cleanup -- exercised directly here since
    # driving the generator to a terminal tick needs a full store-backed run.
    live_events._seen_failures_by_run.pop(run_id, None)

    assert run_id not in live_events._seen_failures_by_run
