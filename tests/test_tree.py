"""ConversationTree — the Blackboard's own unit tests.

Phase C2 changed ``attach_verdicts`` from replace to accumulate semantics (a
conversation-level judge attaches a second batch of verdicts to a node that
already has its own per-turn verdicts — see
``orchestration/deepeval_search.py``'s ``_ingest``); these tests pin that
behavior down directly rather than only exercising it incidentally through
search/regression/dashboard tests.
"""

import pytest

from agent_stress_test.models import Message, Node, Verdict
from agent_stress_test.orchestration.tree import ConversationTree


def _node(run_id: str = "r") -> Node:
    return Node(run_id=run_id, messages=[Message(role="user", content="hi")], target_reply="ok")


def _verdict(run_id: str, node_id: str, *, passed: bool, rule_id: str = "some-rule") -> Verdict:
    return Verdict(
        run_id=run_id,
        node_id=node_id,
        passed=passed,
        rule_id=rule_id,
        reason="reason",
        tier="rules",
        confidence=1.0,
        severity="major",
    )


def test_attach_verdicts_accumulates_across_multiple_calls():
    tree = ConversationTree("r")
    node = tree.add(_node("r"))

    tree.attach_verdicts(node.id, [_verdict("r", node.id, passed=True, rule_id="rule-a")])
    tree.attach_verdicts(node.id, [_verdict("r", node.id, passed=True, rule_id="rule-b")])

    rule_ids = {v.rule_id for v in tree.verdicts(node.id)}
    assert rule_ids == {"rule-a", "rule-b"}


def test_attach_verdicts_sets_verdict_id_from_the_first_failing_verdict_only_once():
    tree = ConversationTree("r")
    node = tree.add(_node("r"))

    first_failure = _verdict("r", node.id, passed=False, rule_id="rule-a")
    tree.attach_verdicts(node.id, [first_failure])
    second_failure = _verdict("r", node.id, passed=False, rule_id="rule-b")
    tree.attach_verdicts(node.id, [second_failure])

    # The node keeps pointing at the FIRST failure it ever received, not the
    # most recent batch's.
    assert tree.get(node.id).verdict_id == first_failure.id


def test_attach_verdicts_on_an_all_passing_second_batch_leaves_the_earlier_verdict_id_alone():
    tree = ConversationTree("r")
    node = tree.add(_node("r"))

    first_failure = _verdict("r", node.id, passed=False, rule_id="rule-a")
    tree.attach_verdicts(node.id, [first_failure])
    tree.attach_verdicts(node.id, [_verdict("r", node.id, passed=True, rule_id="rule-b")])

    assert tree.get(node.id).verdict_id == first_failure.id


def test_attach_verdicts_on_unknown_node_raises_key_error():
    tree = ConversationTree("r")

    with pytest.raises(KeyError):
        tree.attach_verdicts("no-such-node", [])
