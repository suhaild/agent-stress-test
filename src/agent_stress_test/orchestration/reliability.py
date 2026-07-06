"""Compounding reliability score.

A stress-tested agent has to survive a whole conversation, not just one turn.
So the headline reliability number is *compounding*: multiply the per-step
success probabilities (``1 - failure_rate``) across a conversation. An
85%-per-step agent over 8 turns is only ``0.85 ** 8 ≈ 0.27`` reliable end to
end — small per-step slips compound into a big end-to-end risk.

The search tree holds many branches (one per tactic per expansion), not one
conversation — its total node count grows with search breadth (``--budget``
x tactics), not with how long a real conversation runs. Compounding over that
raw count would make the score collapse toward 0 for any nonzero failure rate
once the tree gets wide, regardless of true reliability, and would make scores
incomparable across different budgets. So the per-step failure rate is still
estimated from every judged node (more branches = a better estimate), but it
is compounded over the *average conversation depth* (mean root-to-leaf path
length) instead — the turn count a user would actually experience.

Pure math over the persisted structures (nodes + verdicts), so the same score
falls out of an in-memory tree or of rows reloaded from the store: depth is
recomputed from each node's ``parent_id``, with no dependency on ConversationTree.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from agent_stress_test.models import Node, Verdict


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compounding_reliability(step_failure_rates: Iterable[float]) -> float:
    """Product of per-step success probabilities ``Π (1 - p_i)``, in [0, 1].

    Each rate is clamped to [0, 1] first. No steps means nothing could fail, so
    the score is 1.0.
    """
    score = 1.0
    for rate in step_failure_rates:
        score *= 1.0 - _clamp01(rate)
    return score


def average_conversation_depth(nodes: list[Node]) -> float:
    """Mean root-to-leaf path length (in turns), 0.0 for an empty tree.

    A leaf is any node no other node names as its parent. Depth is computed by
    walking each leaf's ``parent_id`` chain, so it needs only the node list —
    the same structure that is persisted and reloaded.
    """
    if not nodes:
        return 0.0
    by_id = {node.id: node for node in nodes}
    parent_ids = {node.parent_id for node in nodes if node.parent_id is not None}
    leaves = [node for node in nodes if node.id not in parent_ids]

    def depth(node: Node) -> int:
        length = 1
        current = node
        while current.parent_id is not None:
            current = by_id[current.parent_id]
            length += 1
        return length

    return sum(depth(leaf) for leaf in leaves) / len(leaves)


@dataclass(frozen=True)
class ReliabilityReport:
    """The headline reliability score plus the counts it was derived from."""

    score: float
    total_steps: int
    failing_steps: int
    per_step_failure_rate: float
    conversation_depth: float


def score_run(nodes: list[Node], verdicts: list[Verdict]) -> ReliabilityReport:
    """Compounding reliability of a run from its nodes and verdicts.

    Each judged node is one candidate step; a node fails if any verdict on it
    failed. The observed per-step failure rate ``p`` is estimated across every
    judged node in the tree, then compounded over the average conversation
    depth ``d``, so ``score = (1 - p) ** d``.
    """
    total_steps = len(nodes)
    failing_node_ids = {verdict.node_id for verdict in verdicts if not verdict.passed}
    failing_steps = sum(1 for node in nodes if node.id in failing_node_ids)

    per_step_failure_rate = failing_steps / total_steps if total_steps else 0.0
    conversation_depth = average_conversation_depth(nodes)
    score = (1.0 - _clamp01(per_step_failure_rate)) ** conversation_depth

    return ReliabilityReport(
        score=score,
        total_steps=total_steps,
        failing_steps=failing_steps,
        per_step_failure_rate=per_step_failure_rate,
        conversation_depth=conversation_depth,
    )
