"""Compounding reliability score.

A stress-tested agent has to survive a whole conversation, not just one turn.
So the headline reliability number is *compounding*: multiply the per-step
success probabilities (``1 - failure_rate``) across the run. An 85%-per-step
agent over 8 steps is only ``0.85 ** 8 ≈ 0.27`` reliable end to end — small
per-step slips compound into a big end-to-end risk.

Pure math over the persisted structures (nodes + verdicts), so the same score
falls out of an in-memory tree or of rows reloaded from the store.
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


@dataclass(frozen=True)
class ReliabilityReport:
    """The headline reliability score plus the counts it was derived from."""

    score: float
    total_steps: int
    failing_steps: int
    per_step_failure_rate: float


def score_run(nodes: list[Node], verdicts: list[Verdict]) -> ReliabilityReport:
    """Compounding reliability of a run from its nodes and verdicts.

    Each judged node is one step; a node fails if any verdict on it failed. The
    observed per-step failure rate ``p`` is applied uniformly across the steps,
    so ``score = (1 - p) ** total_steps``.
    """
    total_steps = len(nodes)
    failing_node_ids = {verdict.node_id for verdict in verdicts if not verdict.passed}
    failing_steps = sum(1 for node in nodes if node.id in failing_node_ids)

    per_step_failure_rate = failing_steps / total_steps if total_steps else 0.0
    score = compounding_reliability([per_step_failure_rate] * total_steps)

    return ReliabilityReport(
        score=score,
        total_steps=total_steps,
        failing_steps=failing_steps,
        per_step_failure_rate=per_step_failure_rate,
    )
