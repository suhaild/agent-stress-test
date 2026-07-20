"""Compounding reliability score.

An agent has to survive a whole conversation, not just one turn, so the
headline score multiplies per-step success probabilities (``1 -
failure_rate``) across it: an 85%-per-step agent over 8 turns is only
``0.85 ** 8 ~= 0.27`` reliable end to end.

The search tree's node count grows with search breadth, not with real
conversation length, so compounding over the raw node count would collapse
scores toward 0 as the tree widens and make scores incomparable across
budgets. The failure rate is still estimated from every judged node, but
compounded over the *average conversation depth* (mean root-to-leaf length)
instead — the turn count a user would actually experience.

How the per-step failure rate itself is computed is a selectable
``ScoringModel`` (Strategy): ``SeverityWeightedModel`` (default),
``UnweightedFailureModel``, and ``TaskSuccessModel``. The ``(1 - p) ** d``
compounding shape is shared by every model; only ``p`` is model-specific.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar

from agent_stress_test.models import Node, Severity, Verdict
from agent_stress_test.orchestration.search import (
    SEVERITY_WEIGHT,
    failure_proximity,
    graded_proximity,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compounding_reliability(step_failure_rates: Iterable[float]) -> float:
    """Product of per-step success probabilities ``Π (1 - p_i)``, in [0, 1].
    Each rate is clamped to [0, 1] first; no steps means a score of 1.0."""
    score = 1.0
    for rate in step_failure_rates:
        score *= 1.0 - _clamp01(rate)
    return score


def average_conversation_depth(nodes: list[Node]) -> float:
    """Mean root-to-leaf path length (in turns), 0.0 for an empty tree. A
    leaf is any node no other node names as its parent."""
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


def _verdicts_by_node(verdicts: list[Verdict]) -> dict[str, list[Verdict]]:
    by_node: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        by_node.setdefault(verdict.node_id, []).append(verdict)
    return by_node


def _worst_severity(node_verdicts: list[Verdict]) -> Severity | None:
    """``None`` if every verdict passed."""
    failing = [v for v in node_verdicts if not v.passed]
    if not failing:
        return None
    return max(failing, key=lambda v: SEVERITY_WEIGHT[v.severity]).severity


def _empty_breakdown() -> dict[Severity, int]:
    return {"critical": 0, "major": 0, "minor": 0}


@dataclass(frozen=True)
class ModelResult:
    """What a ``ScoringModel`` computes before ``score_run``'s shared
    compounding step."""

    total_steps: int
    failing_steps: int
    per_step_failure_rate: float
    severity_breakdown: dict[Severity, int]
    applicable: bool = True


class ScoringModel(ABC):
    """Strategy: which nodes count as a "step" and how the per-step failure
    rate is derived from them."""

    name: ClassVar[str]

    @abstractmethod
    def evaluate(self, nodes: list[Node], verdicts: list[Verdict]) -> ModelResult: ...


class UnweightedFailureModel(ScoringModel):
    """A step fails if any verdict on it failed, full stop — a critical
    violation and a minor nit count the same toward the rate. Not the
    default; pass ``model=UnweightedFailureModel()`` explicitly to use it."""

    name = "unweighted"

    def evaluate(self, nodes: list[Node], verdicts: list[Verdict]) -> ModelResult:
        by_node = _verdicts_by_node(verdicts)
        total_steps = len(nodes)
        breakdown = _empty_breakdown()
        failing_steps = 0
        for node in nodes:
            worst = _worst_severity(by_node.get(node.id, []))
            if worst is not None:
                failing_steps += 1
                breakdown[worst] += 1
        rate = failing_steps / total_steps if total_steps else 0.0
        return ModelResult(total_steps, failing_steps, rate, breakdown)


class SeverityWeightedModel(ScoringModel):
    """The default model. Same failing/passing steps as
    ``UnweightedFailureModel``, but the per-step failure rate is the mean
    ``SEVERITY_WEIGHT`` of each step's worst failure rather than a flat
    1.0/0.0, so a run of only minor nits scores higher than one with the
    same failure count but all critical."""

    name = "severity_weighted"

    def evaluate(self, nodes: list[Node], verdicts: list[Verdict]) -> ModelResult:
        by_node = _verdicts_by_node(verdicts)
        total_steps = len(nodes)
        breakdown = _empty_breakdown()
        failing_steps = 0
        weight_sum = 0.0
        for node in nodes:
            node_verdicts = by_node.get(node.id, [])
            weight_sum += failure_proximity(node_verdicts)
            worst = _worst_severity(node_verdicts)
            if worst is not None:
                failing_steps += 1
                breakdown[worst] += 1
        rate = weight_sum / total_steps if total_steps else 0.0
        return ModelResult(total_steps, failing_steps, rate, breakdown)


class TaskSuccessModel(ScoringModel):
    """Fed by ``TaskCompletionMetric`` (``scope="task"`` verdicts) instead
    of rule/tool violations. Scoped to nodes that actually have a
    task-completion verdict; ``applicable=False`` when none do, so a
    caller/renderer can show "not measured" instead of a misleading 100%."""

    name = "task_success"

    def evaluate(self, nodes: list[Node], verdicts: list[Verdict]) -> ModelResult:
        by_node = _verdicts_by_node(verdicts)
        breakdown = _empty_breakdown()
        total_steps = 0
        failing_steps = 0
        for node in nodes:
            task_verdicts = [v for v in by_node.get(node.id, []) if v.scope == "task"]
            if not task_verdicts:
                continue
            total_steps += 1
            worst = _worst_severity(task_verdicts)
            if worst is not None:
                failing_steps += 1
                breakdown[worst] += 1
        if total_steps == 0:
            return ModelResult(0, 0, 0.0, breakdown, applicable=False)
        rate = failing_steps / total_steps
        return ModelResult(total_steps, failing_steps, rate, breakdown)


@dataclass(frozen=True)
class ReliabilityReport:
    """The headline reliability score plus the counts it was derived from."""

    score: float
    total_steps: int
    failing_steps: int
    per_step_failure_rate: float
    conversation_depth: float
    model_name: str = SeverityWeightedModel.name
    severity_breakdown: dict[Severity, int] = field(default_factory=_empty_breakdown)
    applicable: bool = True


# Kept below report/dashboard's "fail" color threshold (< 0.4) so a capped
# score always renders as unambiguously bad, not a borderline "warn".
CRITICAL_FAILURE_SCORE_CAP = 0.3


def apply_mandatory_minimum_cap(
    score: float,
    severity_breakdown: dict[Severity, int],
    *,
    cap: float = CRITICAL_FAILURE_SCORE_CAP,
) -> float:
    """Cap ``score`` at ``cap`` when ``severity_breakdown`` shows at least
    one critical failure. Never raises a score, only ever lowers it."""
    if severity_breakdown.get("critical", 0) > 0:
        return min(score, cap)
    return score


def score_run(
    nodes: list[Node], verdicts: list[Verdict], *, model: ScoringModel | None = None
) -> ReliabilityReport:
    """Compounding reliability of a run from its nodes and verdicts.
    ``model`` selects how the per-step failure rate is computed; defaults
    to ``SeverityWeightedModel``. The result is then passed through
    ``apply_mandatory_minimum_cap``."""
    resolved_model = model if model is not None else SeverityWeightedModel()
    result = resolved_model.evaluate(nodes, verdicts)
    conversation_depth = average_conversation_depth(nodes)
    score = (1.0 - _clamp01(result.per_step_failure_rate)) ** conversation_depth
    score = apply_mandatory_minimum_cap(score, result.severity_breakdown)

    return ReliabilityReport(
        score=score,
        total_steps=result.total_steps,
        failing_steps=result.failing_steps,
        per_step_failure_rate=result.per_step_failure_rate,
        conversation_depth=conversation_depth,
        model_name=resolved_model.name,
        severity_breakdown=result.severity_breakdown,
        applicable=result.applicable,
    )


@dataclass(frozen=True)
class NearMiss:
    """One passing node that came close to failing."""

    node_id: str
    proximity: float
    tactic: str | None


def near_miss_ranking(
    nodes: list[Node], verdicts: list[Verdict], *, limit: int = 5
) -> list[NearMiss]:
    """The passing nodes with the highest ``graded_proximity``, ranked
    descending and capped to ``limit``. Computed directly from
    ``nodes``/``verdicts`` rather than ``SearchResult.exploration_detail``
    (which only ``GreedyBestFirstSearch`` populates), so this works for any
    ``SearchStrategy``'s output."""
    by_node = _verdicts_by_node(verdicts)
    scored: list[NearMiss] = []
    for node in nodes:
        node_verdicts = by_node.get(node.id, [])
        if any(not v.passed for v in node_verdicts):
            continue  # an outright failure, not a near-miss
        proximity = graded_proximity(node_verdicts)
        if proximity > 0.0:
            scored.append(NearMiss(node_id=node.id, proximity=proximity, tactic=node.tactic))
    scored.sort(key=lambda near_miss: near_miss.proximity, reverse=True)
    return scored[:limit]
