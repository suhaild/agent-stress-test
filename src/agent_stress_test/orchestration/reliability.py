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

Phase C4 — how the per-step failure rate itself is computed is now a
selectable ``ScoringModel`` (Strategy): ``SeverityWeightedModel`` (a step's
failure weighs by its worst verdict's severity, so a run of only minor nits
scores higher than one with the same failure count but all critical — the
DEFAULT, since it's the more meaningful headline number and is always
computable from any run's ordinary rule/tool verdicts), ``UnweightedFailureModel``
(the ORIGINAL model — any failing verdict counts a step as failed, full
stop, regardless of severity — kept available, not deleted, for anyone who
wants the flat count instead), and ``TaskSuccessModel`` (scoped to Phase-C's
``TaskCompletionMetric`` verdicts instead of rule/tool violations — only
meaningful when a run actually enabled ``task_completion``). The
``(1 - p) ** d`` compounding shape itself is shared by every model; only
``p`` (and, for ``TaskSuccessModel``, which nodes count as a "step" at all)
is model-specific.

Choosing ``SeverityWeightedModel`` as the default means a run's headline
score is NOT numerically comparable to one computed before this change —
that tradeoff was made deliberately (see ``tests/test_reliability.py``,
which was updated accordingly): pass ``model=UnweightedFailureModel()``
explicitly wherever the original flat-count number is needed instead.

Phase C6 adds ``near_miss_ranking`` — not a scoring model, just a read of the
same nodes/verdicts through C5's ``graded_proximity`` lens, surfaced by both
report surfaces (``report/terminal.py``, the dashboard) alongside the
confirmed failures.
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


def _verdicts_by_node(verdicts: list[Verdict]) -> dict[str, list[Verdict]]:
    by_node: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        by_node.setdefault(verdict.node_id, []).append(verdict)
    return by_node


def _worst_severity(node_verdicts: list[Verdict]) -> Severity | None:
    """The severity of the worst-failing verdict among ``node_verdicts``, or
    ``None`` if every one of them passed."""
    failing = [v for v in node_verdicts if not v.passed]
    if not failing:
        return None
    return max(failing, key=lambda v: SEVERITY_WEIGHT[v.severity]).severity


def _empty_breakdown() -> dict[Severity, int]:
    return {"critical": 0, "major": 0, "minor": 0}


@dataclass(frozen=True)
class ModelResult:
    """What a ``ScoringModel`` computes before the shared compounding step
    (``score_run`` takes that step itself, identically for every model)."""

    total_steps: int
    failing_steps: int
    per_step_failure_rate: float
    severity_breakdown: dict[Severity, int]
    applicable: bool = True


class ScoringModel(ABC):
    """Strategy: which nodes count as a "step" and how the per-step failure
    rate is derived from them. Every model still compounds via
    ``(1 - p) ** d`` (see ``score_run``) — only this part is model-specific.
    """

    name: ClassVar[str]

    @abstractmethod
    def evaluate(self, nodes: list[Node], verdicts: list[Verdict]) -> ModelResult: ...


class UnweightedFailureModel(ScoringModel):
    """The original model: a step fails if ANY verdict on it failed, full
    stop — a critical rule violation and a barely-there minor nit count
    exactly the same toward the rate. No longer the default (see
    ``SeverityWeightedModel``) — kept available, not deleted, for anyone who
    wants the original flat-count number back (pass
    ``model=UnweightedFailureModel()`` to ``score_run`` explicitly).
    """

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
    """The DEFAULT model (see ``score_run``). Same failing/passing steps as
    ``UnweightedFailureModel`` — the per-step failure RATE is the mean
    ``SEVERITY_WEIGHT`` of each step's worst failure (0.0 for a passing step)
    instead of a flat 1.0/0.0, so a run whose only failures are minor nits
    scores meaningfully higher than one with the same failure COUNT but all
    critical (see ``orchestration/search.py``'s ``failure_proximity``, reused
    here rather than reimplemented).
    """

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
    """Fed by Phase-C's ``TaskCompletionMetric`` (``scope="task"`` verdicts)
    instead of rule/tool violations.

    A step is scoped to nodes that actually HAVE a task-completion verdict —
    ``task_completion`` is opt-in (see ``build_runner``), so most runs have
    none at all. ``applicable=False`` when no node in the run has one, so a
    caller/renderer can show "not measured" instead of a misleading 100%
    (mirrors ``TokenUsage.pricing_unavailable``'s "the number technically
    computed but don't trust it" idiom).
    """

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


def score_run(
    nodes: list[Node], verdicts: list[Verdict], *, model: ScoringModel | None = None
) -> ReliabilityReport:
    """Compounding reliability of a run from its nodes and verdicts.

    ``model`` selects how the per-step failure rate is computed (see the
    module docstring); left ``None``, ``SeverityWeightedModel`` is used — a
    run's headline score now weighs failures by severity rather than
    counting them flat. Pass ``model=UnweightedFailureModel()`` for the
    original flat-count number instead.
    """
    resolved_model = model if model is not None else SeverityWeightedModel()
    result = resolved_model.evaluate(nodes, verdicts)
    conversation_depth = average_conversation_depth(nodes)
    score = (1.0 - _clamp01(result.per_step_failure_rate)) ** conversation_depth

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
    """One passing node that came close to failing — see ``near_miss_ranking``."""

    node_id: str
    proximity: float
    tactic: str | None


def near_miss_ranking(
    nodes: list[Node], verdicts: list[Verdict], *, limit: int = 5
) -> list[NearMiss]:
    """The passing nodes with the highest ``graded_proximity`` (Phase C5),
    ranked descending and capped to ``limit`` — "how close did we come",
    reported alongside the confirmed failures rather than buried in the tree.

    Computed directly from ``nodes``/``verdicts`` (the same persisted shape
    every report surface already has), not from
    ``SearchResult.exploration_detail`` — that field is only ever populated
    by ``GreedyBestFirstSearch``'s result (``None`` for other strategies —
    see its docstring), while this works for a run produced by *any*
    ``SearchStrategy``, including the ``DeepEvalConversationSearch`` engine
    ``build_runner`` actually wires by default. A node with 0.0 proximity (a
    clean pass with nothing to report) is excluded — there's nothing "near"
    about it.
    """
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
