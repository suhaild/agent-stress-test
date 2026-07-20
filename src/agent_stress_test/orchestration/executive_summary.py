"""Executive summary: a synthesized takeaway above the raw per-node/per-cluster
data. Plain rules and math only — no LLM. An opt-in LLM rephrasing of this
text lives in ``reasoning/summary.py``'s ``RunSummarizer``.
"""

from dataclasses import dataclass
from typing import Literal

from agent_stress_test.models import Cluster, Node, Severity, Verdict
from agent_stress_test.orchestration.reliability import NearMiss, ReliabilityReport
from agent_stress_test.orchestration.search import SEVERITY_WEIGHT


def _cluster_worst_severity(cluster: Cluster, verdicts: list[Verdict]) -> Severity:
    weights = [
        SEVERITY_WEIGHT[v.severity]
        for v in verdicts
        if not v.passed and v.node_id in cluster.member_node_ids
    ]
    if not weights:
        return "minor"
    best = max(weights)
    return next(sev for sev, weight in SEVERITY_WEIGHT.items() if weight == best)


@dataclass(frozen=True)
class FixFirstItem:
    """One ranked entry in the "fix this first" list: either a confirmed
    failure cluster or a near-miss, on the same priority scale."""

    kind: Literal["cluster", "near_miss"]
    label: str
    priority: float
    severity: Severity | None  # None for a near-miss — nothing has failed yet
    size: int  # cluster member count, or 1 for a single near-miss node
    representative_node_id: str | None


def fix_this_first(
    clusters: list[Cluster],
    verdicts: list[Verdict],
    near_misses: list[NearMiss],
    *,
    limit: int = 10,
) -> list[FixFirstItem]:
    """Confirmed failure clusters and near-misses, in one combined priority
    order: severity x cluster size for a cluster, proximity alone for a
    near-miss. Both share the same [0, 1] scale, so a near-miss only
    outranks a cluster that's a single minor/major node."""
    items = []
    for cluster in clusters:
        severity = _cluster_worst_severity(cluster, verdicts)
        size = len(cluster.member_node_ids)
        items.append(
            FixFirstItem(
                kind="cluster",
                label=cluster.label,
                priority=SEVERITY_WEIGHT[severity] * size,
                severity=severity,
                size=size,
                representative_node_id=cluster.representative_node_id,
            )
        )
    items += [
        FixFirstItem(
            kind="near_miss",
            label=near_miss.tactic or near_miss.node_id,
            priority=near_miss.proximity,
            severity=None,
            size=1,
            representative_node_id=near_miss.node_id,
        )
        for near_miss in near_misses
    ]
    items.sort(key=lambda item: item.priority, reverse=True)
    return items[:limit]


@dataclass(frozen=True)
class RuleCallout:
    """The rule with the most failing verdicts this run."""

    rule_id: str
    failure_count: int
    worst_severity: Severity


def top_offending_rule(verdicts: list[Verdict]) -> RuleCallout | None:
    """The rule (``scope="rule"`` only) with the most failing verdicts.
    Ties broken by worst severity, then rule id."""
    by_rule: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        if not verdict.passed and verdict.scope == "rule" and verdict.rule_id:
            by_rule.setdefault(verdict.rule_id, []).append(verdict)
    if not by_rule:
        return None

    def _rank(item: tuple[str, list[Verdict]]) -> tuple[int, float, str]:
        rule_id, failing = item
        worst = max(SEVERITY_WEIGHT[v.severity] for v in failing)
        return (len(failing), worst, rule_id)

    rule_id, failing = max(by_rule.items(), key=_rank)
    worst_severity = max(failing, key=lambda v: SEVERITY_WEIGHT[v.severity]).severity
    return RuleCallout(rule_id=rule_id, failure_count=len(failing), worst_severity=worst_severity)


@dataclass(frozen=True)
class PersonaCallout:
    """The tactic/persona whose nodes accumulated the most failing verdicts."""

    tactic: str
    failure_count: int


def top_offending_persona(nodes: list[Node], verdicts: list[Verdict]) -> PersonaCallout | None:
    """The tactic with the most nodes carrying at least one failing verdict.
    Ties broken by tactic name; ``None`` if every failing node is untagged."""
    failing_node_ids = {verdict.node_id for verdict in verdicts if not verdict.passed}
    by_tactic: dict[str, int] = {}
    for node in nodes:
        if node.id in failing_node_ids and node.tactic:
            by_tactic[node.tactic] = by_tactic.get(node.tactic, 0) + 1
    if not by_tactic:
        return None
    tactic = max(by_tactic.items(), key=lambda item: (item[1], item[0]))[0]
    return PersonaCallout(tactic=tactic, failure_count=by_tactic[tactic])


@dataclass(frozen=True)
class RunSummary:
    """The always-available, deterministic executive summary."""

    text: str
    top_rule: RuleCallout | None
    top_persona: PersonaCallout | None
    cluster_count: int
    near_miss_count: int


def deterministic_summary(
    reliability: ReliabilityReport,
    clusters: list[Cluster],
    near_misses: list[NearMiss],
    top_rule: RuleCallout | None,
    top_persona: PersonaCallout | None,
) -> RunSummary:
    """A templated takeaway paragraph — no LLM call, so it's always shown
    by default."""
    if not reliability.applicable:
        text = (
            f"Reliability not measured under the '{reliability.model_name}' model "
            "— no relevant verdicts on this run yet."
        )
    else:
        sentences = [
            f"This run scored {reliability.score:.0%} reliability "
            f"({reliability.failing_steps} of {reliability.total_steps} steps failed)."
        ]
        if top_rule is not None:
            occurrences = "occurrence" if top_rule.failure_count == 1 else "occurrences"
            sentences.append(
                f"The most common failure was rule '{top_rule.rule_id}' "
                f"({top_rule.failure_count} {occurrences}, {top_rule.worst_severity})."
            )
        if top_persona is not None:
            sentences.append(
                f"The '{top_persona.tactic}' tactic triggered the most failures "
                f"({top_persona.failure_count})."
            )
        if clusters:
            patterns = "pattern" if len(clusters) == 1 else "patterns"
            sentences.append(f"{len(clusters)} distinct failure {patterns} found.")
        if near_misses:
            misses = "near-miss" if len(near_misses) == 1 else "near-misses"
            sentences.append(f"{len(near_misses)} {misses} came close to failing.")
        text = " ".join(sentences)
    return RunSummary(
        text=text,
        top_rule=top_rule,
        top_persona=top_persona,
        cluster_count=len(clusters),
        near_miss_count=len(near_misses),
    )
