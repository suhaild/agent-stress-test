"""Pure ranking/grouping helpers shared by both report front ends —
``report/terminal.py`` (the CLI) and ``report/dashboard/*`` (the web
dashboard) — so cluster ordering and conversation-verdict grouping can never
drift between the two. Nothing here talks to a ``Store``, a provider, or the
filesystem; same isolation contract as ``terminal.py`` itself.
"""

from agent_stress_test.models import Cluster, Node, Verdict
from agent_stress_test.orchestration.cross_run import TrendPoint
from agent_stress_test.orchestration.executive_summary import (
    deterministic_summary,
    fix_this_first,
    top_offending_persona,
    top_offending_rule,
)
from agent_stress_test.orchestration.reliability import NearMiss, ReliabilityReport
from agent_stress_test.orchestration.search import SEVERITY_WEIGHT


def _worst_severity(cluster: Cluster, verdicts: list[Verdict]) -> str:
    """The highest-weighted severity among a cluster's failing member verdicts."""
    weights = [
        SEVERITY_WEIGHT[v.severity]
        for v in verdicts
        if not v.passed and v.node_id in cluster.member_node_ids
    ]
    if not weights:
        return "minor"
    best = max(weights)
    return next(sev for sev, weight in SEVERITY_WEIGHT.items() if weight == best)


def ranked_clusters(clusters: list[Cluster], verdicts: list[Verdict]) -> list[dict]:
    """Clusters worst-severity-first then largest-first, each paired with its
    severity — pre-computed so a renderer just iterates, no ranking logic of
    its own."""
    ranked = sorted(
        clusters,
        key=lambda c: (SEVERITY_WEIGHT[_worst_severity(c, verdicts)], len(c.member_node_ids)),
        reverse=True,
    )
    return [{"cluster": c, "severity": _worst_severity(c, verdicts)} for c in ranked]


def trend_chart_points(trend: list[TrendPoint]) -> list[dict]:
    """``TrendPoint``s as plain JSON-safe dicts (a Chart.js label/score/run_id
    triple) — the one conversion Chart.js's data option actually needs;
    everything else about a ``TrendPoint`` (``started_at`` as a real
    ``datetime``) stays on the dataclass for any other renderer."""
    return [
        {
            "label": point.started_at.strftime("%b %d") if point.started_at else point.run_id[:8],
            "score": round(point.score * 100, 1),
            "run_id": point.run_id,
        }
        for point in trend
    ]


def executive_summary_context(
    nodes: list[Node],
    verdicts: list[Verdict],
    clusters: list[Cluster],
    reliability: ReliabilityReport,
    near_misses: list[NearMiss],
) -> dict:
    """Phase RE2's executive-summary template context — the one place that
    folds ``executive_summary.py``'s deterministic pieces (top-rule/persona
    callouts, the summary paragraph, the combined fix-this-first ranking)
    together, so both ``server.py``'s route and ``live_events.py``'s terminal
    panel build it identically."""
    top_rule = top_offending_rule(verdicts)
    top_persona = top_offending_persona(nodes, verdicts)
    return {
        "summary": deterministic_summary(reliability, clusters, near_misses, top_rule, top_persona),
        "fix_first": fix_this_first(clusters, verdicts, near_misses),
    }


def conversation_verdicts_by_leaf(verdicts: list[Verdict]) -> dict[str, list[Verdict]]:
    """Group ``scope="conversation"`` verdicts by their leaf node id — see
    ``Verdict.scope``'s docstring: that id is the persona chain's LEAF node,
    since ``tree.path_to_root()`` of it reconstructs exactly the conversation
    each group judged."""
    grouped: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        if verdict.scope == "conversation":
            grouped.setdefault(verdict.node_id, []).append(verdict)
    return grouped
