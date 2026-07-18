"""Pure ranking/grouping helpers shared by both report front ends —
``report/terminal.py`` (the CLI) and ``report/dashboard/*`` (the web
dashboard) — so cluster ordering and conversation-verdict grouping can never
drift between the two. Nothing here talks to a ``Store``, a provider, or the
filesystem; same isolation contract as ``terminal.py`` itself.
"""

from agent_stress_test.models import Cluster, Verdict
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
