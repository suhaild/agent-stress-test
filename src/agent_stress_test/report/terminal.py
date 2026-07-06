"""CLI/terminal report — a colorful Rich rendering of a completed run.

Pure presentation: every function here takes already-loaded domain objects
(``Run``, ``ReliabilityReport``, ``Cluster``s, a ``ConversationTree``,
``Verdict``s) plus an injectable ``rich.console.Console``. Nothing here talks
to a ``Store``, a provider, or the filesystem — that keeps it trivially
testable (render to a recording console and assert on the text) and keeps the
hexagonal boundary intact (no ``litellm``, ``httpx``, or ``sqlite3`` here).
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_stress_test.models import Cluster, Run, Verdict
from agent_stress_test.orchestration.reliability import ReliabilityReport
from agent_stress_test.orchestration.search import SEVERITY_WEIGHT
from agent_stress_test.orchestration.tree import ConversationTree

_SEVERITY_STYLE = {"critical": "bold red", "major": "yellow", "minor": "cyan"}
_ROLE_STYLE = {"system": "dim italic", "user": "bold magenta", "assistant": "bold white"}


def _score_style(score: float) -> str:
    if score >= 0.7:
        return "bold green"
    if score >= 0.4:
        return "bold yellow"
    return "bold red"


def render_reliability(console: Console, run: Run, report: ReliabilityReport) -> None:
    """A headline reliability panel: the compounding score plus step counts."""
    style = _score_style(report.score)
    body = Text()
    body.append(f"{report.score:.0%}\n", style=f"{style} underline")
    body.append(
        f"{report.failing_steps} of {report.total_steps} steps failed "
        f"({report.per_step_failure_rate:.0%} per-step failure rate, "
        f"compounded over ~{report.conversation_depth:.1f} turns)",
        style="dim",
    )
    console.print(
        Panel(
            body,
            title=f"Reliability - {run.agent_spec.name} ({run.provider})",
            subtitle=f"run {run.id}",
            border_style=style,
        )
    )


def _worst_severity(cluster: Cluster, verdicts: list[Verdict]) -> str:
    weights = [
        SEVERITY_WEIGHT[v.severity]
        for v in verdicts
        if not v.passed and v.node_id in cluster.member_node_ids
    ]
    if not weights:
        return "minor"
    best = max(weights)
    return next(sev for sev, weight in SEVERITY_WEIGHT.items() if weight == best)


def render_clusters(console: Console, clusters: list[Cluster], verdicts: list[Verdict]) -> None:
    """A ranked table of failure clusters (worst severity first, then size)."""
    if not clusters:
        console.print(Panel("No failure clusters - no confirmed failures.", border_style="green"))
        return

    ranked = sorted(
        clusters,
        key=lambda c: (SEVERITY_WEIGHT[_worst_severity(c, verdicts)], len(c.member_node_ids)),
        reverse=True,
    )

    table = Table(title="Failure Clusters", show_lines=False)
    table.add_column("Label", style="bold")
    table.add_column("Severity", justify="center")
    table.add_column("Members", justify="right")
    table.add_column("Representative node", overflow="fold")

    for cluster in ranked:
        severity = _worst_severity(cluster, verdicts)
        table.add_row(
            cluster.label,
            Text(severity, style=_SEVERITY_STYLE[severity]),
            str(len(cluster.member_node_ids)),
            cluster.representative_node_id or "-",
        )
    console.print(table)


def _print_turn(console: Console, role: str, content: str) -> None:
    console.print(Text(f"{role}: ", style=_ROLE_STYLE.get(role, "")) + Text(content))


def render_transcript(
    console: Console, tree: ConversationTree, node_id: str, verdicts: list[Verdict]
) -> None:
    """The full root-to-node conversation, ending with the failing verdict."""
    path = tree.path_to_root(node_id)
    console.rule(f"Transcript - node {node_id}")

    for index, node in enumerate(path):
        if index == 0:
            for message in node.messages:
                _print_turn(console, message.role, message.content)
        else:
            probe = node.messages[-1]
            if node.tactic:
                console.print(Text(f"tactic: {node.tactic}", style="dim italic"))
            _print_turn(console, probe.role, probe.content)
        _print_turn(console, "assistant", node.target_reply)

    failing = [v for v in verdicts if v.node_id == node_id and not v.passed]
    if not failing:
        console.print(Panel("No rule violation at this node.", border_style="green"))
        return

    verdict = failing[0]
    style = _SEVERITY_STYLE[verdict.severity]
    body = (
        f"rule: {verdict.rule_id or '-'}\n"
        f"tier: {verdict.tier}   confidence: {verdict.confidence:.0%}\n"
        f"reason: {verdict.reason}"
    )
    console.print(Panel(body, title=f"VERDICT - {verdict.severity.upper()}", border_style=style))


def render_full_report(
    console: Console,
    *,
    run: Run,
    reliability: ReliabilityReport,
    clusters: list[Cluster],
    tree: ConversationTree,
    verdicts: list[Verdict],
) -> None:
    """The complete report: reliability, ranked clusters, one transcript per cluster."""
    render_reliability(console, run, reliability)
    render_clusters(console, clusters, verdicts)
    for cluster in clusters:
        if cluster.representative_node_id:
            render_transcript(console, tree, cluster.representative_node_id, verdicts)


def render_replay(
    console: Console, *, tree: ConversationTree, node_ids: list[str], verdicts: list[Verdict]
) -> None:
    """Every failing node's transcript, reproduced identically to the report."""
    seen: set[str] = set()
    if not node_ids:
        console.print(Panel("No failing nodes to replay.", border_style="green"))
        return
    for node_id in node_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        render_transcript(console, tree, node_id, verdicts)


__all__ = [
    "render_reliability",
    "render_clusters",
    "render_transcript",
    "render_full_report",
    "render_replay",
]
