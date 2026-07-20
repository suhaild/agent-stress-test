"""CLI/terminal report — a colorful Rich rendering of a completed run.

Pure presentation: every function takes already-loaded domain objects plus
an injectable ``rich.console.Console``. Nothing here talks to a ``Store``, a
provider, or the filesystem.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_stress_test.models import Cluster, Node, Run, Verdict
from agent_stress_test.orchestration.executive_summary import FixFirstItem, RunSummary
from agent_stress_test.orchestration.reliability import (
    NearMiss,
    ReliabilityReport,
    near_miss_ranking,
)
from agent_stress_test.orchestration.rule_coverage import RuleCoverage, rule_coverage
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.report.shared import (
    conversation_verdicts_by_leaf,
    executive_summary_context,
    ranked_clusters,
)

_SEVERITY_STYLE = {"critical": "bold red", "major": "yellow", "minor": "cyan"}
_ROLE_STYLE = {"system": "dim italic", "user": "bold magenta", "assistant": "bold white"}


def _score_style(score: float) -> str:
    if score >= 0.7:
        return "bold green"
    if score >= 0.4:
        return "bold yellow"
    return "bold red"


_SEVERITY_MIX_BAR_WIDTH = 24
_SEVERITY_ORDER = ("critical", "major", "minor")


def _severity_mix_bar(breakdown: dict[str, int]) -> Text:
    """A stacked bar proportional to each severity's share of failing steps.
    Uses ``#``, not a Unicode block char: a legacy Windows console (cp1252)
    raises ``UnicodeEncodeError`` on ``█``."""
    total = sum(breakdown.values())
    bar = Text()
    if not total:
        return bar
    present = [
        (severity, breakdown.get(severity, 0))
        for severity in _SEVERITY_ORDER
        if breakdown.get(severity)
    ]
    remaining = _SEVERITY_MIX_BAR_WIDTH
    for index, (severity, count) in enumerate(present):
        is_last = index == len(present) - 1
        width = remaining if is_last else max(1, round(_SEVERITY_MIX_BAR_WIDTH * count / total))
        remaining -= width
        bar.append("#" * width, style=_SEVERITY_STYLE[severity])
    return bar


def render_reliability(console: Console, run: Run, report: ReliabilityReport) -> None:
    """A headline reliability panel: the compounding score, step counts, which
    ``ScoringModel`` produced the number, and a per-severity breakdown."""
    style = _score_style(report.score) if report.applicable else "dim"
    body = Text()
    if not report.applicable:
        body.append(
            f"not measured — the '{report.model_name}' model has no relevant verdicts on this run",
            style="dim italic",
        )
    else:
        body.append(f"{report.score:.0%}\n", style=f"{style} underline")
        body.append(
            f"{report.failing_steps} of {report.total_steps} steps failed "
            f"({report.per_step_failure_rate:.0%} per-step failure rate, "
            f"compounded over ~{report.conversation_depth:.1f} turns)",
            style="dim",
        )
        if report.failing_steps:
            breakdown = ", ".join(
                f"{severity}={count}" for severity, count in report.severity_breakdown.items()
            )
            body.append(f"\nfailing steps by severity: {breakdown}\n", style="dim")
            body.append(_severity_mix_bar(report.severity_breakdown))
    console.print(
        Panel(
            body,
            title=f"Reliability (model: {report.model_name}) - {run.agent_spec.name} ({run.provider})",
            subtitle=f"run {run.id}",
            border_style=style,
        )
    )


def render_clusters(console: Console, clusters: list[Cluster], verdicts: list[Verdict]) -> None:
    """A ranked table of failure clusters (worst severity first, then size)."""
    if not clusters:
        console.print(Panel("No failure clusters - no confirmed failures.", border_style="green"))
        return

    table = Table(title="Failure Clusters", show_lines=False)
    table.add_column("Label", style="bold")
    table.add_column("Severity", justify="center")
    table.add_column("Members", justify="right")
    table.add_column("Representative node", overflow="fold")

    for entry in ranked_clusters(clusters, verdicts):
        cluster, severity = entry["cluster"], entry["severity"]
        table.add_row(
            cluster.label,
            Text(severity, style=_SEVERITY_STYLE[severity]),
            str(len(cluster.member_node_ids)),
            cluster.representative_node_id or "-",
        )
    console.print(table)


def _print_turn(console: Console, role: str, content: str) -> None:
    console.print(Text(f"{role}: ", style=_ROLE_STYLE.get(role, "")) + Text(content))


def _print_assistant_turn(console: Console, node: Node) -> None:
    """The assistant's reply, plus its instability badge."""
    line = Text("assistant: ", style=_ROLE_STYLE.get("assistant", "")) + Text(node.target_reply)
    if node.instability_score is not None:
        style = _score_style(1.0 - node.instability_score)  # invert: high instability -> red
        line.append(f"  [instability: {node.instability_score:.0%}]", style=style)
    console.print(line)


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
        _print_assistant_turn(console, node)

    node_failures = [v for v in verdicts if v.node_id == node_id and not v.passed]
    # Tool-scoped failures have no rule_id and would read as "rule: -" in the
    # generic rule panel, so they get their own line instead.
    tool_failures = [v for v in node_failures if v.scope == "tool"]
    rule_failures = [v for v in node_failures if v.scope != "tool"]

    for verdict in tool_failures:
        console.print(
            Text(
                f"tool arguments: {verdict.reason} (confidence {verdict.confidence:.0%})",
                style=_SEVERITY_STYLE[verdict.severity],
            )
        )

    if not rule_failures:
        if not tool_failures:
            console.print(Panel("No rule violation at this node.", border_style="green"))
        return

    verdict = rule_failures[0]
    style = _SEVERITY_STYLE[verdict.severity]
    body = (
        f"rule: {verdict.rule_id or '-'}\n"
        f"tier: {verdict.tier}   confidence: {verdict.confidence:.0%}\n"
        f"reason: {verdict.reason}"
    )
    console.print(Panel(body, title=f"VERDICT - {verdict.severity.upper()}", border_style=style))


def render_near_misses(console: Console, near_misses: list[NearMiss]) -> None:
    """Passing nodes that came nearest to failing, surfaced alongside the
    confirmed failures instead of buried in the tree."""
    if not near_misses:
        console.print(
            Panel("No near-misses — nothing came close to failing.", border_style="green")
        )
        return
    table = Table(title="Near Misses", show_lines=False)
    table.add_column("Tactic", style="bold")
    table.add_column("Proximity")
    table.add_column("Node", overflow="fold")
    bar_width = 20
    for near_miss in near_misses:
        filled = round(bar_width * near_miss.proximity)
        proximity = Text(f"{near_miss.proximity:.0%} ")
        proximity.append("#" * filled, style="yellow")
        proximity.append("-" * (bar_width - filled), style="dim")
        table.add_row(near_miss.tactic or "-", proximity, near_miss.node_id)
    console.print(table)


def render_conversation_verdicts(
    console: Console, tree: ConversationTree, verdicts: list[Verdict]
) -> None:
    """Whole-conversation metric verdicts (role adherence, knowledge
    retention, ...), one table per persona chain."""
    grouped = conversation_verdicts_by_leaf(verdicts)
    if not grouped:
        return
    for leaf_id, leaf_verdicts in grouped.items():
        path = tree.path_to_root(leaf_id)
        tactic = path[-1].tactic if path else None
        table = Table(
            title=f"Conversation Verdicts — {tactic or leaf_id} ({len(path)} turns)",
            show_lines=False,
        )
        table.add_column("Metric", style="bold")
        table.add_column("Result", justify="center")
        table.add_column("Severity", justify="center")
        table.add_column("Reason", overflow="fold")
        for verdict in leaf_verdicts:
            result = (
                Text("PASS", style="bold green")
                if verdict.passed
                else Text("FAIL", style="bold red")
            )
            table.add_row(
                verdict.rule_id or "-",
                result,
                Text(verdict.severity, style=_SEVERITY_STYLE[verdict.severity]),
                verdict.reason,
            )
        console.print(table)


_COVERAGE_STYLE = {
    "failed": "bold red",
    "near_miss": "yellow",
    "passed": "green",
    "never_exercised": "dim",
}


def render_summary(console: Console, summary: RunSummary, fix_first: list[FixFirstItem]) -> None:
    console.print(Panel(summary.text, title="Executive Summary", border_style="cyan"))
    if not fix_first:
        return
    table = Table(title="Fix This First", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Label", style="bold")
    table.add_column("Kind", justify="center")
    table.add_column("Severity", justify="center")
    table.add_column("Size", justify="right")
    for index, item in enumerate(fix_first, start=1):
        severity_cell = (
            Text(item.severity, style=_SEVERITY_STYLE[item.severity])
            if item.severity
            else Text("-", style="dim")
        )
        table.add_row(
            str(index),
            item.label,
            "failure" if item.kind == "cluster" else "near-miss",
            severity_cell,
            str(item.size),
        )
    console.print(table)


def render_rule_coverage(console: Console, coverage: list[RuleCoverage]) -> None:
    """Every declared rule x pass/fail/near-miss/never-exercised, so a rule
    that never fired still shows up."""
    if not coverage:
        return
    table = Table(title="Rule Coverage", show_lines=False)
    table.add_column("Rule", style="bold")
    table.add_column("Severity", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Pass / Fail", justify="right")
    for row in coverage:
        table.add_row(
            row.rule_id,
            Text(row.severity, style=_SEVERITY_STYLE[row.severity]),
            Text(row.status.replace("_", " "), style=_COVERAGE_STYLE[row.status]),
            f"{row.pass_count} / {row.fail_count}",
        )
    console.print(table)


def render_full_report(
    console: Console,
    *,
    run: Run,
    reliability: ReliabilityReport,
    clusters: list[Cluster],
    tree: ConversationTree,
    verdicts: list[Verdict],
) -> None:
    """The complete report: reliability, executive summary, ranked clusters,
    rule coverage, conversation-level verdicts, near misses, then one
    transcript per cluster."""
    near_misses = near_miss_ranking(tree.nodes(), verdicts)
    exec_ctx = executive_summary_context(tree.nodes(), verdicts, clusters, reliability, near_misses)
    render_reliability(console, run, reliability)
    render_summary(console, exec_ctx["summary"], exec_ctx["fix_first"])
    render_clusters(console, clusters, verdicts)
    render_rule_coverage(console, rule_coverage(run.agent_spec.rules, verdicts))
    render_conversation_verdicts(console, tree, verdicts)
    render_near_misses(console, near_misses)
    for cluster in clusters:
        if cluster.representative_node_id:
            render_transcript(console, tree, cluster.representative_node_id, verdicts)


__all__ = [
    "render_reliability",
    "render_summary",
    "render_clusters",
    "render_rule_coverage",
    "render_transcript",
    "render_conversation_verdicts",
    "render_near_misses",
    "render_full_report",
]
