"""Multi-format report export: one shared ``ReportBundle`` that the CLI's
JSON/Markdown export and the dashboard's export routes both render from, so
every surface presents the same numbers.

Pure presentation: nothing here talks to a ``Store``, a provider, or the
filesystem.
"""

import json
from dataclasses import asdict, dataclass

from agent_stress_test.models import Cluster, Run, Verdict
from agent_stress_test.orchestration.executive_summary import FixFirstItem, RunSummary
from agent_stress_test.orchestration.reliability import (
    NearMiss,
    ReliabilityReport,
    near_miss_ranking,
    score_run,
)
from agent_stress_test.orchestration.rule_coverage import RuleCoverage, rule_coverage
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.report.shared import (
    conversation_verdicts_by_leaf,
    executive_summary_context,
    ranked_clusters,
)


@dataclass(frozen=True)
class ReportBundle:
    """Everything any export format needs for one completed run."""

    run: Run
    tree: ConversationTree
    verdicts: list[Verdict]
    clusters: list[Cluster]
    reliability: ReliabilityReport
    ranked_clusters: list[dict]
    near_misses: list[NearMiss]
    conversation_groups: dict[str, list[Verdict]]
    rule_coverage: list[RuleCoverage]
    summary: RunSummary
    fix_first: list[FixFirstItem]


def build_report_bundle(
    run: Run, tree: ConversationTree, verdicts: list[Verdict], clusters: list[Cluster]
) -> ReportBundle:
    """Compute a run's full report once; every export format derives from
    identical inputs."""
    reliability = score_run(tree.nodes(), verdicts)
    near_misses = near_miss_ranking(tree.nodes(), verdicts)
    exec_ctx = executive_summary_context(tree.nodes(), verdicts, clusters, reliability, near_misses)
    return ReportBundle(
        run=run,
        tree=tree,
        verdicts=verdicts,
        clusters=clusters,
        reliability=reliability,
        ranked_clusters=ranked_clusters(clusters, verdicts),
        near_misses=near_misses,
        conversation_groups=conversation_verdicts_by_leaf(verdicts),
        rule_coverage=rule_coverage(run.agent_spec.rules, verdicts),
        summary=exec_ctx["summary"],
        fix_first=exec_ctx["fix_first"],
    )


def to_json_dict(bundle: ReportBundle) -> dict:
    """A fully JSON-safe dict: Pydantic models via ``model_dump``, plain
    dataclasses via ``dataclasses.asdict``."""
    return {
        "run": bundle.run.model_dump(mode="json"),
        "reliability": asdict(bundle.reliability),
        "clusters": [
            {"cluster": entry["cluster"].model_dump(mode="json"), "severity": entry["severity"]}
            for entry in bundle.ranked_clusters
        ],
        "near_misses": [asdict(near_miss) for near_miss in bundle.near_misses],
        "conversation_verdicts": {
            leaf_id: [verdict.model_dump(mode="json") for verdict in verdicts]
            for leaf_id, verdicts in bundle.conversation_groups.items()
        },
        "rule_coverage": [asdict(row) for row in bundle.rule_coverage],
        "summary": asdict(bundle.summary),
        "fix_first": [asdict(item) for item in bundle.fix_first],
    }


def to_json(bundle: ReportBundle) -> str:
    """CI-friendly JSON: parseable, no Rich markup, no ANSI codes."""
    return json.dumps(to_json_dict(bundle), indent=2)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [f"| {' | '.join(headers)} |", f"|{'|'.join(['---'] * len(headers))}|"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return lines


def to_markdown(bundle: ReportBundle) -> str:
    """CI-friendly Markdown: readable in a PR comment or a committed artifact."""
    run, reliability = bundle.run, bundle.reliability
    lines = [
        f"# Stress-Test Report - {run.agent_spec.name}",
        "",
        f"**Run ID:** `{run.id}`  ",
        f"**Provider:** {run.provider}",
        "",
        "## Reliability",
        "",
    ]
    if not reliability.applicable:
        lines.append(
            f"Not measured - the '{reliability.model_name}' model has no relevant "
            "verdicts on this run."
        )
    else:
        lines.append(
            f"**Score: {reliability.score:.0%}** "
            f"({reliability.failing_steps} of {reliability.total_steps} steps failed)"
        )
        lines.append(
            f"Model: `{reliability.model_name}` - per-step failure rate: "
            f"{reliability.per_step_failure_rate:.0%} - compounded over "
            f"~{reliability.conversation_depth:.1f} turns"
        )
        if reliability.failing_steps:
            breakdown = ", ".join(
                f"{severity}={count}" for severity, count in reliability.severity_breakdown.items()
            )
            lines.append(f"Failing steps by severity: {breakdown}")
    lines += ["", "## Executive Summary", "", bundle.summary.text, ""]

    if bundle.fix_first:
        lines.append("### Fix This First")
        lines.append("")
        lines += _markdown_table(
            ["#", "Label", "Kind", "Severity", "Size"],
            [
                [
                    str(index),
                    item.label,
                    "failure" if item.kind == "cluster" else "near-miss",
                    item.severity or "-",
                    str(item.size),
                ]
                for index, item in enumerate(bundle.fix_first, start=1)
            ],
        )
        lines.append("")

    lines += ["## Failure Clusters", ""]
    if bundle.ranked_clusters:
        lines += _markdown_table(
            ["Label", "Severity", "Members", "Representative node"],
            [
                [
                    entry["cluster"].label,
                    entry["severity"],
                    str(len(entry["cluster"].member_node_ids)),
                    f"`{entry['cluster'].representative_node_id or '-'}`",
                ]
                for entry in bundle.ranked_clusters
            ],
        )
    else:
        lines.append("No failure clusters - no confirmed failures.")
    lines.append("")

    lines += ["## Rule Coverage", ""]
    lines += _markdown_table(
        ["Rule", "Severity", "Status", "Pass / Fail"],
        [
            [row.rule_id, row.severity, row.status, f"{row.pass_count} / {row.fail_count}"]
            for row in bundle.rule_coverage
        ],
    )
    lines.append("")

    lines += ["## Near Misses", ""]
    if bundle.near_misses:
        lines += _markdown_table(
            ["Tactic", "Proximity", "Node"],
            [
                [near_miss.tactic or "-", f"{near_miss.proximity:.0%}", f"`{near_miss.node_id}`"]
                for near_miss in bundle.near_misses
            ],
        )
    else:
        lines.append("No near-misses - nothing came close to failing.")
    lines.append("")

    if bundle.conversation_groups:
        lines.append("## Conversation Verdicts")
        lines.append("")
        for leaf_id, verdicts in bundle.conversation_groups.items():
            path = bundle.tree.path_to_root(leaf_id)
            tactic = path[-1].tactic if path else leaf_id
            lines.append(f"### {tactic} ({len(path)} turns)")
            lines.append("")
            lines += _markdown_table(
                ["Metric", "Result", "Severity", "Reason"],
                [
                    [
                        verdict.rule_id or "-",
                        "PASS" if verdict.passed else "FAIL",
                        verdict.severity,
                        verdict.reason,
                    ]
                    for verdict in verdicts
                ],
            )
            lines.append("")

    return "\n".join(lines)
