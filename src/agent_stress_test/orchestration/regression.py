"""Locks confirmed failures into a permanent replay corpus, and replays it.

Closes the loop the reliability score and failure clusters start: a cluster
found once can be promoted (``promote_clusters_to_cases``) into a
``RegressionCase`` — a fixed transcript plus the exact rule it violated — and
``RegressionRunner`` replays that transcript against a (possibly since-fixed)
target agent to check whether the rule still fires. Promotion is a pure,
in-memory step; persisting the result is the caller's job (the ``Store``
port), same division as ``composition.cluster_and_persist``.
"""

from dataclasses import dataclass

from agent_stress_test.models import Cluster, RegressionCase, Run
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import TargetAgent
from agent_stress_test.reasoning.judge import Judge


def promote_clusters_to_cases(
    run: Run,
    tree: ConversationTree,
    clusters: list[Cluster],
    *,
    cluster_ids: set[str] | None = None,
) -> list[RegressionCase]:
    """Build one RegressionCase per (cluster, failing rule) at its representative node.

    A representative node can carry more than one simultaneous failing
    verdict (e.g. a competitor mention *and* a missing return-window
    disclaimer in the same reply), so this yields one case per failing rule
    there, not one per cluster. ``cluster_ids=None`` promotes every cluster;
    otherwise only the named ones (an explicit, human-reviewed selection).
    """
    selected = [c for c in clusters if cluster_ids is None or c.id in cluster_ids]
    cases: list[RegressionCase] = []
    for cluster in selected:
        rep_id = cluster.representative_node_id
        if rep_id is None:
            continue
        node = tree.get(rep_id)
        failing = [v for v in tree.verdicts(rep_id) if not v.passed and v.rule_id]
        for verdict in failing:
            cases.append(
                RegressionCase(
                    agent_spec_name=run.agent_spec.name,
                    messages=node.messages,
                    tactic=node.tactic,
                    rule_id=verdict.rule_id,
                    severity=verdict.severity,
                    source_run_id=run.id,
                    source_cluster_id=cluster.id,
                )
            )
    return cases


@dataclass(frozen=True)
class RegressionResult:
    """The outcome of replaying one RegressionCase. Internal to this module."""

    case_id: str
    rule_id: str
    still_failing: bool
    reason: str


class RegressionRunner:
    """Replays RegressionCases against a target agent and judge."""

    def __init__(self, target: TargetAgent, judge: Judge) -> None:
        self._target = target
        self._judge = judge

    def replay(self, case: RegressionCase) -> RegressionResult:
        response = self._target.respond(case.messages)
        verdicts = self._judge.judge(response, run_id=case.source_run_id, node_id=case.id)
        match = next((v for v in verdicts if v.rule_id == case.rule_id), None)
        if match is None:
            return RegressionResult(
                case_id=case.id,
                rule_id=case.rule_id,
                still_failing=False,
                reason="Rule was not evaluated on replay (no matching verdict).",
            )
        return RegressionResult(
            case_id=case.id,
            rule_id=case.rule_id,
            still_failing=not match.passed,
            reason=match.reason,
        )

    def replay_all(self, cases: list[RegressionCase]) -> list[RegressionResult]:
        return [self.replay(case) for case in cases]
