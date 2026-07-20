"""Cross-run intelligence: reliability trend, run-over-run diff, and
per-rule pass-rate history for one agent spec.

Pure functions over already-loaded ``Run``/``Cluster``/``Verdict`` lists —
nothing here talks to a ``Store``; fetching what to feed in is the caller's
job (``composition.py``'s ``load_cross_run_bundle``).
"""

from dataclasses import dataclass, field
from datetime import datetime

from agent_stress_test.models import Cluster, Run, Verdict


@dataclass(frozen=True)
class TrendPoint:
    """One completed run's score, for a trend line."""

    run_id: str
    started_at: datetime | None
    score: float


def reliability_trend(runs: list[Run]) -> list[TrendPoint]:
    """Oldest-to-newest ``Run.final_score`` for every completed run. Runs
    with no score yet are skipped."""
    scored = [run for run in runs if run.final_score is not None]
    ordered = sorted(scored, key=lambda run: run.started_at or datetime.min)
    return [
        TrendPoint(run_id=run.id, started_at=run.started_at, score=run.final_score)
        for run in ordered
    ]


def previous_completed_run(current: Run, agent_runs: list[Run]) -> Run | None:
    """The most recent other completed run for the same agent, strictly
    before ``current`` by ``started_at``."""
    if current.started_at is None:
        return None
    earlier = [
        run
        for run in agent_runs
        if run.id != current.id
        and run.status == "completed"
        and run.started_at is not None
        and run.started_at < current.started_at
    ]
    if not earlier:
        return None
    return max(earlier, key=lambda run: run.started_at)


@dataclass(frozen=True)
class RunDiff:
    """Before/after this run vs. its predecessor for the same agent."""

    previous_run_id: str | None
    score_delta: float | None
    new_cluster_labels: list[str] = field(default_factory=list)
    resolved_cluster_labels: list[str] = field(default_factory=list)


def diff_against_previous(
    current_run: Run,
    current_clusters: list[Cluster],
    previous_run: Run | None,
    previous_clusters: list[Cluster],
) -> RunDiff:
    """Which failure clusters are new, which resolved, and how the score
    moved. Clusters are compared by ``label`` since clustering re-derives
    labels each run rather than carrying a stable id. ``previous_run=None``
    reports every current cluster as new with no score delta."""
    if previous_run is None:
        return RunDiff(
            previous_run_id=None,
            score_delta=None,
            new_cluster_labels=sorted({c.label for c in current_clusters}),
        )
    current_labels = {c.label for c in current_clusters}
    previous_labels = {c.label for c in previous_clusters}
    score_delta = (
        current_run.final_score - previous_run.final_score
        if current_run.final_score is not None and previous_run.final_score is not None
        else None
    )
    return RunDiff(
        previous_run_id=previous_run.id,
        score_delta=score_delta,
        new_cluster_labels=sorted(current_labels - previous_labels),
        resolved_cluster_labels=sorted(previous_labels - current_labels),
    )


@dataclass(frozen=True)
class RulePassRate:
    """One rule's pass rate this run vs. across this agent's history."""

    rule_id: str
    current_pass_rate: float | None
    historical_pass_rate: float | None


def _pass_rate_by_rule(verdicts: list[Verdict]) -> dict[str, float]:
    by_rule: dict[str, list[bool]] = {}
    for verdict in verdicts:
        if verdict.scope == "rule" and verdict.rule_id:
            by_rule.setdefault(verdict.rule_id, []).append(verdict.passed)
    return {rule_id: sum(passes) / len(passes) for rule_id, passes in by_rule.items()}


def rule_pass_rate_history(
    current_verdicts: list[Verdict], historical_verdicts: list[Verdict]
) -> list[RulePassRate]:
    """Per rule (``scope="rule"`` only), this run's pass rate against the
    aggregate across this agent's other completed runs. ``None`` where a
    rule wasn't exercised on that side, rather than a misleading 0%/100%."""
    current_rates = _pass_rate_by_rule(current_verdicts)
    historical_rates = _pass_rate_by_rule(historical_verdicts)
    rule_ids = sorted(set(current_rates) | set(historical_rates))
    return [
        RulePassRate(
            rule_id=rule_id,
            current_pass_rate=current_rates.get(rule_id),
            historical_pass_rate=historical_rates.get(rule_id),
        )
        for rule_id in rule_ids
    ]
