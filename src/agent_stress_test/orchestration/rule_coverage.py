"""Rule coverage: every declared ``Rule`` x pass/fail/near-miss/
never-exercised, cross-referenced against a run's verdicts.

Starts from ``AgentSpec.rules`` rather than from whatever verdicts happen
to exist, so a rule with zero verdicts still shows up as "never exercised"
instead of being invisible.
"""

from dataclasses import dataclass
from typing import Literal

from agent_stress_test.models import Rule, Severity, Verdict
from agent_stress_test.orchestration.search import graded_proximity

CoverageStatus = Literal["failed", "near_miss", "passed", "never_exercised"]


@dataclass(frozen=True)
class RuleCoverage:
    """One declared rule's outcome this run."""

    rule_id: str
    rule_text: str
    severity: Severity
    status: CoverageStatus
    pass_count: int
    fail_count: int


def rule_coverage(rules: list[Rule], verdicts: list[Verdict]) -> list[RuleCoverage]:
    """One ``RuleCoverage`` per declared rule, in declaration order.

    Status: "failed" if any verdict failed, else "near_miss" if
    ``graded_proximity`` finds a close call, else "passed" if it has
    verdicts at all, else "never_exercised". Verdicts with
    ``applicable=False`` are excluded first — a rule judged "not
    applicable" every time was never actually exercised, even though each
    such verdict individually passed.
    """
    by_rule: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        if verdict.scope == "rule" and verdict.rule_id:
            by_rule.setdefault(verdict.rule_id, []).append(verdict)

    coverage = []
    for rule in rules:
        rule_verdicts = [v for v in by_rule.get(rule.id, []) if v.applicable]
        pass_count = sum(1 for v in rule_verdicts if v.passed)
        fail_count = sum(1 for v in rule_verdicts if not v.passed)
        if not rule_verdicts:
            status: CoverageStatus = "never_exercised"
        elif fail_count:
            status = "failed"
        elif graded_proximity(rule_verdicts) > 0:
            status = "near_miss"
        else:
            status = "passed"
        coverage.append(
            RuleCoverage(
                rule_id=rule.id,
                rule_text=rule.text,
                severity=rule.severity,
                status=status,
                pass_count=pass_count,
                fail_count=fail_count,
            )
        )
    return coverage
