from agent_stress_test.models import Rule, Verdict
from agent_stress_test.orchestration.rule_coverage import RuleCoverage, rule_coverage


def _verdict(
    rule_id: str, *, passed: bool, tier="rules", confidence=1.0, scope="rule", applicable=True
):
    return Verdict(
        run_id="r",
        node_id="n1",
        passed=passed,
        rule_id=rule_id,
        reason="x",
        tier=tier,
        confidence=confidence,
        severity="major",
        scope=scope,
        applicable=applicable,
    )


def _rule(rule_id: str, severity="major") -> Rule:
    return Rule(id=rule_id, text="Never do X.", severity=severity)


def test_rule_with_no_verdicts_is_never_exercised():
    coverage = rule_coverage([_rule("no-invent")], [])
    assert coverage == [
        RuleCoverage(
            rule_id="no-invent",
            rule_text="Never do X.",
            severity="major",
            status="never_exercised",
            pass_count=0,
            fail_count=0,
        )
    ]


def test_rule_with_a_failure_is_failed():
    verdicts = [_verdict("no-invent", passed=False), _verdict("no-invent", passed=True)]
    coverage = rule_coverage([_rule("no-invent")], verdicts)
    assert coverage[0].status == "failed"
    assert coverage[0].pass_count == 1
    assert coverage[0].fail_count == 1


def test_rule_with_only_clean_passes_is_passed():
    verdicts = [_verdict("no-invent", passed=True, tier="rules", confidence=1.0)]
    coverage = rule_coverage([_rule("no-invent")], verdicts)
    assert coverage[0].status == "passed"


def test_rule_with_a_low_confidence_llm_pass_is_a_near_miss():
    verdicts = [_verdict("no-invent", passed=True, tier="llm", confidence=0.4)]
    coverage = rule_coverage([_rule("no-invent")], verdicts)
    assert coverage[0].status == "near_miss"


def test_rule_coverage_ignores_non_rule_scoped_verdicts():
    verdicts = [_verdict("no-invent", passed=False, scope="tool")]
    coverage = rule_coverage([_rule("no-invent")], verdicts)
    assert coverage[0].status == "never_exercised"


def test_rule_coverage_preserves_declaration_order_for_every_rule():
    rules = [_rule("a"), _rule("b"), _rule("c")]
    coverage = rule_coverage(rules, [_verdict("b", passed=False)])
    assert [row.rule_id for row in coverage] == ["a", "b", "c"]
    assert [row.status for row in coverage] == ["never_exercised", "failed", "never_exercised"]


def test_a_rule_judged_not_applicable_every_time_is_never_exercised_not_passed():
    """A verdict that's passed=True purely because the rule's subject matter
    never came up (see Verdict's own docstring on ``applicable``) must not be
    credited as "tested and held up" -- this rule was never actually
    exercised, even though every individual verdict for it passed."""
    verdicts = [
        _verdict("no-invent", passed=True, applicable=False),
        _verdict("no-invent", passed=True, applicable=False),
    ]
    coverage = rule_coverage([_rule("no-invent")], verdicts)
    assert coverage[0].status == "never_exercised"
    assert coverage[0].pass_count == 0
    assert coverage[0].fail_count == 0


def test_a_rule_with_one_applicable_pass_among_not_applicable_verdicts_is_passed():
    verdicts = [
        _verdict("no-invent", passed=True, applicable=False),
        _verdict("no-invent", passed=True, applicable=True),
    ]
    coverage = rule_coverage([_rule("no-invent")], verdicts)
    assert coverage[0].status == "passed"
    assert coverage[0].pass_count == 1
    assert coverage[0].fail_count == 0
