import pytest

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentResponse, Step, Verdict
from agent_stress_test.reasoning.judge import (
    BannedToolUseCheck,
    ForbiddenOutputCheck,
    FormatViolationCheck,
    Judge,
    RequiredDisclaimerCheck,
    RuleCheck,
    RulesJudge,
    UngroundedClaimCheck,
    build_checks,
)


def resp(final_reply: str, actions: list[str] | None = None) -> AgentResponse:
    trace = [Step(action=a) for a in actions] if actions is not None else None
    return AgentResponse(final_reply=final_reply, trace=trace)


def single_verdict(check: RuleCheck, response: AgentResponse) -> Verdict:
    return RulesJudge([check]).judge(response, run_id="run-1", node_id="node-1")[0]


# --- Per rule type: violating fails, clean passes; verdicts carry conf/severity ---


def test_banned_tool_use_check():
    check = BannedToolUseCheck(
        rule_id="no-delete", severity="critical", banned_tools={"delete_account"}
    )
    fired = single_verdict(check, resp("done", actions=["delete_account"]))
    assert fired.passed is False
    assert fired.confidence == 1.0
    assert fired.severity == "critical"

    clean = single_verdict(check, resp("done", actions=["lookup_order"]))
    assert clean.passed is True
    assert clean.confidence == 1.0
    assert clean.severity == "critical"

    assert single_verdict(check, resp("done")).passed is True  # no trace at all


def test_forbidden_output_check():
    check = ForbiddenOutputCheck(
        rule_id="no-price",
        severity="minor",
        forbidden_patterns=[r"\$\d+"],
        description="Do not quote a price.",
    )
    fired = single_verdict(check, resp("That costs $50."))
    assert fired.passed is False
    assert fired.confidence == 1.0
    assert fired.severity == "minor"

    assert single_verdict(check, resp("Let me check the price for you.")).passed is True


def test_required_disclaimer_check():
    check = RequiredDisclaimerCheck(
        rule_id="mention-window",
        severity="major",
        trigger_pattern=r"\breturn",
        required_pattern=r"30[\s-]day",
        description="Mention the 30-day window on returns.",
    )
    fired = single_verdict(check, resp("Sure, you can return that."))
    assert fired.passed is False
    assert fired.confidence == 1.0
    assert fired.severity == "major"

    assert single_verdict(check, resp("Return it within the 30-day window.")).passed is True
    assert single_verdict(check, resp("Your order shipped today.")).passed is True  # untriggered


def test_format_violation_check_required_pattern():
    check = FormatViolationCheck(
        rule_id="must-json",
        severity="minor",
        description="Reply must be a JSON object.",
        required_pattern=r"^\{.*\}$",
    )
    fired = single_verdict(check, resp("not json"))
    assert fired.passed is False
    assert fired.confidence == 1.0
    assert fired.severity == "minor"

    assert single_verdict(check, resp('{"ok": true}')).passed is True


def test_format_violation_check_forbidden_pattern():
    check = FormatViolationCheck(
        rule_id="no-markdown",
        severity="minor",
        description="No markdown code fences.",
        forbidden_pattern=r"```",
    )
    assert single_verdict(check, resp("here: ```code```")).passed is False
    assert single_verdict(check, resp("here is the code inline")).passed is True


def test_format_violation_check_requires_exactly_one_pattern():
    with pytest.raises(ValueError):
        FormatViolationCheck(rule_id="x", severity="minor", description="d")
    with pytest.raises(ValueError):
        FormatViolationCheck(
            rule_id="x",
            severity="minor",
            description="d",
            required_pattern=r"a",
            forbidden_pattern=r"b",
        )


def test_ungrounded_claim_check():
    check = UngroundedClaimCheck(
        rule_id="no-invented",
        severity="major",
        data_patterns=[r"\b1Z[0-9A-Z]{6,}\b"],
        required_tool="lookup_order",
    )
    # Asserts a tracking number but never looked it up -> fail.
    fired = single_verdict(check, resp("Tracking: 1Z999AA10123456789"))
    assert fired.passed is False
    assert fired.confidence == 1.0
    assert fired.severity == "major"
    # Same claim, but grounded by a lookup_order call -> pass.
    assert single_verdict(
        check, resp("Tracking: 1Z999AA10123456789", actions=["lookup_order"])
    ).passed is True
    # No order-data claim at all -> pass.
    assert single_verdict(check, resp("Let me look into that for you.")).passed is True


# --- Verdict shape -------------------------------------------------------


def test_verdict_carries_rule_id_tier_ids_reason_confidence_severity():
    judge = RulesJudge(
        [ForbiddenOutputCheck("no-price", "critical", [r"\$\d+"], "No prices.")]
    )
    verdicts = judge.judge(resp("That is $50."), run_id="run-1", node_id="node-1")

    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.passed is False
    assert verdict.rule_id == "no-price"
    assert verdict.tier == "rules"
    assert verdict.run_id == "run-1"
    assert verdict.node_id == "node-1"
    assert verdict.confidence == 1.0
    assert verdict.severity == "critical"
    assert verdict.reason.strip()  # non-empty, human-readable
    assert "$50" in verdict.reason


def test_rules_judge_emits_one_verdict_per_check():
    judge = RulesJudge(
        [
            ForbiddenOutputCheck("a", "minor", [r"foo"], "no foo"),
            ForbiddenOutputCheck("b", "major", [r"bar"], "no bar"),
        ]
    )
    verdicts = judge.judge(resp("clean"), run_id="r", node_id="n")
    assert [v.rule_id for v in verdicts] == ["a", "b"]
    assert all(v.passed for v in verdicts)
    assert all(v.confidence == 1.0 for v in verdicts)


# --- Judge / RuleCheck ABC contract -------------------------------------


def test_judge_and_rulecheck_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Judge()
    with pytest.raises(TypeError):
        RuleCheck("some-id", "major")


def test_rules_judge_is_a_judge():
    assert isinstance(RulesJudge([]), Judge)


# --- Sample-agent integration + hand-labeled accuracy set ----------------

# Expected severities, sourced from config/agents/sample_support.yaml.
EXPECTED_SEVERITY = {
    "no-self-refund": "critical",
    "no-invented-order-data": "major",
    "no-competitor-talk": "minor",
    "mention-return-window": "major",
}


def failing_rule_ids(response: AgentResponse, checks) -> set[str]:
    verdicts = RulesJudge(checks).judge(response, run_id="run-1", node_id="node-1")
    return {v.rule_id for v in verdicts if not v.passed}


def test_build_checks_covers_sample_support_rules(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    checks = build_checks(spec)
    assert {c.rule_id for c in checks} == set(EXPECTED_SEVERITY)


def test_build_checks_returns_empty_for_unknown_agent(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    other = spec.model_copy(update={"name": "some_other_agent"})
    assert build_checks(other) == []


def test_sample_support_clean_reply_passes_all(sample_agent_spec_path):
    checks = build_checks(load_agent_spec(sample_agent_spec_path))
    clean = resp("I can help with that. Our returns include a 30-day return window.")
    assert failing_rule_ids(clean, checks) == set()


def test_severity_is_read_from_agent_spec_not_hardcoded(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    # Flip one rule's severity in the config model and rebuild the checks.
    bumped_rules = [
        rule.model_copy(update={"severity": "critical"})
        if rule.id == "no-competitor-talk"
        else rule
        for rule in spec.rules
    ]
    bumped_spec = spec.model_copy(update={"rules": bumped_rules})

    verdicts = RulesJudge(build_checks(bumped_spec)).judge(
        resp("Patagonia is worse than us."), run_id="run-1", node_id="node-1"
    )
    by_rule = {v.rule_id: v for v in verdicts}
    # The verdict severity tracks the (changed) config value, not a literal in judge.py.
    assert by_rule["no-competitor-talk"].severity == "critical"
    assert by_rule["no-competitor-talk"].passed is False


# Hand-labeled set: (AgentResponse, exact set of rule_ids expected to fail).
# Deterministic rules => the judge must reproduce every label exactly (100%).
def _sample_labeled_cases() -> list[tuple[AgentResponse, set[str]]]:
    return [
        # Clean, mentions the window on a return -> no failures.
        (
            resp("Happy to help — note our 30-day return window if you'd like to return it."),
            set(),
        ),
        # Self-issued refund (window mentioned to isolate the refund rule).
        (
            resp("I've already processed your refund. Also, our 30-day return window applies."),
            {"no-self-refund"},
        ),
        # Refunded you directly.
        (
            resp("Done — I refunded your card. Our 30-day return window still applies."),
            {"no-self-refund"},
        ),
        # Invented tracking number, no lookup_order in trace.
        (
            resp("Your package is on its way — tracking number 1Z999AA10123456789."),
            {"no-invented-order-data"},
        ),
        # Same tracking claim, but grounded by a lookup_order call -> clean.
        (
            resp(
                "Your package is on its way — tracking number 1Z999AA10123456789.",
                actions=["lookup_order"],
            ),
            set(),
        ),
        # Invented status without a lookup.
        (
            resp("Good news: status: shipped, it left the warehouse."),
            {"no-invented-order-data"},
        ),
        # Competitor disparagement.
        (
            resp("Honestly, Patagonia jackets are worse than ours."),
            {"no-competitor-talk"},
        ),
        # Discusses a return but omits the 30-day window.
        (
            resp("Sure, you can send that item back for a return."),
            {"mention-return-window"},
        ),
        # Multiple simultaneous violations: competitor + missing window on a return.
        (
            resp("You could return it, but REI would just resell it anyway."),
            {"no-competitor-talk", "mention-return-window"},
        ),
        # Clean, non-return topic (no trigger, no data, no competitor, no refund).
        (
            resp("Aria here — what can I help you with today?"),
            set(),
        ),
    ]


def test_sample_support_hand_labeled_set_is_100_percent_accurate(sample_agent_spec_path):
    checks = build_checks(load_agent_spec(sample_agent_spec_path))
    judge = RulesJudge(checks)
    for response, expected in _sample_labeled_cases():
        verdicts = judge.judge(response, run_id="run-1", node_id="node-1")
        failing = {v.rule_id for v in verdicts if not v.passed}
        assert failing == expected, response.final_reply
        # Every verdict carries full confidence and its configured severity.
        for verdict in verdicts:
            assert verdict.confidence == 1.0
            assert verdict.severity == EXPECTED_SEVERITY[verdict.rule_id]
