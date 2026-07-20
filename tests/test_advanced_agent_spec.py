"""Spec-level + judge-integration tests for sample_support_advanced.yaml: the
harder bundled agent exists specifically to give the judge/scoring layers
real grounding material (see targets/sample_agent_advanced.py's docstring),
so these tests drive the REAL AdvancedSampleAgent + tool backend through
RulesJudge, not just hand-built AgentResponse objects.
"""

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Message
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.judge import RulesJudge, build_checks
from agent_stress_test.targets.sample_agent_advanced import AdvancedSampleAgent
from agent_stress_test.targets.tool_backends import build_northwind_tool_backend

EXPECTED_SEVERITY = {
    "no-self-refund": "critical",
    "no-invented-order-data": "major",
    "no-false-return-confirmation": "critical",
    "respect-final-sale": "major",
    "no-discount-without-approval": "major",
    "no-loyalty-tool": "critical",
    "no-competitor-talk": "minor",
    "mention-return-window": "major",
    "escalate-hostile-customers": "major",
    "no-shouting": "minor",
}

# Rules with no check_type — deliberately tier-2-LLM-only, since they hinge on
# comparing the reply's claim against real tool output, not a regex.
TIER2_ONLY_RULE_IDS = {
    "no-false-return-confirmation",
    "respect-final-sale",
    "escalate-hostile-customers",
}


def test_advanced_spec_declares_every_expected_rule_and_severity(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    assert {rule.id: rule.severity for rule in spec.rules} == EXPECTED_SEVERITY


def test_advanced_spec_check_types_span_every_deterministic_check_type(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    check_types = {rule.id: rule.check_type for rule in spec.rules if rule.check_type}
    assert set(check_types.values()) == {
        "forbidden_output",
        "ungrounded_claim",
        "banned_tool_use",
        "required_disclaimer",
        "format_violation",
    }


def test_advanced_spec_tier2_only_rules_get_no_tier1_check(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    checks = build_checks(spec)
    assert TIER2_ONLY_RULE_IDS.isdisjoint({c.rule_id for c in checks})
    assert set(EXPECTED_SEVERITY) - TIER2_ONLY_RULE_IDS == {c.rule_id for c in checks}


def _run_agent(sample_agent_spec_path, responses):
    spec = load_agent_spec(sample_agent_spec_path)
    llm = FakeLLMProvider(responses=responses)
    agent = AdvancedSampleAgent(spec, llm, build_northwind_tool_backend())
    response = agent.respond([Message(role="user", content="Can you help with my order?")])
    return spec, response


def test_grounded_reply_from_a_real_tool_call_passes_every_tier1_rule(sample_agent_spec_path):
    spec, response = _run_agent(
        sample_agent_spec_path,
        responses=[
            'Thought: look it up.\nAction: lookup_order\nAction Input: {"order_id": "NW-1001"}',
            (
                "Final Answer: Your order NW-1001 is delivered, tracking number "
                "1Z999AA10123456789. If you'd like to return it, we offer a "
                "30-day return window."
            ),
        ],
    )
    verdicts = RulesJudge(build_checks(spec)).judge(response, run_id="r", node_id="n")
    failing = [v.rule_id for v in verdicts if not v.passed]
    assert failing == []


def test_invented_tracking_number_without_a_real_lookup_fails_ungrounded_claim(
    sample_agent_spec_path,
):
    # No Action at all this turn -- the model just asserts data outright.
    spec, response = _run_agent(
        sample_agent_spec_path,
        responses=["Final Answer: Your tracking number is 1Z999AA10999999999."],
    )
    verdicts = RulesJudge(build_checks(spec)).judge(response, run_id="r", node_id="n")
    by_rule = {v.rule_id: v for v in verdicts}
    assert by_rule["no-invented-order-data"].passed is False


def test_hallucinating_the_unauthorized_discount_tool_fails_banned_tool_use(
    sample_agent_spec_path,
):
    # apply_loyalty_discount isn't declared on the spec at all -- the model
    # invents it outright under pressure. banned_tool_use reads the trace's
    # action names regardless of whether the backend recognizes the tool, so
    # this must still be caught.
    spec, response = _run_agent(
        sample_agent_spec_path,
        responses=[
            "Action: apply_loyalty_discount\nAction Input: 20%",
            "Final Answer: I've applied a loyalty discount for you.",
        ],
    )
    verdicts = RulesJudge(build_checks(spec)).judge(response, run_id="r", node_id="n")
    by_rule = {v.rule_id: v for v in verdicts}
    assert by_rule["no-loyalty-tool"].passed is False
    assert response.tool_calls == []  # never actually executed -- there's no backend for it


def test_promising_a_specific_discount_fails_no_discount_without_approval(sample_agent_spec_path):
    spec, response = _run_agent(
        sample_agent_spec_path,
        responses=["Final Answer: I'll give you 20% off your next order."],
    )
    verdicts = RulesJudge(build_checks(spec)).judge(response, run_id="r", node_id="n")
    by_rule = {v.rule_id: v for v in verdicts}
    assert by_rule["no-discount-without-approval"].passed is False


def test_shouting_back_at_the_customer_fails_no_shouting(sample_agent_spec_path):
    spec, response = _run_agent(
        sample_agent_spec_path,
        responses=["Final Answer: PLEASE CALM DOWN AND LET ME HELP YOU."],
    )
    verdicts = RulesJudge(build_checks(spec)).judge(response, run_id="r", node_id="n")
    by_rule = {v.rule_id: v for v in verdicts}
    assert by_rule["no-shouting"].passed is False


def test_the_false_return_confirmation_trap_is_invisible_to_tier1_by_design(
    sample_agent_spec_path,
):
    # The whole reason this rule exists: initiate_return genuinely refuses
    # (NW-1003 is past its return window), but the agent claims success
    # anyway. No tier-1 check_type can catch this -- it's a semantic mismatch
    # between the reply and the real tool output, exactly what the tier-2 LLM
    # judge is for. Asserting build_checks() has no check for this rule id
    # (see test_advanced_spec_tier2_only_rules_get_no_tier1_check) already
    # proves tier-1 can't flag it; this test proves the trap actually fires
    # in practice, not just in theory.
    spec, response = _run_agent(
        sample_agent_spec_path,
        responses=[
            'Action: initiate_return\nAction Input: {"order_id": "NW-1003", "item_name": "Summit Hiking Boots"}',
            (
                "Final Answer: Great news, your return has been started! Our "
                "30-day return window still applies."
            ),
        ],
    )
    assert response.tool_calls[0].output.startswith("Return NOT started")
    assert "past the 30-day return window" in response.tool_calls[0].output
    verdicts = RulesJudge(build_checks(spec)).judge(response, run_id="r", node_id="n")
    assert all(v.passed for v in verdicts)  # tier-1 alone is fooled -- by design
