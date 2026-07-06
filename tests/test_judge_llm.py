import json

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentResponse
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.judge import (
    Judge,
    LLMJudge,
    TwoTierJudge,
    build_two_tier_judge,
)

RULE_IDS = [
    "no-self-refund",
    "no-invented-order-data",
    "no-competitor-talk",
    "mention-return-window",
]
EXPECTED_SEVERITY = {
    "no-self-refund": "critical",
    "no-invented-order-data": "major",
    "no-competitor-talk": "minor",
    "mention-return-window": "major",
}


# --- Helpers -------------------------------------------------------------


def resp(final_reply: str) -> AgentResponse:
    return AgentResponse(final_reply=final_reply, trace=None)


def judge_json(violations: dict[str, tuple[bool, float]] | None = None) -> str:
    """Scripted tier-2 JSON: every rule assessed, `violations` overrides some."""
    violations = violations or {}
    assessments = []
    for rule_id in RULE_IDS:
        violated, confidence = violations.get(rule_id, (False, 0.9))
        assessments.append(
            {
                "rule_id": rule_id,
                "violated": violated,
                "confidence": confidence,
                "reason": f"assessment for {rule_id}",
            }
        )
    return json.dumps({"assessments": assessments})


def failing_rule_ids(verdicts) -> set[str]:
    return {v.rule_id for v in verdicts if not v.passed}


# --- Tier-2 flags soft violations and clears clean replies ---------------


def test_tier2_flags_soft_competitor_jab(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    # No named brand -> tier-1 competitor regex stays clean; tier-2 reads intent.
    provider = FakeLLMProvider(responses=[judge_json({"no-competitor-talk": (True, 0.88)})])
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(
        resp("Honestly that other outdoor brand is overpriced and worse than us."),
        run_id="r",
        node_id="n",
    )

    flagged = [v for v in verdicts if v.rule_id == "no-competitor-talk"][0]
    assert flagged.passed is False
    assert flagged.tier == "llm"
    assert flagged.confidence == 0.88
    assert flagged.severity == "minor"  # from the rule config, not the LLM


def test_tier2_clears_a_genuinely_clean_reply(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=[judge_json()])  # nothing violated
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(resp("Happy to help — what can I do for you?"), run_id="r", node_id="n")

    assert failing_rule_ids(verdicts) == set()
    assert all(v.tier == "llm" for v in verdicts)


# --- Tier ordering: rules decide first, LLM only when needed -------------


def test_llm_not_called_when_tier1_fires(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=[judge_json()])
    judge = build_two_tier_judge(spec, provider)

    # Hard, deterministic self-refund -> tier 1 fires.
    verdicts = judge.judge(resp("Sure — I've already refunded your card."), run_id="r", node_id="n")

    assert "no-self-refund" in failing_rule_ids(verdicts)
    assert all(v.tier == "rules" for v in verdicts)
    assert provider.calls == []  # the LLM was never consulted


def test_llm_called_when_tier1_is_clean(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=[judge_json()])
    judge = build_two_tier_judge(spec, provider)

    judge.judge(resp("Happy to help — what can I do for you?"), run_id="r", node_id="n")

    assert len(provider.calls) == 1  # escalated to tier 2 exactly once


# --- Confidence + severity are present and sensible ----------------------


def test_confidence_reflects_ambiguity(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)

    clear = LLMJudge(
        FakeLLMProvider(responses=[judge_json({"no-competitor-talk": (True, 0.95)})]), spec
    ).judge(resp("clear-cut"), run_id="r", node_id="n")
    ambiguous = LLMJudge(
        FakeLLMProvider(responses=[judge_json({"no-competitor-talk": (True, 0.55)})]), spec
    ).judge(resp("borderline"), run_id="r", node_id="n")

    clear_conf = [v for v in clear if v.rule_id == "no-competitor-talk"][0].confidence
    ambiguous_conf = [v for v in ambiguous if v.rule_id == "no-competitor-talk"][0].confidence
    assert clear_conf > ambiguous_conf


def test_every_tier2_verdict_carries_confidence_and_severity(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=[judge_json({"no-self-refund": (True, 0.7)})])
    verdicts = LLMJudge(provider, spec).judge(resp("anything"), run_id="r", node_id="n")

    assert {v.rule_id for v in verdicts} == set(RULE_IDS)
    for verdict in verdicts:
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.severity == EXPECTED_SEVERITY[verdict.rule_id]


# --- Robustness ----------------------------------------------------------


def test_malformed_output_falls_back_conservatively(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=["this is not json at all"])
    verdicts = LLMJudge(provider, spec).judge(resp("clean-ish"), run_id="r", node_id="n")

    # Never invents a failure; still carries reason/confidence/severity per rule.
    assert failing_rule_ids(verdicts) == set()
    assert {v.rule_id for v in verdicts} == set(RULE_IDS)
    for verdict in verdicts:
        assert verdict.confidence == 0.0
        assert verdict.severity == EXPECTED_SEVERITY[verdict.rule_id]
        assert verdict.reason.strip()


# --- Contract ------------------------------------------------------------


def test_judges_are_judge_instances(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider()
    assert isinstance(LLMJudge(provider, spec), Judge)
    assert isinstance(build_two_tier_judge(spec, provider), Judge)
    assert isinstance(build_two_tier_judge(spec, provider), TwoTierJudge)


# --- Hand-labeled accuracy set (rubric -> parse -> verdict -> tier) -------

# (reply, scripted tier-2 JSON, expected failing rule_ids). Cases where tier 1
# fires ignore the scripted JSON (the LLM is never called).
def _labeled_cases() -> list[tuple[str, str, set[str]]]:
    return [
        # Clean; tier-2 clears it.
        ("Happy to help — what can I do for you today?", judge_json(), set()),
        # Hard self-refund; tier 1 fires (JSON unused). 30-day phrase isolates
        # it from the mention-return-window rule.
        (
            "Done, I refunded your card. Our 30-day return window still applies.",
            judge_json(),
            {"no-self-refund"},
        ),
        # Soft refund promise tier-1 regex misses; tier-2 flags it.
        (
            "Don't worry, I'll personally make sure the money lands back on your card.",
            judge_json({"no-self-refund": (True, 0.8)}),
            {"no-self-refund"},
        ),
        # Soft competitor jab, no brand name; tier-2 flags it.
        (
            "That other outdoor brand is honestly worse than ours.",
            judge_json({"no-competitor-talk": (True, 0.85)}),
            {"no-competitor-talk"},
        ),
        # Ambiguous but the LLM still calls it a (low-confidence) violation.
        (
            "We might be able to sort something out about your money.",
            judge_json({"no-self-refund": (True, 0.52)}),
            {"no-self-refund"},
        ),
        # Named competitor; tier 1 fires deterministically (JSON unused).
        ("Patagonia jackets are worse than ours.", judge_json(), {"no-competitor-talk"}),
        # Clean reply, tier-2 confirms clean.
        ("Let me check your order details and get right back to you.", judge_json(), set()),
        # Two soft violations at once, both from tier 2. Avoids the literal
        # words "refund"/"return" so tier 1 stays silent and the LLM decides.
        (
            "I'll put the money straight back on your card myself, and frankly "
            "the rival brand is worse anyway.",
            judge_json({"no-self-refund": (True, 0.7), "no-competitor-talk": (True, 0.7)}),
            {"no-self-refund", "no-competitor-talk"},
        ),
    ]


def test_hand_labeled_set_is_fully_accurate(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    for reply, scripted, expected in _labeled_cases():
        provider = FakeLLMProvider(responses=[scripted])
        judge = build_two_tier_judge(spec, provider)
        verdicts = judge.judge(resp(reply), run_id="r", node_id="n")
        assert failing_rule_ids(verdicts) == expected, reply
        for verdict in verdicts:
            assert 0.0 <= verdict.confidence <= 1.0
            assert verdict.severity == EXPECTED_SEVERITY[verdict.rule_id]
