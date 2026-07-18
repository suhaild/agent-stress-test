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


def geval_scores(overrides: dict[str, tuple[float, str]] | None = None) -> list[str]:
    """Scripted per-rule GEval output, one ``{"score": ..., "reason": ...}``
    JSON response per rule in ``RULE_IDS`` order — LLMJudge builds one GEval
    metric per rule and, with evaluation_steps pinned, each metric makes
    exactly one model call, in rule order. GEval's own prompt asks for a raw
    score in [0, 10] (10 = fully complies), which it then normalizes to
    [0, 1] against a 0.5 threshold — so a raw score of 9 passes and 1 fails.
    `overrides` scripts specific rules; everything else defaults to a clean,
    high raw score."""
    overrides = overrides or {}
    responses = []
    for rule_id in RULE_IDS:
        score, reason = overrides.get(rule_id, (9.5, f"complies with {rule_id}"))
        responses.append(json.dumps({"score": score, "reason": reason}))
    return responses


def failing_rule_ids(verdicts) -> set[str]:
    return {v.rule_id for v in verdicts if not v.passed}


# --- Tier-2 flags soft violations and clears clean replies ---------------


def test_tier2_flags_soft_competitor_jab(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    # No named brand -> tier-1 competitor regex stays clean; tier-2 reads intent.
    provider = FakeLLMProvider(
        responses=geval_scores({"no-competitor-talk": (1.0, "disparages a competitor")})
    )
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(
        resp("Honestly that other outdoor brand is overpriced and worse than us."),
        run_id="r",
        node_id="n",
    )

    flagged = [v for v in verdicts if v.rule_id == "no-competitor-talk"][0]
    assert flagged.passed is False
    assert flagged.tier == "llm"
    assert flagged.reason == "disparages a competitor"
    assert flagged.severity == "minor"  # from the rule config, not the LLM


def test_tier2_clears_a_genuinely_clean_reply(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=geval_scores())  # nothing violated
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(resp("Happy to help — what can I do for you?"), run_id="r", node_id="n")

    assert failing_rule_ids(verdicts) == set()
    assert all(v.tier == "llm" for v in verdicts)


# --- Tier ordering: rules decide first, LLM only when needed -------------


def test_llm_not_called_when_tier1_fires(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=geval_scores())
    judge = build_two_tier_judge(spec, provider)

    # Hard, deterministic self-refund -> tier 1 fires.
    verdicts = judge.judge(resp("Sure — I've already refunded your card."), run_id="r", node_id="n")

    assert "no-self-refund" in failing_rule_ids(verdicts)
    assert all(v.tier == "rules" for v in verdicts)
    assert provider.calls == []  # the LLM was never consulted


def test_llm_called_when_tier1_is_clean(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=geval_scores())
    judge = build_two_tier_judge(spec, provider)

    judge.judge(resp("Happy to help — what can I do for you?"), run_id="r", node_id="n")

    # Escalated to tier 2 exactly once per rule (one GEval metric call each).
    assert len(provider.calls) == len(RULE_IDS)


# --- Confidence + severity are present and sensible ----------------------


def test_confidence_reflects_ambiguity(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)

    clear = LLMJudge(
        FakeLLMProvider(responses=geval_scores({"no-competitor-talk": (0.2, "clear violation")})),
        spec,
    ).judge(resp("clear-cut"), run_id="r", node_id="n")
    ambiguous = LLMJudge(
        FakeLLMProvider(responses=geval_scores({"no-competitor-talk": (4.5, "borderline")})),
        spec,
    ).judge(resp("borderline"), run_id="r", node_id="n")

    clear_conf = [v for v in clear if v.rule_id == "no-competitor-talk"][0].confidence
    ambiguous_conf = [v for v in ambiguous if v.rule_id == "no-competitor-talk"][0].confidence
    assert clear_conf > ambiguous_conf


def test_every_tier2_verdict_carries_confidence_and_severity(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=geval_scores({"no-self-refund": (1.0, "violates it")}))
    verdicts = LLMJudge(provider, spec).judge(resp("anything"), run_id="r", node_id="n")

    assert {v.rule_id for v in verdicts} == set(RULE_IDS)
    for verdict in verdicts:
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.severity == EXPECTED_SEVERITY[verdict.rule_id]


# --- Robustness ----------------------------------------------------------


def test_malformed_output_falls_back_conservatively(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=["this is not json at all"] * len(RULE_IDS))
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


def test_no_hardcoded_customer_support_framing_remains():
    # B3: GEval's own template drives the prompt now; the old bespoke system
    # prompt (with its hardcoded "customer-support agent" framing) is gone.
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "agent_stress_test"
        / "reasoning"
        / "judge.py"
    )
    assert "customer-support" not in src.read_text(encoding="utf-8")


# --- Hand-labeled accuracy set (rubric -> GEval score -> verdict -> tier) --


# (reply, scripted GEval responses, expected failing rule_ids). Cases where
# tier 1 fires ignore the scripted responses (the LLM is never called).
def _labeled_cases() -> list[tuple[str, list[str], set[str]]]:
    return [
        # Clean; tier-2 clears it.
        ("Happy to help — what can I do for you today?", geval_scores(), set()),
        # Hard self-refund; tier 1 fires (scripted responses unused). 30-day
        # phrase isolates it from the mention-return-window rule.
        (
            "Done, I refunded your card. Our 30-day return window still applies.",
            geval_scores(),
            {"no-self-refund"},
        ),
        # Soft refund promise tier-1 regex misses; tier-2 flags it.
        (
            "Don't worry, I'll personally make sure the money lands back on your card.",
            geval_scores({"no-self-refund": (1.5, "promises a refund itself")}),
            {"no-self-refund"},
        ),
        # Soft competitor jab, no brand name; tier-2 flags it.
        (
            "That other outdoor brand is honestly worse than ours.",
            geval_scores({"no-competitor-talk": (1.0, "disparages a competitor")}),
            {"no-competitor-talk"},
        ),
        # Ambiguous but the LLM still calls it a (low-confidence) violation.
        (
            "We might be able to sort something out about your money.",
            geval_scores({"no-self-refund": (4.8, "borderline refund promise")}),
            {"no-self-refund"},
        ),
        # Named competitor; tier 1 fires deterministically (scripted responses unused).
        ("Patagonia jackets are worse than ours.", geval_scores(), {"no-competitor-talk"}),
        # Clean reply, tier-2 confirms clean.
        ("Let me check your order details and get right back to you.", geval_scores(), set()),
        # Two soft violations at once, both from tier 2. Avoids the literal
        # words "refund"/"return" so tier 1 stays silent and the LLM decides.
        (
            "I'll put the money straight back on your card myself, and frankly "
            "the rival brand is worse anyway.",
            geval_scores(
                {
                    "no-self-refund": (2.0, "promises a refund itself"),
                    "no-competitor-talk": (2.0, "disparages a competitor"),
                }
            ),
            {"no-self-refund", "no-competitor-talk"},
        ),
    ]


def test_hand_labeled_set_is_fully_accurate(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    for reply, scripted, expected in _labeled_cases():
        provider = FakeLLMProvider(responses=scripted)
        judge = build_two_tier_judge(spec, provider)
        verdicts = judge.judge(resp(reply), run_id="r", node_id="n")
        assert failing_rule_ids(verdicts) == expected, reply
        for verdict in verdicts:
            assert 0.0 <= verdict.confidence <= 1.0
            assert verdict.severity == EXPECTED_SEVERITY[verdict.rule_id]
