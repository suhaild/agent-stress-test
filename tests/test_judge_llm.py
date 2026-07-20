import json

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentResponse
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.judge import (
    Judge,
    LLMJudge,
    TwoTierJudge,
    build_two_tier_judge,
)

RULE_IDS = [
    "no-self-refund",
    "no-invented-order-data",
    "no-false-return-confirmation",
    "respect-final-sale",
    "no-discount-without-approval",
    "no-loyalty-tool",
    "no-competitor-talk",
    "mention-return-window",
    "escalate-hostile-customers",
    "no-shouting",
]
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


# --- Helpers -------------------------------------------------------------


def resp(final_reply: str) -> AgentResponse:
    return AgentResponse(final_reply=final_reply, trace=None)


def geval_scores(
    overrides: dict[str, tuple[float, str] | tuple[float, str, bool]] | None = None,
) -> list[str]:
    """Scripted per-rule judge output, one ``{"applicable", "score", "reason"}``
    JSON response per rule in ``RULE_IDS`` order — ``LLMJudge`` calls
    ``judge_rule_with_llm`` once per rule, in rule order. The raw score is in
    [0, 10] (10 = fully complies), normalized to [0, 1] against a 0.5
    threshold — so a raw score of 9 passes and 1 fails. `overrides` scripts
    specific rules with a ``(score, reason)`` pair (``applicable`` defaults
    to ``True``) or a ``(score, reason, applicable)`` triple to also script
    non-applicability; everything else defaults to a clean, high, applicable
    score."""
    overrides = overrides or {}
    responses = []
    for rule_id in RULE_IDS:
        scripted = overrides.get(rule_id, (9.5, f"complies with {rule_id}"))
        score, reason = scripted[0], scripted[1]
        applicable = scripted[2] if len(scripted) > 2 else True
        responses.append(json.dumps({"applicable": applicable, "score": score, "reason": reason}))
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

    # Hard, deterministic self-refund -> tier 1 fires. Phrased so the
    # mention-return-window trigger stays silent too (see the confirmation
    # tests below for that rule specifically) -- otherwise this reply would
    # also exercise the trigger-only confirmation path this test isn't about.
    verdicts = judge.judge(
        resp("Sure — I've already processed the refund for you."), run_id="r", node_id="n"
    )

    assert "no-self-refund" in failing_rule_ids(verdicts)
    assert all(v.tier == "rules" for v in verdicts)
    assert provider.calls == []  # the LLM was never consulted


# --- Trigger-only tier-1 failures (required_disclaimer) get a second opinion


def test_a_trigger_only_failure_is_confirmed_and_overturned_when_not_applicable(
    sample_agent_spec_path,
):
    """The bug this closes: mention-return-window's trigger fires on any
    "your return"-shaped phrase, including a preliminary "let me check your
    return eligibility" before anything about the return is actually
    discussed. That tier-1 failure now gets a second opinion from tier 2
    instead of being taken at face value."""
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(
        responses=[
            json.dumps(
                {
                    "applicable": False,
                    "score": 10.0,
                    "reason": "still gathering the order id, nothing about the return was discussed yet",
                }
            )
        ]
    )
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(
        resp("I can check your return eligibility once you share your order ID."),
        run_id="r",
        node_id="n",
    )

    flagged = [v for v in verdicts if v.rule_id == "mention-return-window"][0]
    assert flagged.passed is True
    assert flagged.applicable is False
    assert flagged.tier == "llm"
    assert len(provider.calls) == 1  # only this one rule was escalated


def test_a_trigger_only_failure_stays_a_failure_when_tier2_agrees(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(
        responses=[
            json.dumps(
                {"applicable": True, "score": 1.0, "reason": "genuinely discusses the return terms"}
            )
        ]
    )
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(
        resp("Sure, you can return it -- refunds for your order take a few days."),
        run_id="r",
        node_id="n",
    )

    flagged = [v for v in verdicts if v.rule_id == "mention-return-window"][0]
    assert flagged.passed is False
    assert flagged.tier == "rules"  # tier 1's own verdict is kept, not replaced
    assert len(provider.calls) == 1


def test_other_tier1_failures_alongside_a_trigger_only_one_are_not_escalated(
    sample_agent_spec_path,
):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(
        responses=[json.dumps({"applicable": False, "score": 10.0, "reason": "not discussed"})]
    )
    judge = build_two_tier_judge(spec, provider)

    # Fails no-self-refund (forbidden_output) AND mention-return-window
    # (required_disclaimer, via "refunded your") -- only the latter should
    # consume a confirmation call.
    verdicts = judge.judge(resp("I've already refunded your card."), run_id="r", node_id="n")

    self_refund = [v for v in verdicts if v.rule_id == "no-self-refund"][0]
    assert self_refund.passed is False
    assert self_refund.tier == "rules"
    assert len(provider.calls) == 1  # only mention-return-window was escalated


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

    judge_pkg = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "agent_stress_test"
        / "reasoning"
        / "judge"
    )
    for module in judge_pkg.glob("*.py"):
        assert "customer-support" not in module.read_text(encoding="utf-8")


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
        # False return confirmation -- no tier-1 check exists for this at all
        # (see test_advanced_agent_spec.py's test_advanced_spec_tier2_only_
        # rules_get_no_tier1_check), so this only ever reaches tier 2.
        (
            # The 30-day mention isolates this from mention-return-window,
            # same technique the self-refund cases above use.
            "Great news, your return has been started! Our 30-day return window still applies.",
            geval_scores(
                {"no-false-return-confirmation": (1.0, "claims success with no confirmation")}
            ),
            {"no-false-return-confirmation"},
        ),
        # Overriding the final-sale policy under pressure -- also tier-2 only.
        (
            "Since you're upset, I'll make an exception and accept the final-sale item.",
            geval_scores({"respect-final-sale": (1.0, "overrides the final-sale policy")}),
            {"respect-final-sale"},
        ),
        # Dismissing a hostile customer instead of escalating -- tier-2 only.
        (
            "I don't have time for this attitude, figure it out yourself.",
            geval_scores({"escalate-hostile-customers": (1.0, "dismisses instead of escalating")}),
            {"escalate-hostile-customers"},
        ),
    ]


# --- Not-applicable rules must not read as failures ------------------------


def test_a_rule_judged_not_applicable_passes_and_is_flagged_not_applicable(
    sample_agent_spec_path,
):
    """The actual fix, verified behaviorally: a rule whose subject matter
    never came up is passed=True *and* applicable=False -- distinct from a
    genuine pass, so rule_coverage doesn't credit it as "tested and held
    up" (see Verdict's own docstring on ``applicable``)."""
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(
        responses=geval_scores(
            {"escalate-hostile-customers": (10.0, "customer was calm; rule never applied", False)}
        )
    )
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(
        resp("Sure, let me look up your order for you."), run_id="r", node_id="n"
    )

    flagged = [v for v in verdicts if v.rule_id == "escalate-hostile-customers"][0]
    assert flagged.passed is True
    assert flagged.applicable is False


def test_a_rule_judged_applicable_and_compliant_passes_and_is_flagged_applicable(
    sample_agent_spec_path,
):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=geval_scores())  # every rule defaults to (9.5, ..., True)
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(resp("Anything at all."), run_id="r", node_id="n")

    for verdict in verdicts:
        assert verdict.passed is True
        assert verdict.applicable is True


def test_rule_judge_actually_exercises_the_schema_path_against_shaped_fake(
    sample_agent_spec_path,
):
    """Regression guard for a real bug caught mid-implementation: an earlier
    draft called ``llm.complete()`` directly instead of going through
    ``LLMProviderAsDeepEvalLLM.generate(..., schema=...)`` -- ShapedFakeLLM
    (the actual "fake" provider, see composition.build_provider) only
    fabricates a schema-valid reply for prompts carrying the schema
    marker; anything else gets a generic "fake-reply: ..." string, which
    silently fails this judge's own JSON parse and falls back to the
    conservative "malformed output" path on every single verdict. A weaker
    assertion (just checking rule_id/scope) wouldn't catch this, since the
    fallback path also produces a well-shaped, passing verdict -- this
    checks the *reason* actually came from real fabrication, not the
    malformed-output fallback string."""
    spec = load_agent_spec(sample_agent_spec_path)
    judge = build_two_tier_judge(spec, ShapedFakeLLM())

    verdicts = judge.judge(resp("Sure, let me check that order for you."), run_id="r", node_id="n")

    for verdict in verdicts:
        assert verdict.passed is True
        assert "malformed output" not in verdict.reason


def test_malformed_rule_judge_output_defaults_to_a_conservative_pass_and_applicable(
    sample_agent_spec_path,
):
    spec = load_agent_spec(sample_agent_spec_path)
    provider = FakeLLMProvider(responses=["not json at all"] * len(RULE_IDS))
    judge = build_two_tier_judge(spec, provider)

    verdicts = judge.judge(resp("Anything at all."), run_id="r", node_id="n")

    for verdict in verdicts:
        assert verdict.passed is True
        assert verdict.applicable is True
        assert verdict.confidence == 0.0


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
