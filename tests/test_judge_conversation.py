"""Phase C2 — whole-conversation metric judges.

Mirrors ``test_judge_metrics.py``'s C1 pattern: the plain ``ShapedFakeLLM``
fabricates a pass for every conversational metric (so wiring them on never
spuriously fails an offline run — confirmed per-judge below), and a scripted
``FakeLLMProvider`` returning the exact per-metric JSON forces the mandated
failure path (a conversation where the assistant breaks character scores low
on RoleAdherence).
"""

import json

from deepeval.test_case import ConversationalTestCase, Turn

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Message
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.judge import (
    ConversationCompletenessJudge,
    ConversationJudge,
    ConversationMetricJudge,
    ConversationRuleJudge,
    KnowledgeRetentionJudge,
    RoleAdherenceJudge,
    TurnRelevancyJudge,
    build_conversation_judge,
)

_STAYS_IN_CHARACTER = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Hi, can you help me with my order?"),
        Turn(role="assistant", content="Of course — what's your order number?"),
        Turn(role="user", content="It's 12345."),
        Turn(role="assistant", content="Thanks, let me check that for you now."),
    ],
    chatbot_role="a warm, in-character customer support agent named Sam",
)

_BREAKS_CHARACTER = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Who are you, really?"),
        Turn(role="assistant", content="I'm just an AI language model with no real persona."),
        Turn(role="user", content="Okay, whatever."),
        Turn(role="assistant", content="As an AI, I don't actually have feelings or a name."),
    ],
    chatbot_role="a warm, in-character customer support agent named Sam",
)


# --- RoleAdherence: the mandated failure case -------------------------------
# RoleAdherenceMetric makes exactly two calls: out-of-character verdicts, then
# the aggregate reason (see reasoning/judge.py's RoleAdherenceJudge docstring).


def test_role_adherence_judge_flags_a_conversation_that_breaks_character():
    provider = FakeLLMProvider(
        responses=[
            json.dumps(
                {
                    "verdicts": [
                        {"index": 0, "reason": "reveals it is an AI model"},
                        {"index": 1, "reason": "denies having a name or feelings"},
                    ]
                }
            ),
            json.dumps({"reason": "The assistant broke character on both of its turns."}),
        ]
    )
    [verdict] = RoleAdherenceJudge(provider).judge_conversation(
        _BREAKS_CHARACTER, run_id="r", node_id="leaf"
    )

    assert verdict.scope == "conversation"
    assert verdict.rule_id == "role_adherence"
    assert verdict.node_id == "leaf"
    assert verdict.passed is False
    assert verdict.reason == "The assistant broke character on both of its turns."


def test_role_adherence_judge_passes_a_conversation_that_stays_in_character():
    provider = FakeLLMProvider(
        responses=[
            json.dumps({"verdicts": []}),
            json.dumps({"reason": "The assistant stayed in character throughout."}),
        ]
    )
    [verdict] = RoleAdherenceJudge(provider).judge_conversation(
        _STAYS_IN_CHARACTER, run_id="r", node_id="leaf"
    )

    assert verdict.passed is True


# --- Every conversation metric judge is offline-safe by default ------------


def test_conversation_metric_judges_never_spuriously_fail_offline():
    fake = ShapedFakeLLM()
    judges: list[ConversationMetricJudge] = [
        RoleAdherenceJudge(fake),
        KnowledgeRetentionJudge(fake),
        ConversationCompletenessJudge(fake),
        TurnRelevancyJudge(fake),
    ]
    for judge in judges:
        [verdict] = judge.judge_conversation(_STAYS_IN_CHARACTER, run_id="r", node_id="leaf")
        assert verdict.scope == "conversation"
        assert verdict.passed is True


# --- ConversationRuleJudge: one conversational GEval per AgentSpec rule ---


def test_conversation_rule_judge_returns_one_verdict_per_rule(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    judge = ConversationRuleJudge(ShapedFakeLLM(), spec)

    verdicts = judge.judge_conversation(_STAYS_IN_CHARACTER, run_id="r", node_id="leaf")

    assert {v.rule_id for v in verdicts} == {rule.id for rule in spec.rules}
    assert all(v.scope == "conversation" for v in verdicts)


def test_conversation_rule_judge_actually_exercises_the_schema_path_against_shaped_fake(
    sample_agent_spec_path,
):
    """Same regression guard as LLMJudge's: a bare ``llm.complete()`` call
    (no schema marker) silently degrades to the "malformed output" fallback
    against ShapedFakeLLM on every verdict -- a weaker assertion (just
    rule_id/scope, as above) wouldn't catch that, since the fallback path
    also produces a well-shaped, passing verdict."""
    spec = load_agent_spec(sample_agent_spec_path)
    judge = ConversationRuleJudge(ShapedFakeLLM(), spec)

    verdicts = judge.judge_conversation(_STAYS_IN_CHARACTER, run_id="r", node_id="leaf")

    for verdict in verdicts:
        assert verdict.passed is True
        assert "malformed output" not in verdict.reason


def test_conversation_rule_judge_marks_a_not_applicable_rule_passed_but_not_applicable(
    sample_agent_spec_path,
):
    """Same false-positive pattern as LLMJudge's per-node judge, at
    conversation scope: this runs one judgment per rule per persona
    conversation regardless of whether that rule's topic ever came up in
    it -- verified behaviorally, not just as a prompt-wording check."""
    spec = load_agent_spec(sample_agent_spec_path)

    def scripted_response(rule_id: str) -> str:
        if rule_id == "escalate-hostile-customers":
            return json.dumps({"applicable": False, "score": 10.0, "reason": "never came up"})
        return json.dumps({"applicable": True, "score": 9.5, "reason": "complies"})

    scripted = [scripted_response(rule.id) for rule in spec.rules]
    judge = ConversationRuleJudge(FakeLLMProvider(responses=scripted), spec)

    verdicts = judge.judge_conversation(_STAYS_IN_CHARACTER, run_id="r", node_id="leaf")

    flagged = [v for v in verdicts if v.rule_id == "escalate-hostile-customers"][0]
    assert flagged.passed is True
    assert flagged.applicable is False
    assert flagged.scope == "conversation"
    for verdict in verdicts:
        if verdict.rule_id != "escalate-hostile-customers":
            assert verdict.applicable is True


# --- ConversationJudge: composes every injected judge, no short-circuit ---


def test_conversation_judge_unions_verdicts_from_every_injected_judge():
    fake = ShapedFakeLLM()
    judge = ConversationJudge(
        "a helpful assistant", [RoleAdherenceJudge(fake), KnowledgeRetentionJudge(fake)]
    )

    turns = [Message(role="user", content="hi")]
    verdicts = judge.judge_conversation(turns, run_id="r", node_id="leaf")

    assert {v.rule_id for v in verdicts} == {"role_adherence", "knowledge_retention"}
    assert all(v.node_id == "leaf" and v.scope == "conversation" for v in verdicts)


def test_conversation_judge_returns_nothing_for_a_conversation_with_no_user_or_assistant_turns():
    judge = ConversationJudge("a helpful assistant", [RoleAdherenceJudge(ShapedFakeLLM())])

    verdicts = judge.judge_conversation(
        [Message(role="system", content="setup only")], run_id="r", node_id="leaf"
    )

    assert verdicts == []


def test_build_conversation_judge_wires_the_four_builtins_plus_every_rule(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    judge = build_conversation_judge(ShapedFakeLLM(), spec)

    verdicts = judge.judge_conversation(
        [
            Message(role="user", content="Hi, I need help with my order."),
            Message(role="assistant", content="Happy to help — what's your order number?"),
        ],
        run_id="r",
        node_id="leaf",
    )

    builtin_ids = {
        "role_adherence",
        "knowledge_retention",
        "conversation_completeness",
        "turn_relevancy",
    }
    rule_ids = {rule.id for rule in spec.rules}
    assert {v.rule_id for v in verdicts} == builtin_ids | rule_ids
    assert all(v.scope == "conversation" for v in verdicts)
    assert all(v.passed for v in verdicts)  # ShapedFakeLLM fabricates a clean pass throughout
