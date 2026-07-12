import json

from agent_stress_test.config import load_agent_spec
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.remediation import RemediationSuggester


def _suggestion_json(prompt: str, rationale: str = "because", confidence: float = 0.8) -> str:
    return json.dumps(
        {"suggested_system_prompt": prompt, "rationale": rationale, "confidence": confidence}
    )


def _rule(spec, rule_id: str):
    return next(r for r in spec.rules if r.id == rule_id)


def test_remediation_suggester_parses_a_valid_response(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    rule = _rule(spec, "no-self-refund")
    provider = FakeLLMProvider(
        responses=[_suggestion_json("Never say the word refund yourself.", "keeps the promise")]
    )

    suggestion = RemediationSuggester(provider).suggest(
        spec, rule, "I've already refunded your card.", "Agent processed a refund itself."
    )

    assert suggestion.suggested_system_prompt == "Never say the word refund yourself."
    assert suggestion.rationale == "keeps the promise"
    assert suggestion.confidence == 0.8


def test_remediation_suggester_clamps_confidence_into_0_1(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    rule = _rule(spec, "no-self-refund")
    provider = FakeLLMProvider(responses=[_suggestion_json("x", confidence=5.0)])

    suggestion = RemediationSuggester(provider).suggest(spec, rule, "reply", "reason")

    assert suggestion.confidence == 1.0


def test_remediation_suggester_falls_back_on_malformed_output(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    rule = _rule(spec, "no-self-refund")
    provider = FakeLLMProvider(responses=["not json at all"])

    suggestion = RemediationSuggester(provider).suggest(spec, rule, "reply", "reason")

    assert suggestion.suggested_system_prompt == spec.system_prompt
    assert suggestion.confidence == 0.0
    assert "failed" in suggestion.rationale.lower()


def test_remediation_suggester_strips_a_markdown_json_fence(sample_agent_spec_path):
    # Confirmed against a live model: Claude wraps its JSON in ```json ... ```
    # even when told to respond with ONLY the JSON object.
    spec = load_agent_spec(sample_agent_spec_path)
    rule = _rule(spec, "no-self-refund")
    fenced = "```json\n" + _suggestion_json("Revised prompt text.", "why", 0.9) + "\n```"
    provider = FakeLLMProvider(responses=[fenced])

    suggestion = RemediationSuggester(provider).suggest(spec, rule, "reply", "reason")

    assert suggestion.suggested_system_prompt == "Revised prompt text."
    assert suggestion.confidence == 0.9
