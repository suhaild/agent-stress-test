"""Suggests a system-prompt fix for a confirmed rule violation.

Scoped to ``system_prompt`` only — never ``rules`` (that would be fixing the
test, not the agent) or ``tools`` (structural, not behavioral). A human
reviews and applies every suggestion; nothing here writes back automatically.
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from agent_stress_test.models import AgentSpec, Message, Rule
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.json_utils import _strip_json_fence

_REMEDIATION_SYSTEM = (
    "You are helping harden a support agent's system prompt so it stops violating "
    "one of its own behavioral rules. You are given the agent's current system "
    "prompt, the rule it violated, the reply that violated it, and why it "
    "violated it. Propose a MINIMAL revision to the system prompt that would "
    "have prevented this specific failure, without removing or weakening any "
    "existing instruction the prompt already gets right. Respond with ONLY a "
    'JSON object of the form {"suggested_system_prompt": str, "rationale": str, '
    '"confidence": number in [0, 1]}.'
)


class _RemediationOutput(BaseModel):
    """The tier-2 JSON payload, as parsed from the LLM's output."""

    suggested_system_prompt: str
    rationale: str
    confidence: float


@dataclass(frozen=True)
class RemediationSuggestion:
    """One proposed system-prompt fix. Internal to the reasoning layer."""

    suggested_system_prompt: str
    rationale: str
    confidence: float


class RemediationSuggester:
    """Proposes a system-prompt fix for a violated rule (Strategy).

    Malformed LLM output falls back to the prompt unchanged with a
    zero-confidence rationale, rather than inventing a plausible rewrite.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def suggest(
        self, agent_spec: AgentSpec, rule: Rule, violating_reply: str, verdict_reason: str
    ) -> RemediationSuggestion:
        raw = self._llm.complete(self._prompt(agent_spec, rule, violating_reply, verdict_reason))
        return self._parse(raw, fallback_prompt=agent_spec.system_prompt)

    @staticmethod
    def _prompt(
        agent_spec: AgentSpec, rule: Rule, violating_reply: str, verdict_reason: str
    ) -> list[Message]:
        user = (
            f'Current system prompt:\n"""\n{agent_spec.system_prompt}\n"""\n\n'
            f"Violated rule ({rule.severity}): {rule.text}\n\n"
            f'Agent reply that violated it:\n"""\n{violating_reply}\n"""\n\n'
            f"Why it violated the rule: {verdict_reason}\n\n"
            "Propose the revised system prompt and return the JSON object."
        )
        return [
            # Constant across every suggestion call — a prime prompt-caching breakpoint.
            Message(role="system", content=_REMEDIATION_SYSTEM, cache=True),
            Message(role="user", content=user),
        ]

    @staticmethod
    def _parse(raw: str, *, fallback_prompt: str) -> RemediationSuggestion:
        try:
            output = _RemediationOutput.model_validate_json(_strip_json_fence(raw))
        except (ValidationError, ValueError, json.JSONDecodeError):
            return RemediationSuggestion(
                suggested_system_prompt=fallback_prompt,
                rationale="Suggestion parsing failed; no change proposed.",
                confidence=0.0,
            )
        return RemediationSuggestion(
            suggested_system_prompt=output.suggested_system_prompt,
            rationale=output.rationale,
            confidence=max(0.0, min(1.0, output.confidence)),
        )
