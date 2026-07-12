"""Suggests a system-prompt fix for a confirmed rule violation.

Deliberately scoped to ``system_prompt`` only: never ``rules`` (rewriting the
rule you failed is "fixing" the test, not the agent) and never ``tools``
(structural, not behavioral). Every suggestion is exactly that — a suggestion
for a human to review and paste in themselves; nothing in this codebase
writes it back to an AgentSpec automatically.
"""

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from agent_stress_test.models import AgentSpec, Message, Rule
from agent_stress_test.ports import LLMProvider

# Real models routinely wrap their JSON output in a markdown code fence even
# when told to respond with ONLY the JSON object (Claude does this reliably,
# confirmed against a live model) — model_validate_json() can't parse the
# fence's backticks, so it's stripped before validation rather than treated
# as a parse failure. Mirrors judge.py's identical helper for LLMJudge.
_JSON_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_json_fence(raw: str) -> str:
    match = _JSON_FENCE.match(raw.strip())
    return match.group(1).strip() if match else raw

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
    """Proposes a system-prompt fix for a rule a target agent violated (Strategy).

    Malformed or missing LLM output falls back to the prompt unchanged (never
    invents a plausible-looking rewrite it can't stand behind) with a
    zero-confidence rationale explaining the failure.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def suggest(
        self, spec: AgentSpec, rule: Rule, violating_reply: str, verdict_reason: str
    ) -> RemediationSuggestion:
        raw = self._llm.complete(self._prompt(spec, rule, violating_reply, verdict_reason))
        return self._parse(raw, fallback_prompt=spec.system_prompt)

    @staticmethod
    def _prompt(
        spec: AgentSpec, rule: Rule, violating_reply: str, verdict_reason: str
    ) -> list[Message]:
        user = (
            f'Current system prompt:\n"""\n{spec.system_prompt}\n"""\n\n'
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
