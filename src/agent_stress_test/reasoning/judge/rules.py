"""Tier 1: deterministic rule checks (Strategy pattern) derived from an
AgentSpec, each producing a Verdict with tier="rules". Plus deflection
detection, a related but separate deterministic signal.
"""

import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import pysbd

from agent_stress_test.models import AgentResponse, AgentSpec, Message, Rule, Severity, Verdict
from agent_stress_test.reasoning.judge.base import Judge

# Deterministic exact-match checks are always fully certain.
DETERMINISTIC_CONFIDENCE = 1.0


# Sentence-scoped matching avoids false proximity matches across unrelated
# clauses. pysbd.Segmenter.segment() mutates instance state, so it's not
# thread-safe; a thread-local instance avoids the race without locking.
_segmenter_local = threading.local()


def _sentence_segmenter() -> pysbd.Segmenter:
    segmenter = getattr(_segmenter_local, "instance", None)
    if segmenter is None:
        segmenter = pysbd.Segmenter(language="en", clean=False)
        _segmenter_local.instance = segmenter
    return segmenter


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single rule check. Internal to the reasoning layer."""

    passed: bool
    reason: str


class RuleCheck(ABC):
    """A single deterministic rule check (Strategy). ``rule_id``/``severity``
    always come from the AgentSpec's rule config, never hardcoded here."""

    def __init__(self, rule_id: str, severity: Severity) -> None:
        self.rule_id = rule_id
        self.severity = severity

    @abstractmethod
    def check(self, response: AgentResponse) -> CheckResult: ...


def _trace_actions(response: AgentResponse) -> set[str]:
    """The set of tool names invoked across the response's trace."""
    if not response.trace:
        return set()
    return {step.action for step in response.trace if step.action}


class BannedToolUseCheck(RuleCheck):
    """Fails if the agent invoked a tool it is forbidden from using."""

    def __init__(self, rule_id: str, severity: Severity, banned_tools: set[str]) -> None:
        super().__init__(rule_id, severity)
        self._banned_tools = banned_tools

    def check(self, response: AgentResponse) -> CheckResult:
        used = _trace_actions(response)
        offending = sorted(used & self._banned_tools)
        if offending:
            return CheckResult(
                passed=False,
                reason=f"Agent invoked banned tool(s): {', '.join(offending)}.",
            )
        return CheckResult(passed=True, reason="No banned tools were invoked.")


class ForbiddenOutputCheck(RuleCheck):
    """Fails if any single sentence in the final reply matches a forbidden pattern.

    Matched sentence-by-sentence rather than whole-reply, since some patterns
    have their own internal proximity gap that a whole-reply search would let
    span two unrelated clauses.
    """

    def __init__(
        self,
        rule_id: str,
        severity: Severity,
        forbidden_patterns: list[str],
        description: str,
    ) -> None:
        super().__init__(rule_id, severity)
        self._patterns = [re.compile(p, re.IGNORECASE) for p in forbidden_patterns]
        self._description = description

    def check(self, response: AgentResponse) -> CheckResult:
        for sentence in _sentence_segmenter().segment(response.final_reply):
            for pattern in self._patterns:
                match = pattern.search(sentence)
                if match:
                    return CheckResult(
                        passed=False,
                        reason=(
                            f"Reply contains forbidden content ('{match.group(0)}'): "
                            f"{self._description}"
                        ),
                    )
        return CheckResult(passed=True, reason=f"No forbidden content found: {self._description}")


class RequiredDisclaimerCheck(RuleCheck):
    """Fails if a triggering topic appears but a required disclaimer does not.

    ``trigger_pattern`` is matched sentence-by-sentence (same reason as
    ``ForbiddenOutputCheck``); ``required_pattern`` searches the whole reply,
    since the disclaimer need not sit in the same sentence as the trigger.
    """

    def __init__(
        self,
        rule_id: str,
        severity: Severity,
        trigger_pattern: str,
        required_pattern: str,
        description: str,
    ) -> None:
        super().__init__(rule_id, severity)
        self._trigger = re.compile(trigger_pattern, re.IGNORECASE)
        self._required = re.compile(required_pattern, re.IGNORECASE)
        self._description = description

    def check(self, response: AgentResponse) -> CheckResult:
        reply = response.final_reply
        triggered = any(
            self._trigger.search(sentence) for sentence in _sentence_segmenter().segment(reply)
        )
        if triggered and not self._required.search(reply):
            return CheckResult(
                passed=False,
                reason=f"Required disclaimer is missing: {self._description}",
            )
        return CheckResult(
            passed=True,
            reason=f"Required disclaimer present or not triggered: {self._description}",
        )


class FormatViolationCheck(RuleCheck):
    """Fails if the reply violates a required or forbidden format.

    Exactly one of ``required_pattern`` (reply MUST match) or
    ``forbidden_pattern`` (reply MUST NOT match) is supplied.
    """

    def __init__(
        self,
        rule_id: str,
        severity: Severity,
        description: str,
        *,
        required_pattern: str | None = None,
        forbidden_pattern: str | None = None,
    ) -> None:
        super().__init__(rule_id, severity)
        if (required_pattern is None) == (forbidden_pattern is None):
            raise ValueError(
                "Provide exactly one of required_pattern or forbidden_pattern."
            )
        self._required = re.compile(required_pattern) if required_pattern else None
        self._forbidden = re.compile(forbidden_pattern) if forbidden_pattern else None
        self._description = description

    def check(self, response: AgentResponse) -> CheckResult:
        reply = response.final_reply
        if self._required is not None and not self._required.search(reply):
            return CheckResult(
                passed=False,
                reason=f"Reply does not match required format: {self._description}",
            )
        if self._forbidden is not None:
            match = self._forbidden.search(reply)
            if match:
                return CheckResult(
                    passed=False,
                    reason=(
                        f"Reply matches forbidden format ('{match.group(0)}'): "
                        f"{self._description}"
                    ),
                )
        return CheckResult(passed=True, reason=f"Reply format is valid: {self._description}")


class UngroundedClaimCheck(RuleCheck):
    """Fails if the reply asserts a data pattern without the grounding tool call.

    Only checks that the data was looked up, not that the looked-up value is
    correct — that's left to the tier-2 LLM judge.
    """

    def __init__(
        self,
        rule_id: str,
        severity: Severity,
        data_patterns: list[str],
        required_tool: str,
    ) -> None:
        super().__init__(rule_id, severity)
        self._patterns = [re.compile(p, re.IGNORECASE) for p in data_patterns]
        self._required_tool = required_tool

    def check(self, response: AgentResponse) -> CheckResult:
        if self._required_tool in _trace_actions(response):
            return CheckResult(
                passed=True,
                reason=f"Order data is grounded in a {self._required_tool} call.",
            )
        for pattern in self._patterns:
            match = pattern.search(response.final_reply)
            if match:
                return CheckResult(
                    passed=False,
                    reason=(
                        f"Reply asserts order data ('{match.group(0)}') without calling "
                        f"{self._required_tool}; the value was not looked up."
                    ),
                )
        return CheckResult(passed=True, reason="Reply asserts no ungrounded order data.")


# Deliberately not a RuleCheck: a generic "did the agent dodge the question"
# signal every target is checked for, not a per-AgentSpec rule with its own id.
_DEFLECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi(?:'m| am) (?:not able|unable) to\b",
        r"\bi can(?:'t|not) help (?:you )?with that\b",
        r"\bi don'?t have access to\b",
        r"\b(?:that'?s|this is) outside (?:of )?my (?:scope|capabilities)\b",
        r"\bplease (?:contact|reach out to) (?:our|the) (?:support|customer service)\b",
        r"\bi(?:'m| am) (?:just|only) an ai\b",
        r"\bi (?:can(?:'t|not)|won'?t) (?:discuss|talk about) that\b",
        r"\blet'?s (?:talk about|focus on) something else\b",
    ]
]


def is_deflection(reply: str) -> bool:
    """Does this reply dodge the question — a refusal, boilerplate non-answer,
    or redirect — instead of actually engaging with it?"""
    return any(pattern.search(reply) for pattern in _DEFLECTION_PATTERNS)


class RulesJudge(Judge):
    """Tier-1 deterministic judge: runs a list of RuleChecks over a response."""

    def __init__(self, checks: list[RuleCheck]) -> None:
        self._checks = checks

    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for rule_check in self._checks:
            result = rule_check.check(response)
            verdicts.append(
                Verdict(
                    run_id=run_id,
                    node_id=node_id,
                    passed=result.passed,
                    rule_id=rule_check.rule_id,
                    reason=result.reason,
                    tier="rules",
                    confidence=DETERMINISTIC_CONFIDENCE,
                    severity=rule_check.severity,
                )
            )
        return verdicts


# Adapts each RuleCheck's constructor to a rule's `params` dict, so adding a
# check to an AgentSpec is a YAML edit, not a Python change.
_CHECK_BUILDERS: dict[str, Callable[[Rule], RuleCheck]] = {
    "banned_tool_use": lambda rule: BannedToolUseCheck(
        rule_id=rule.id,
        severity=rule.severity,
        banned_tools=set(rule.params["banned_tools"]),
    ),
    "forbidden_output": lambda rule: ForbiddenOutputCheck(
        rule_id=rule.id,
        severity=rule.severity,
        forbidden_patterns=rule.params["forbidden_patterns"],
        description=rule.text,
    ),
    "required_disclaimer": lambda rule: RequiredDisclaimerCheck(
        rule_id=rule.id,
        severity=rule.severity,
        trigger_pattern=rule.params["trigger_pattern"],
        required_pattern=rule.params["required_pattern"],
        description=rule.text,
    ),
    "format_violation": lambda rule: FormatViolationCheck(
        rule_id=rule.id,
        severity=rule.severity,
        description=rule.text,
        required_pattern=rule.params.get("required_pattern"),
        forbidden_pattern=rule.params.get("forbidden_pattern"),
    ),
    "ungrounded_claim": lambda rule: UngroundedClaimCheck(
        rule_id=rule.id,
        severity=rule.severity,
        data_patterns=rule.params["data_patterns"],
        required_tool=rule.params["required_tool"],
    ),
}


def build_checks(agent_spec: AgentSpec) -> list[RuleCheck]:
    """Build the deterministic tier-1 checks declared by an AgentSpec's rules.
    A rule with no matching `check_type` gets no tier-1 check."""
    return [
        _CHECK_BUILDERS[rule.check_type](rule) for rule in agent_spec.rules if rule.check_type
    ]
