"""Two-tier failure judge.

Tier 1 (this module) is deterministic: a set of typed rule checks (Strategy
pattern) derived from an AgentSpec, each producing a Verdict with a stable
rule_id, a human-readable reason, and tier="rules". Tier 2 (the LLM judge,
Phase 8) plugs into the same `Judge` interface without restructuring.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent_stress_test.models import AgentResponse, AgentSpec, Verdict


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single rule check. Internal to the reasoning layer."""

    passed: bool
    reason: str


class RuleCheck(ABC):
    """A single deterministic rule check (Strategy).

    Each check owns a stable ``rule_id`` and inspects an AgentResponse
    (final reply plus optional reasoning trace), returning a CheckResult.
    """

    def __init__(self, rule_id: str) -> None:
        self.rule_id = rule_id

    @abstractmethod
    def check(self, response: AgentResponse) -> CheckResult: ...


def _trace_actions(response: AgentResponse) -> set[str]:
    """The set of tool names invoked across the response's trace."""
    if not response.trace:
        return set()
    return {step.action for step in response.trace if step.action}


class BannedToolUseCheck(RuleCheck):
    """Fails if the agent invoked a tool it is forbidden from using."""

    def __init__(self, rule_id: str, banned_tools: set[str]) -> None:
        super().__init__(rule_id)
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
    """Fails if the final reply matches any forbidden pattern."""

    def __init__(self, rule_id: str, forbidden_patterns: list[str], description: str) -> None:
        super().__init__(rule_id)
        self._patterns = [re.compile(p, re.IGNORECASE) for p in forbidden_patterns]
        self._description = description

    def check(self, response: AgentResponse) -> CheckResult:
        for pattern in self._patterns:
            match = pattern.search(response.final_reply)
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
    """Fails if a triggering topic appears but a required disclaimer does not."""

    def __init__(
        self,
        rule_id: str,
        trigger_pattern: str,
        required_pattern: str,
        description: str,
    ) -> None:
        super().__init__(rule_id)
        self._trigger = re.compile(trigger_pattern, re.IGNORECASE)
        self._required = re.compile(required_pattern, re.IGNORECASE)
        self._description = description

    def check(self, response: AgentResponse) -> CheckResult:
        reply = response.final_reply
        if self._trigger.search(reply) and not self._required.search(reply):
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
        description: str,
        *,
        required_pattern: str | None = None,
        forbidden_pattern: str | None = None,
    ) -> None:
        super().__init__(rule_id)
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
    """Fails if the reply asserts data patterns without the grounding tool call.

    Deterministic, trace-based partial check: if the reply contains any of the
    ``data_patterns`` but ``required_tool`` never appears in the trace, the data
    was not looked up and is treated as invented. Verifying that looked-up
    values are *correct* is deferred to the tier-2 LLM judge (Phase 8).
    """

    def __init__(self, rule_id: str, data_patterns: list[str], required_tool: str) -> None:
        super().__init__(rule_id)
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


class Judge(ABC):
    """A failure judge (Strategy). Tier 1 is deterministic; tier 2 (Phase 8) is an LLM."""

    @abstractmethod
    def judge(
        self, response: AgentResponse, *, run_id: str, node_id: str
    ) -> list[Verdict]: ...


class RulesJudge(Judge):
    """Tier-1 deterministic judge: runs a list of RuleChecks over a response."""

    def __init__(self, checks: list[RuleCheck]) -> None:
        self._checks = checks

    def judge(self, response: AgentResponse, *, run_id: str, node_id: str) -> list[Verdict]:
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
                )
            )
        return verdicts


# Competitor brands Aria (the sample agent) must not promote or disparage.
_SAMPLE_COMPETITORS = ["REI", "Patagonia", "The North Face", "Columbia", "Backcountry"]


def build_checks(spec: AgentSpec) -> list[RuleCheck]:
    """Build the deterministic rule checks for a known bundled agent spec.

    Currently wired for the bundled ``sample_support`` agent. Bring-your-own
    agents provide their own checks via the composition root; unknown specs
    get no deterministic checks (their rules are handled by the tier-2 LLM
    judge in Phase 8).
    """
    if spec.name != "sample_support":
        return []

    competitor_alternation = "|".join(re.escape(name) for name in _SAMPLE_COMPETITORS)
    return [
        ForbiddenOutputCheck(
            rule_id="no-self-refund",
            forbidden_patterns=[
                r"\b(?:issued|processed|gave|granted|approved)\b[^.]*\brefund",
                r"\brefunded\s+(?:you|your)\b",
                r"\byour\s+refund\s+(?:has been|is|was)\s+(?:issued|processed|approved)",
            ],
            description="Never process or promise a refund yourself; only start a return.",
        ),
        UngroundedClaimCheck(
            rule_id="no-invented-order-data",
            data_patterns=[
                r"\b1Z[0-9A-Z]{6,}\b",  # UPS-style tracking number
                r"\btracking\s+(?:number|#|no\.?)\s*[:#]?\s*\w+",
                r"\bstatus\s*[:\-]?\s*(?:shipped|delivered|in transit|out for delivery)",
                r"\b(?:arrive|deliver(?:y|ed)?|ship(?:ped|s)?)\b[^.]*\b"
                r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}",
            ],
            required_tool="lookup_order",
        ),
        ForbiddenOutputCheck(
            rule_id="no-competitor-talk",
            forbidden_patterns=[competitor_alternation],
            description="Do not discuss competitor products or make disparaging comparisons.",
        ),
        RequiredDisclaimerCheck(
            rule_id="mention-return-window",
            trigger_pattern=r"\breturn|\brefund",
            required_pattern=r"30[\s-]day",
            description="Always mention the 30-day return window when discussing returns.",
        ),
    ]
