"""Tier 2: an LLM-as-judge per rule, plus the two-tier composition that runs
tier 1 first and only escalates to tier 2 when it doesn't fire.
"""

from agent_stress_test.models import AgentResponse, AgentSpec, Message, Rule, Verdict
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.judge.base import Judge, judge_rule_with_llm
from agent_stress_test.reasoning.judge.rules import RulesJudge, build_checks

_TRIGGER_ONLY_CHECK_TYPES = {"required_disclaimer"}


class LLMJudge(Judge):
    """Tier-2 interpretive judge: one LLM call per rule, judged against the
    reply alone (no conversation history).

    A rule whose subject matter never came up is ``passed=True,
    applicable=False`` rather than a false failure or a false pass.
    """

    def __init__(self, llm: LLMProvider, agent_spec: AgentSpec) -> None:
        self._llm = llm
        self._rules = list(agent_spec.rules)

    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for rule in self._rules:
            passed, applicable, confidence, reason = judge_rule_with_llm(
                self._llm,
                rule_text=rule.text,
                subject_label="Agent's reply",
                subject=response.final_reply,
            )
            verdicts.append(
                Verdict(
                    run_id=run_id,
                    node_id=node_id,
                    passed=passed,
                    rule_id=rule.id,
                    reason=reason,
                    tier="llm",
                    confidence=confidence,
                    severity=rule.severity,
                    applicable=applicable,
                )
            )
        return verdicts


class TwoTierJudge(Judge):
    """Runs tier 1 first; escalates to tier 2 only when tier 1 does not fire.

    A tier-1 failure decides the verdict, except a ``_TRIGGER_ONLY_CHECK_TYPES``
    failure, which gets a tier-2 second opinion before being finalized. If
    tier 1 is fully clean, tier 2 evaluates the reply against every rule.
    """

    def __init__(
        self, rules_judge: "RulesJudge", llm_judge: LLMJudge, llm: LLMProvider, agent_spec: AgentSpec
    ) -> None:
        self._rules_judge = rules_judge
        self._llm_judge = llm_judge
        self._llm = llm
        self._rules_by_id: dict[str, Rule] = {rule.id: rule for rule in agent_spec.rules}

    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]:
        tier1 = self._rules_judge.judge(
            response, run_id=run_id, node_id=node_id, conversation=conversation
        )
        if any(not verdict.passed for verdict in tier1):
            return [self._confirm_if_trigger_only(verdict, response) for verdict in tier1]
        return self._llm_judge.judge(
            response, run_id=run_id, node_id=node_id, conversation=conversation
        )

    def _confirm_if_trigger_only(self, verdict: Verdict, response: AgentResponse) -> Verdict:
        if verdict.passed:
            return verdict
        rule = self._rules_by_id.get(verdict.rule_id)
        if rule is None or rule.check_type not in _TRIGGER_ONLY_CHECK_TYPES:
            return verdict
        passed, applicable, confidence, reason = judge_rule_with_llm(
            self._llm,
            rule_text=rule.text,
            subject_label="Agent's reply",
            subject=response.final_reply,
        )
        if applicable and not passed:
            return verdict  # tier 2 agrees: a real violation, keep tier 1's verdict as-is
        return Verdict(
            run_id=verdict.run_id,
            node_id=verdict.node_id,
            passed=True,
            rule_id=verdict.rule_id,
            reason=reason,
            tier="llm",
            confidence=confidence,
            severity=verdict.severity,
            applicable=applicable,
        )


def build_two_tier_judge(agent_spec: AgentSpec, llm: LLMProvider) -> TwoTierJudge:
    """Compose the deterministic tier-1 checks with the tier-2 LLM judge."""
    return TwoTierJudge(RulesJudge(build_checks(agent_spec)), LLMJudge(llm, agent_spec), llm, agent_spec)
