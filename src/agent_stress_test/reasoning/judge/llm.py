"""Tier 2: an LLM-as-judge (DeepEval's ``GEval``) per rule, plus the two-tier
composition that runs tier 1 first and only escalates to tier 2 when it
doesn't fire.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import ValidationError

from agent_stress_test.models import AgentResponse, AgentSpec, Message, Rule, Verdict
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.deepeval_bridge import LLMProviderAsDeepEvalLLM
from agent_stress_test.reasoning.judge.base import Judge, _confidence_from_score
from agent_stress_test.reasoning.judge.rules import RulesJudge, build_checks


def _rule_metric(rule: Rule, model: LLMProviderAsDeepEvalLLM) -> GEval:
    """One GEval metric per rule, judged against the reply alone (the same
    surface tier 1 reasons over — no conversation history, see LLMJudge's
    docstring). ``evaluation_steps`` is pinned so GEval never spends a call
    regenerating its own rubric per verdict (it otherwise would, per-call)."""
    return GEval(
        name=rule.id,
        criteria=rule.text,
        evaluation_steps=[f"Check whether the actual output complies with: {rule.text}"],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=model,
        async_mode=False,
    )


class LLMJudge(Judge):
    """Tier-2 interpretive judge: a GEval metric per rule, scored by an LLMProvider.

    Each rule becomes its own GEval metric, criteria set to the rule's own
    text (framed as the desired behavior, e.g. "Never process a refund
    yourself") — GEval's score then measures how well the reply complies, so
    ``metric.success`` (score >= threshold) is the pass/fail verdict directly.
    Severity stays config-driven, read from the rule so it matches tier 1. A
    metric that fails to produce a valid score (malformed model output) falls
    back to a conservative pass (never invents a failure) while still
    carrying a reason, a confidence, and a severity.
    """

    def __init__(self, llm: LLMProvider, agent_spec: AgentSpec) -> None:
        self._rules = list(agent_spec.rules)
        model = LLMProviderAsDeepEvalLLM(llm)
        self._metrics = {rule.id: _rule_metric(rule, model) for rule in self._rules}

    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]:
        test_case = LLMTestCase(input="", actual_output=response.final_reply)
        verdicts: list[Verdict] = []
        for rule in self._rules:
            metric = self._metrics[rule.id]
            try:
                metric.measure(test_case, _show_indicator=False)
                passed, reason, confidence = (
                    metric.success,
                    metric.reason,
                    _confidence_from_score(metric),
                )
            except ValidationError:
                passed, reason, confidence = (
                    True,
                    "Tier-2 judge returned malformed output; defaulting to pass.",
                    0.0,
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
                )
            )
        return verdicts


class TwoTierJudge(Judge):
    """Runs tier 1 first; escalates to tier 2 only when tier 1 does not fire.

    If any deterministic check fails, those verdicts decide and the LLM is not
    consulted. Otherwise the LLM judge evaluates the reply against every rule.
    """

    def __init__(self, rules_judge: "RulesJudge", llm_judge: LLMJudge) -> None:
        self._rules_judge = rules_judge
        self._llm_judge = llm_judge

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
            return tier1
        return self._llm_judge.judge(
            response, run_id=run_id, node_id=node_id, conversation=conversation
        )


def build_two_tier_judge(agent_spec: AgentSpec, llm: LLMProvider) -> TwoTierJudge:
    """Compose the deterministic tier-1 checks with the tier-2 LLM judge."""
    return TwoTierJudge(RulesJudge(build_checks(agent_spec)), LLMJudge(llm, agent_spec))
