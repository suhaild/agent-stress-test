"""Tier 2: an LLM-as-judge per rule, plus the two-tier composition that runs
tier 1 first and only escalates to tier 2 when it doesn't fire.
"""

from agent_stress_test.models import AgentResponse, AgentSpec, Message, Rule, Verdict
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.judge.base import Judge, judge_rule_with_llm
from agent_stress_test.reasoning.judge.rules import RulesJudge, build_checks

# check_types whose tier-1 failure is only a proximity/keyword trigger, with
# no sense of whether the topic was genuinely being discussed (as opposed to,
# say, a preliminary "let me check on that" before anything is actually
# decided) -- see RequiredDisclaimerCheck's docstring. The other check types
# (banned tool use, forbidden output, format, ungrounded claim) are all
# present-or-absent facts about the reply/trace with nothing to disambiguate,
# so a tier-1 failure there stands on its own.
_TRIGGER_ONLY_CHECK_TYPES = {"required_disclaimer"}


class LLMJudge(Judge):
    """Tier-2 interpretive judge: one LLM call per rule, judged against the
    reply alone (no conversation history — tier 1 reasons the same way).

    Each rule is judged via ``judge_rule_with_llm`` (see its own docstring
    for why this is a hand-rolled structured call rather than a DeepEval
    GEval metric): the model decides applicability and compliance together
    in one call, so a rule whose subject matter never came up in this reply
    is ``passed=True, applicable=False`` instead of either a false failure
    or a rule wrongly credited as "tested and held up" (see ``Verdict``'s
    own docstring on ``applicable``, and ``rule_coverage``, which reads it).
    Severity stays config-driven, read from the rule so it matches tier 1.
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

    If any deterministic check fails, those verdicts decide, EXCEPT for a
    failure from a ``_TRIGGER_ONLY_CHECK_TYPES`` check (currently just
    ``required_disclaimer``): that one specific failing verdict gets a second
    opinion from the tier-2 judge before being finalized, since its trigger
    can fire on a reply that merely gestures at the topic (e.g. "let me check
    your return eligibility") rather than actually discussing it. This costs
    one extra LLM call only on the rare node where that trigger already
    fired -- every clean reply still costs nothing extra. If tier 1 is fully
    clean, the LLM judge evaluates the reply against every rule as before.
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
