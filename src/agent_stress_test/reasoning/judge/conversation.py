"""Phase C2: whole-conversation metric judges.

A conversation-level judge scores a full root-to-leaf persona conversation
(a ConversationalTestCase), not one node's AgentResponse — a different unit
of judgment from ``Judge``, so it gets its own small interface
(``ConversationMetricJudge``) rather than forcing it through
``Judge.judge()``'s per-node shape.
"""

from abc import ABC, abstractmethod

from deepeval.metrics import (
    ConversationalGEval,
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TurnRelevancyMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn
from pydantic import ValidationError

from agent_stress_test.models import AgentSpec, Message, Rule, Severity, Verdict
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.calibration import METRIC_PASS_THRESHOLD, severity_from_score
from agent_stress_test.reasoning.deepeval_bridge import LLMProviderAsDeepEvalLLM
from agent_stress_test.reasoning.judge.base import _confidence_from_score


def _measure_conversation_metric(
    metric: BaseMetric,
    test_case: ConversationalTestCase,
    *,
    rule_id: str | None,
    run_id: str,
    node_id: str,
    severity: Severity | None = None,
) -> Verdict:
    """Shared measure/verdict-building step every conversation metric below
    uses — same conservative-pass-on-malformed-output contract as the
    node-level metrics (ToolArgumentJudge/TaskCompletionJudge).

    ``severity``, when given, is used as-is — the rule-driven case
    (``ConversationRuleJudge``, matching ``rule.severity`` the same way
    tier-2's node-level ``LLMJudge`` does). Left ``None`` (every other
    conversation metric here, which has no AgentSpec rule to key off), it's
    derived from the metric's own score via ``severity_from_score`` (Phase
    C3's calibration — see reasoning/calibration.py)."""
    try:
        metric.measure(test_case, _show_indicator=False)
        passed, reason = metric.success, metric.reason
        confidence = _confidence_from_score(metric)
        resolved_severity = (
            severity
            if severity is not None
            else severity_from_score(metric.score, threshold=metric.threshold)
        )
    except ValidationError:
        passed, reason, confidence = (
            True,
            f"Conversation metric '{rule_id}' returned malformed output; defaulting to pass.",
            0.0,
        )
        resolved_severity = severity if severity is not None else "minor"
    return Verdict(
        run_id=run_id,
        node_id=node_id,
        passed=passed,
        rule_id=rule_id,
        reason=reason,
        tier="llm",
        confidence=confidence,
        severity=resolved_severity,
        scope="conversation",
    )


class ConversationMetricJudge(ABC):
    """A single whole-conversation metric (Strategy) — scores an already-built
    ``ConversationalTestCase`` and returns its verdict(s), scope="conversation"."""

    @abstractmethod
    def judge_conversation(
        self, test_case: ConversationalTestCase, *, run_id: str, node_id: str
    ) -> list[Verdict]: ...


class RoleAdherenceJudge(ConversationMetricJudge):
    """Does the assistant stay in character across the whole conversation?
    DeepEval's ``RoleAdherenceMetric`` — needs ``test_case.chatbot_role``,
    which ``ConversationJudge`` sets before any judge here ever runs."""

    _RULE_ID = "role_adherence"

    def __init__(self, llm: LLMProvider) -> None:
        self._metric = RoleAdherenceMetric(
            model=LLMProviderAsDeepEvalLLM(llm), threshold=METRIC_PASS_THRESHOLD, async_mode=False
        )

    def judge_conversation(
        self, test_case: ConversationalTestCase, *, run_id: str, node_id: str
    ) -> list[Verdict]:
        return [
            _measure_conversation_metric(
                self._metric, test_case, rule_id=self._RULE_ID, run_id=run_id, node_id=node_id
            )
        ]


class KnowledgeRetentionJudge(ConversationMetricJudge):
    """Does the assistant forget/re-ask for information the user already gave
    it earlier in the conversation? DeepEval's ``KnowledgeRetentionMetric``."""

    _RULE_ID = "knowledge_retention"

    def __init__(self, llm: LLMProvider) -> None:
        self._metric = KnowledgeRetentionMetric(
            model=LLMProviderAsDeepEvalLLM(llm), threshold=METRIC_PASS_THRESHOLD, async_mode=False
        )

    def judge_conversation(
        self, test_case: ConversationalTestCase, *, run_id: str, node_id: str
    ) -> list[Verdict]:
        return [
            _measure_conversation_metric(
                self._metric, test_case, rule_id=self._RULE_ID, run_id=run_id, node_id=node_id
            )
        ]


class ConversationCompletenessJudge(ConversationMetricJudge):
    """Did the assistant fully resolve every intention the user raised across
    the conversation? DeepEval's ``ConversationCompletenessMetric``."""

    _RULE_ID = "conversation_completeness"

    def __init__(self, llm: LLMProvider) -> None:
        self._metric = ConversationCompletenessMetric(
            model=LLMProviderAsDeepEvalLLM(llm), threshold=METRIC_PASS_THRESHOLD, async_mode=False
        )

    def judge_conversation(
        self, test_case: ConversationalTestCase, *, run_id: str, node_id: str
    ) -> list[Verdict]:
        return [
            _measure_conversation_metric(
                self._metric, test_case, rule_id=self._RULE_ID, run_id=run_id, node_id=node_id
            )
        ]


class TurnRelevancyJudge(ConversationMetricJudge):
    """Does each assistant turn stay relevant to its preceding user turns?
    DeepEval's ``TurnRelevancyMetric``."""

    _RULE_ID = "turn_relevancy"

    def __init__(self, llm: LLMProvider) -> None:
        self._metric = TurnRelevancyMetric(
            model=LLMProviderAsDeepEvalLLM(llm), threshold=METRIC_PASS_THRESHOLD, async_mode=False
        )

    def judge_conversation(
        self, test_case: ConversationalTestCase, *, run_id: str, node_id: str
    ) -> list[Verdict]:
        return [
            _measure_conversation_metric(
                self._metric, test_case, rule_id=self._RULE_ID, run_id=run_id, node_id=node_id
            )
        ]


def _conversation_rule_metric(rule: Rule, model: LLMProviderAsDeepEvalLLM) -> ConversationalGEval:
    """The whole-conversation counterpart to ``_rule_metric``: the same rule
    text, scored across every turn instead of a single reply — catches
    violations that only emerge across turns (e.g. contradicting something
    promised two turns earlier), which no single-turn GEval call can see."""
    return ConversationalGEval(
        name=rule.id,
        criteria=rule.text,
        evaluation_steps=[f"Check whether the conversation as a whole complies with: {rule.text}"],
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
        model=model,
        async_mode=False,
    )


class ConversationRuleJudge(ConversationMetricJudge):
    """One ``ConversationalGEval`` per AgentSpec rule (see
    ``_conversation_rule_metric``) — the conversation-level counterpart to
    ``LLMJudge``'s per-rule node-level GEval. Returns one verdict per rule,
    keyed by the rule's own id, same as tier 2 — including severity: this is
    the one conversation metric judge NOT covered by C3's calibration (see
    reasoning/calibration.py's module docstring), since it's keyed to a real
    ``Rule`` and so correctly uses ``rule.severity`` (config, set by a
    human), exactly like tier 2's ``LLMJudge``."""

    def __init__(self, llm: LLMProvider, spec: AgentSpec) -> None:
        model = LLMProviderAsDeepEvalLLM(llm)
        self._rules_by_id = {rule.id: rule for rule in spec.rules}
        self._metrics = {rule.id: _conversation_rule_metric(rule, model) for rule in spec.rules}

    def judge_conversation(
        self, test_case: ConversationalTestCase, *, run_id: str, node_id: str
    ) -> list[Verdict]:
        return [
            _measure_conversation_metric(
                metric,
                test_case,
                rule_id=rule_id,
                run_id=run_id,
                node_id=node_id,
                severity=self._rules_by_id[rule_id].severity,
            )
            for rule_id, metric in self._metrics.items()
        ]


class ConversationJudge:
    """Scores one full persona conversation (root-to-leaf), not a node.

    Builds a single ``ConversationalTestCase`` from the turns and hands it to
    every injected ``ConversationMetricJudge`` (Composition, like
    ``CompositeJudge`` — every judge runs, no short-circuit), so DeepEval's
    turn-shape conversion happens once per conversation, not once per metric.
    Off by default in ``build_runner``: Phase C3's real measurement found
    ~19 LLM calls (~$0.02 on Haiku) per persona across the 5 bundled metrics
    (see ``build_conversation_judge``) — cheap per call, but that multiplies
    with persona count on top of the per-node judge already running every
    turn, so it stays an explicit opt-in.
    """

    def __init__(self, chatbot_role: str, judges: list[ConversationMetricJudge]) -> None:
        self._chatbot_role = chatbot_role
        self._judges = judges

    def judge_conversation(
        self,
        turns: list[Message],
        *,
        run_id: str,
        node_id: str,
        scenario: str | None = None,
        user_description: str | None = None,
    ) -> list[Verdict]:
        conv_turns = [
            Turn(role=message.role, content=message.content)
            for message in turns
            if message.role in ("user", "assistant") and isinstance(message.content, str)
        ]
        if not conv_turns:
            return []
        test_case = ConversationalTestCase(
            turns=conv_turns,
            chatbot_role=self._chatbot_role,
            scenario=scenario,
            user_description=user_description,
        )
        verdicts: list[Verdict] = []
        for judge in self._judges:
            verdicts.extend(judge.judge_conversation(test_case, run_id=run_id, node_id=node_id))
        return verdicts


def build_conversation_judge(llm: LLMProvider, spec: AgentSpec) -> ConversationJudge:
    """Compose the bundled whole-conversation metrics with a per-rule
    conversational GEval for ``spec`` (the composition-root-style factory,
    same shape as ``build_two_tier_judge``)."""
    chatbot_role = spec.purpose or spec.system_prompt
    return ConversationJudge(
        chatbot_role,
        [
            RoleAdherenceJudge(llm),
            KnowledgeRetentionJudge(llm),
            ConversationCompletenessJudge(llm),
            TurnRelevancyJudge(llm),
            ConversationRuleJudge(llm, spec),
        ],
    )
