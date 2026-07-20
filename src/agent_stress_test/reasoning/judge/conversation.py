"""Whole-conversation metric judges.

Scores a full root-to-leaf persona conversation (a ConversationalTestCase),
a different unit of judgment from ``Judge``, so it gets its own small
interface (``ConversationMetricJudge``) instead of the per-node shape.
"""

from abc import ABC, abstractmethod

from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TurnRelevancyMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import ConversationalTestCase, Turn
from pydantic import ValidationError

from agent_stress_test.models import AgentSpec, Message, Severity, Verdict
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.calibration import METRIC_PASS_THRESHOLD, severity_from_score
from agent_stress_test.reasoning.deepeval_bridge import LLMProviderAsDeepEvalLLM
from agent_stress_test.reasoning.judge.base import _confidence_from_score, judge_rule_with_llm


def _measure_conversation_metric(
    metric: BaseMetric,
    test_case: ConversationalTestCase,
    *,
    rule_id: str | None,
    run_id: str,
    node_id: str,
    severity: Severity | None = None,
) -> Verdict:
    """Shared measure/verdict-building step for every conversation metric;
    same conservative-pass-on-malformed-output contract as the node-level
    metrics. ``severity`` overrides the score-derived default when the
    caller already has a rule-configured severity to use."""
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


def _format_transcript(test_case: ConversationalTestCase) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in test_case.turns)


class ConversationRuleJudge(ConversationMetricJudge):
    """One rule judged across the whole conversation per AgentSpec rule —
    catches violations that only emerge across turns (e.g. contradicting
    something promised earlier), which no single-turn check can see. Uses
    ``rule.severity`` directly rather than the score-derived default, like
    tier 2's ``LLMJudge``.
    """

    def __init__(self, llm: LLMProvider, agent_spec: AgentSpec) -> None:
        self._llm = llm
        self._rules = list(agent_spec.rules)

    def judge_conversation(
        self, test_case: ConversationalTestCase, *, run_id: str, node_id: str
    ) -> list[Verdict]:
        transcript = _format_transcript(test_case)
        verdicts = []
        for rule in self._rules:
            passed, applicable, confidence, reason = judge_rule_with_llm(
                self._llm,
                rule_text=rule.text,
                subject_label="Conversation transcript",
                subject=transcript,
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
                    scope="conversation",
                    applicable=applicable,
                )
            )
        return verdicts


class ConversationJudge:
    """Scores one full persona conversation (root-to-leaf), not a node.

    Builds a single ``ConversationalTestCase`` and hands it to every injected
    ``ConversationMetricJudge`` (every judge runs, no short-circuit). Off by
    default in ``build_runner``: several LLM calls per persona on top of the
    per-node judge already running every turn, so it stays an explicit opt-in.
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


def build_conversation_judge(llm: LLMProvider, agent_spec: AgentSpec) -> ConversationJudge:
    """Compose the bundled whole-conversation metrics with a per-rule judge for ``agent_spec``."""
    chatbot_role = agent_spec.purpose or agent_spec.system_prompt
    return ConversationJudge(
        chatbot_role,
        [
            RoleAdherenceJudge(llm),
            KnowledgeRetentionJudge(llm),
            ConversationCompletenessJudge(llm),
            TurnRelevancyJudge(llm),
            ConversationRuleJudge(llm, agent_spec),
        ],
    )
