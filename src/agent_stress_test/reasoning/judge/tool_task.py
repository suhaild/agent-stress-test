"""Phase C: node-level tool/task metric judges (DeepEval's
``ArgumentCorrectnessMetric``/``TaskCompletionMetric``), plus ``CompositeJudge``
to run several node-level judges over the same node and concatenate verdicts.
"""

from deepeval.metrics import ArgumentCorrectnessMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from pydantic import ValidationError

from agent_stress_test.models import AgentResponse, Message, Verdict
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.calibration import METRIC_PASS_THRESHOLD, severity_from_score
from agent_stress_test.reasoning.deepeval_bridge import LLMProviderAsDeepEvalLLM
from agent_stress_test.reasoning.deepeval_simulator import to_deepeval_tool_call
from agent_stress_test.reasoning.judge.base import Judge, _confidence_from_score


def _last_user_input(conversation: list[Message] | None) -> str:
    """The most recent user-message text in ``conversation`` (the request a
    metric judges the reply/tool-calls against), or ``""`` if none is
    available — DeepEval's node metrics accept an empty ``input`` (it only
    degrades judgment quality, never crashes; see judge tests)."""
    if not conversation:
        return ""
    for message in reversed(conversation):
        if message.role == "user" and isinstance(message.content, str):
            return message.content
    return ""


def _metric_test_case(response: AgentResponse, conversation: list[Message] | None) -> LLMTestCase:
    return LLMTestCase(
        input=_last_user_input(conversation),
        actual_output=response.final_reply,
        tools_called=[to_deepeval_tool_call(call) for call in response.tool_calls],
    )


def _measure_node_metric(
    metric, test_case: LLMTestCase, *, run_id: str, node_id: str, scope: str, label: str
) -> Verdict:
    """Shared measure/verdict-building step both node-level metric judges
    below use — same conservative-pass-on-malformed-output contract as the
    conversation-level metrics (``_measure_conversation_metric``)."""
    try:
        metric.measure(test_case, _show_indicator=False)
        passed, reason, confidence, severity = (
            metric.success,
            metric.reason,
            _confidence_from_score(metric),
            severity_from_score(metric.score, threshold=metric.threshold),
        )
    except ValidationError:
        passed, reason, confidence, severity = (
            True,
            f"{label} metric returned malformed output; defaulting to pass.",
            0.0,
            "minor",
        )
    return Verdict(
        run_id=run_id,
        node_id=node_id,
        passed=passed,
        rule_id=None,
        reason=reason,
        tier="llm",
        confidence=confidence,
        severity=severity,
        scope=scope,
    )


class ToolArgumentJudge(Judge):
    """Node-level judge: DeepEval's ``ArgumentCorrectnessMetric`` over the
    node's structured tool calls (Phase C1).

    Emits a single ``scope="tool"`` verdict (rendered inline with the node's
    tool-call block, not as a generic rule verdict), and only for nodes that
    actually made tool calls — a node with none has nothing to judge, so it
    gets no verdict rather than a vacuous pass. ``NO expected_tools`` is used
    (that needs a known-good tool set — Phase E's regression replay); this
    judges whether the arguments the agent chose fit the request it was
    given.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._model = LLMProviderAsDeepEvalLLM(llm)

    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]:
        if not response.tool_calls:
            return []
        metric = ArgumentCorrectnessMetric(
            model=self._model, threshold=METRIC_PASS_THRESHOLD, async_mode=False
        )
        return [
            _measure_node_metric(
                metric,
                _metric_test_case(response, conversation),
                run_id=run_id,
                node_id=node_id,
                scope="tool",
                label="Tool-argument",
            )
        ]


class TaskCompletionJudge(Judge):
    """Node-level judge: DeepEval's referenceless ``TaskCompletionMetric``
    (Phase C1) — did the reply actually accomplish what the user asked?

    Emits one ``scope="task"`` verdict per node. Referenceless (no
    ``expected_output``), which fits open-ended adversarial exploration where
    there's no gold answer to compare against. Off by default in the live run
    (see ``build_runner``): it costs 2 LLM calls per node regardless of tool
    use — confirmed cheap in isolation (~$0.002/node on Haiku) by Phase C3's
    real measurement, but still the only per-node metric with no cheap
    early-out, so it stays opt-in rather than compounding with every node
    a run creates.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._model = LLMProviderAsDeepEvalLLM(llm)

    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]:
        metric = TaskCompletionMetric(
            model=self._model, threshold=METRIC_PASS_THRESHOLD, async_mode=False
        )
        return [
            _measure_node_metric(
                metric,
                _metric_test_case(response, conversation),
                run_id=run_id,
                node_id=node_id,
                scope="task",
                label="Task-completion",
            )
        ]


class CompositeJudge(Judge):
    """Runs several judges over the same node and concatenates their verdicts.

    Composition, not tiering: unlike ``TwoTierJudge`` (whose tier 2
    short-circuits on a tier-1 failure), every judge here always runs — the
    rule verdict, the tool-argument verdict, and the task verdict are
    independent axes, all worth reporting on the same node.
    """

    def __init__(self, judges: list[Judge]) -> None:
        self._judges = judges

    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for judge in self._judges:
            verdicts.extend(
                judge.judge(response, run_id=run_id, node_id=node_id, conversation=conversation)
            )
        return verdicts
