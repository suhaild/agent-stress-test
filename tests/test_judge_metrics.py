"""Phase C1 — node-level tool/task metric judges.

Offline throughout: the plain ``ShapedFakeLLM`` fabricates a *pass* for both
metrics (so wiring them on never spuriously fails an offline run), and a
scripted ``FakeLLMProvider`` returning the exact per-metric JSON forces the
failure paths — the same scripting pattern the GEval judge tests (B3) use.
"""

import json

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentResponse, Message
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.judge import (
    CompositeJudge,
    Judge,
    TaskCompletionJudge,
    ToolArgumentJudge,
    build_two_tier_judge,
)
from agent_stress_test.targets.tool_calling_verification_agent import (
    tool_calling_verification_agent,
)

_A4_QUERY = [Message(role="user", content="Where is order 12345?")]


def _a4_response() -> AgentResponse:
    """The A4 verification target's real output: a lookup_order call with a
    deterministically WRONG order id (see tool_calling_verification_agent)."""
    return tool_calling_verification_agent(_A4_QUERY)


# --- ArgumentCorrectness: the metric's two scripted calls ------------------
# ArgumentCorrectnessMetric.measure() makes exactly two model calls when tool
# calls are present: per-tool verdicts, then the aggregate reason.


def _arg_correctness_responses(*, verdict: str, reason: str) -> list[str]:
    return [
        json.dumps({"verdicts": [{"verdict": verdict, "reason": reason}]}),
        json.dumps({"reason": reason}),
    ]


def test_tool_argument_judge_flags_a_wrong_argument_tool_call():
    provider = FakeLLMProvider(
        responses=_arg_correctness_responses(
            verdict="no", reason="lookup_order used the wrong order_id."
        )
    )
    [verdict] = ToolArgumentJudge(provider).judge(
        _a4_response(), run_id="r", node_id="n", conversation=_A4_QUERY
    )

    assert verdict.scope == "tool"
    assert verdict.rule_id is None
    assert verdict.tier == "llm"
    assert verdict.passed is False
    assert verdict.reason == "lookup_order used the wrong order_id."


def test_tool_argument_judge_passes_a_correct_tool_call():
    provider = FakeLLMProvider(
        responses=_arg_correctness_responses(verdict="yes", reason="arguments correct")
    )
    [verdict] = ToolArgumentJudge(provider).judge(_a4_response(), run_id="r", node_id="n")

    assert verdict.scope == "tool"
    assert verdict.passed is True


def test_tool_argument_judge_emits_nothing_for_a_node_with_no_tool_calls():
    # A node that made no tool call has nothing to judge — no vacuous pass.
    verdicts = ToolArgumentJudge(ShapedFakeLLM()).judge(
        AgentResponse(final_reply="Happy to help."), run_id="r", node_id="n"
    )
    assert verdicts == []


def test_tool_argument_judge_offline_shaped_fake_passes():
    # The default offline fake fabricates an empty verdict list -> score 1.0 ->
    # pass, so ArgumentCorrectness being on by default never spuriously fails
    # an offline run.
    [verdict] = ToolArgumentJudge(ShapedFakeLLM()).judge(_a4_response(), run_id="r", node_id="n")
    assert verdict.scope == "tool"
    assert verdict.passed is True


def test_tool_argument_judge_malformed_output_falls_back_to_pass():
    provider = FakeLLMProvider(responses=["not json at all", "also not json"])
    [verdict] = ToolArgumentJudge(provider).judge(_a4_response(), run_id="r", node_id="n")

    assert verdict.passed is True  # never invents a failure from a parse error
    assert verdict.confidence == 0.0


# --- TaskCompletion: extract task+outcome, then verdict --------------------


def test_task_completion_judge_flags_an_incomplete_task():
    provider = FakeLLMProvider(
        responses=[
            json.dumps({"task": "find the order", "outcome": "could not find it"}),
            json.dumps({"verdict": 0.0, "reason": "the task was not completed"}),
        ]
    )
    [verdict] = TaskCompletionJudge(provider).judge(
        AgentResponse(final_reply="I couldn't find anything."),
        run_id="r",
        node_id="n",
        conversation=[Message(role="user", content="find my order")],
    )

    assert verdict.scope == "task"
    assert verdict.rule_id is None
    assert verdict.passed is False


def test_task_completion_judge_offline_shaped_fake_passes():
    [verdict] = TaskCompletionJudge(ShapedFakeLLM()).judge(
        AgentResponse(final_reply="Done."), run_id="r", node_id="n"
    )
    assert verdict.scope == "task"
    assert verdict.passed is True


# --- CompositeJudge: independent axes, no short-circuit --------------------


def test_composite_judge_runs_every_judge_and_unions_their_verdicts(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    composite = CompositeJudge(
        [build_two_tier_judge(spec, ShapedFakeLLM()), ToolArgumentJudge(ShapedFakeLLM())]
    )

    verdicts = composite.judge(_a4_response(), run_id="r", node_id="n", conversation=_A4_QUERY)

    scopes = {v.scope for v in verdicts}
    assert "rule" in scopes  # the rule judge always contributes rule-scoped verdicts
    assert "tool" in scopes  # the tool judge adds its own axis alongside


def test_composite_judge_is_a_judge():
    assert isinstance(CompositeJudge([]), Judge)
