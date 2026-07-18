"""Phase C3, part 2 — metric-score -> pass/fail + severity calibration.

Fully offline: every score in the hand-labeled set is derived by running the
REAL ``ArgumentCorrectnessMetric``/``TaskCompletionMetric`` classes against a
scripted ``FakeLLMProvider`` (not hand-typed numbers standing in for them),
so this is grounded in genuine DeepEval scoring math, not just calibration.py's
own arithmetic. See ``reasoning/calibration.py``'s module docstring for why
these two metrics (and not the other four Phase-C-calibrated ones) were
picked to build the labeled set — they're the two easiest to script an exact
score for.
"""

import json

import pytest

from agent_stress_test.models import AgentResponse, ToolCall
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.calibration import (
    METRIC_PASS_THRESHOLD,
    LabeledCase,
    calibrate,
    severity_from_score,
)
from agent_stress_test.reasoning.deepeval_bridge import LLMProviderAsDeepEvalLLM
from agent_stress_test.reasoning.judge import TaskCompletionJudge, ToolArgumentJudge
from deepeval.metrics import ArgumentCorrectnessMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_case import ToolCall as DeepEvalToolCall


def _arg_correctness_score(verdicts: list[str]) -> float:
    responses = [
        json.dumps({"verdicts": [{"verdict": v, "reason": "r"} for v in verdicts]}),
        json.dumps({"reason": "r"}),
    ]
    provider = FakeLLMProvider(responses=responses)
    metric = ArgumentCorrectnessMetric(model=LLMProviderAsDeepEvalLLM(provider), async_mode=False)
    tc = LLMTestCase(
        input="q", actual_output="a", tools_called=[DeepEvalToolCall(name="t", input_parameters={})]
    )
    metric.measure(tc, _show_indicator=False)
    return metric.score


def _task_completion_score(verdict: float) -> float:
    responses = [
        json.dumps({"task": "resolve the request", "outcome": "outcome"}),
        json.dumps({"verdict": verdict, "reason": "r"}),
    ]
    provider = FakeLLMProvider(responses=responses)
    metric = TaskCompletionMetric(model=LLMProviderAsDeepEvalLLM(provider), async_mode=False)
    tc = LLMTestCase(input="q", actual_output="a")
    metric.measure(tc, _show_indicator=False)
    return metric.score


def _labeled_cases() -> list[LabeledCase]:
    """9 hand-labeled (probe/reply, expected verdict) examples, spanning
    clean passes, one borderline half-wrong tool call, and clear failures at
    each severity — real scores from the metrics named in each case."""
    return [
        # ArgumentCorrectness: clean tool calls.
        LabeledCase(
            "single correct tool call",
            _arg_correctness_score(["yes"]),
            expected_pass=True,
            expected_severity=None,
        ),
        LabeledCase(
            "two correct tool calls",
            _arg_correctness_score(["yes", "yes"]),
            expected_pass=True,
            expected_severity=None,
        ),
        # ArgumentCorrectness: the borderline case that motivates a
        # threshold ABOVE DeepEval's bare 0.5 default.
        LabeledCase(
            "one right, one wrong tool-call argument",
            _arg_correctness_score(["yes", "no"]),
            expected_pass=False,
            expected_severity="minor",
        ),
        LabeledCase(
            "single wrong tool call",
            _arg_correctness_score(["no"]),
            expected_pass=False,
            expected_severity="critical",
        ),
        LabeledCase(
            "both tool calls wrong",
            _arg_correctness_score(["no", "no"]),
            expected_pass=False,
            expected_severity="critical",
        ),
        # TaskCompletion: a clean, a mostly-good, a partial, and a total miss.
        LabeledCase(
            "task fully completed",
            _task_completion_score(1.0),
            expected_pass=True,
            expected_severity=None,
        ),
        LabeledCase(
            "task mostly completed",
            _task_completion_score(0.65),
            expected_pass=True,
            expected_severity=None,
        ),
        LabeledCase(
            "task meaningfully incomplete",
            _task_completion_score(0.35),
            expected_pass=False,
            expected_severity="major",
        ),
        LabeledCase(
            "task entirely unaddressed",
            _task_completion_score(0.0),
            expected_pass=False,
            expected_severity="critical",
        ),
    ]


# --- calibrate(): the chosen threshold really is the best F1 on this set --


def test_calibrate_picks_the_documented_threshold():
    result = calibrate(_labeled_cases())

    assert result.threshold == METRIC_PASS_THRESHOLD
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_calibrate_ties_break_toward_the_lower_threshold():
    # 0.55 through 0.65 all separate the labeled set perfectly (see the
    # module docstring's sweep) -- confirm the tie-break rule, not just that
    # SOME threshold in the plateau was picked.
    result = calibrate(_labeled_cases(), candidates=[0.6, 0.65, 0.55])
    assert result.threshold == 0.55


def test_a_bare_default_threshold_of_0_5_would_have_missed_the_borderline_case():
    # Documents WHY calibration raised the bar above DeepEval's own bare 0.5
    # default: at 0.5, the half-wrong tool call (score 0.5) is NOT caught.
    result = calibrate(_labeled_cases(), candidates=[0.5])
    assert result.recall < 1.0


# --- severity_from_score(): matches every failing case's hand-assigned label


@pytest.mark.parametrize(
    "case", [c for c in _labeled_cases() if not c.expected_pass], ids=lambda c: c.description
)
def test_severity_from_score_matches_the_labeled_severity(case):
    assert (
        severity_from_score(case.score, threshold=METRIC_PASS_THRESHOLD) == case.expected_severity
    )


@pytest.mark.parametrize(
    "case", [c for c in _labeled_cases() if c.expected_pass], ids=lambda c: c.description
)
def test_severity_from_score_never_escalates_a_passing_case_past_minor(case):
    assert severity_from_score(case.score, threshold=METRIC_PASS_THRESHOLD) == "minor"


def test_severity_from_score_boundaries():
    # gap = threshold - score; bands are gap<0.15 -> minor, <0.4 -> major, else critical.
    assert severity_from_score(0.55, threshold=0.55) == "minor"  # gap 0.0
    # 0.15 sits exactly on the minor/major boundary -- confirm it's NOT minor.
    assert severity_from_score(0.40, threshold=0.55) == "major"  # gap 0.15
    assert severity_from_score(0.20, threshold=0.55) == "major"  # gap 0.35
    assert severity_from_score(0.10, threshold=0.55) == "critical"  # gap 0.45


# --- Integration: the real wired judges reproduce these verdicts, offline -


def test_tool_argument_judge_matches_the_calibrated_labels_end_to_end():
    a4_response = AgentResponse(
        final_reply="ok", tool_calls=[ToolCall(id="1", name="t", input_parameters={})]
    )

    provider = FakeLLMProvider(
        responses=[
            json.dumps({"verdicts": [{"verdict": "no", "reason": "wrong id"}]}),
            json.dumps({"reason": "wrong id"}),
        ]
    )
    [verdict] = ToolArgumentJudge(provider).judge(a4_response, run_id="r", node_id="n")

    assert verdict.passed is False
    assert verdict.severity == "critical"  # a single wrong tool call, per the labeled set


def test_task_completion_judge_matches_the_calibrated_labels_end_to_end():
    provider = FakeLLMProvider(
        responses=[
            json.dumps({"task": "resolve the request", "outcome": "partially resolved"}),
            json.dumps({"verdict": 0.35, "reason": "left something out"}),
        ]
    )
    [verdict] = TaskCompletionJudge(provider).judge(
        AgentResponse(final_reply="partial reply"), run_id="r", node_id="n"
    )

    assert verdict.passed is False
    assert verdict.severity == "major"  # meaningfully incomplete, per the labeled set
