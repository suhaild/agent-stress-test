"""Phase C3 — metric-score -> pass/fail + severity calibration.

Scope: this covers exactly the six Phase-C metric judges that have no
human-configured severity to fall back on — ``ToolArgumentJudge``,
``TaskCompletionJudge``, ``RoleAdherenceJudge``, ``KnowledgeRetentionJudge``,
``ConversationCompletenessJudge``, ``TurnRelevancyJudge``. It deliberately
does NOT touch ``LLMJudge`` (tier 2) or ``ConversationRuleJudge`` — both are
already keyed to a real ``AgentSpec`` ``Rule``, so they correctly use
``rule.severity`` (config, set by a human) instead of a guess; only the six
metrics above previously shared ``judge.py``'s old flat
``_METRIC_SEVERITY = "major"`` placeholder with no severity signal at all.

Every one of those six ultimately produces one continuous DeepEval
``metric.score`` in ``[0, 1]`` — a ratio (fraction of satisfied checks, e.g.
``ArgumentCorrectnessMetric``/``RoleAdherenceMetric``/etc.) or a direct
LLM-assigned float (``TaskCompletionMetric``) — scored against the same kind
of pass/fail ``threshold``. Since they all land in the same space, ONE
score -> severity policy can serve every one of them.

``METRIC_PASS_THRESHOLD`` and the severity bands below were picked by running
``calibrate()`` against a small hand-labeled set built from
``ArgumentCorrectnessMetric``/``TaskCompletionMetric`` (the two easiest to
script exact scores for) spanning clean passes, a borderline half-wrong
tool call, and clear failures at several severities — see
``tests/test_calibration.py``, which re-derives the same scores from
scripted responses and re-runs ``calibrate()`` itself, so it's the source of
truth for *why* these specific numbers, not just what they are. This is a
STARTING calibration off 9 labeled cases, not a rigorous statistical fit —
refine as real runs accumulate more labeled failures.
"""

from dataclasses import dataclass

from agent_stress_test.models import Severity

# Picked via calibrate() against the hand-labeled set (see
# tests/test_calibration.py): every candidate in [0.55, 0.65] ties at
# F1=1.0 (perfectly separates the 5 hand-labeled failures from the 4
# hand-labeled passes); ties break toward the LOWER/more-sensitive
# threshold, since in this domain a missed real failure costs more than a
# false alarm. Notably higher than DeepEval's own bare 0.5 default: the
# labeled set includes a tool call with one right/one wrong argument
# (score=0.5) that a bare 0.5 threshold would let through as a "pass".
METRIC_PASS_THRESHOLD = 0.55

# Severity bands for a metric verdict, keyed by how far its score falls
# below `threshold` ("gap"). A passing verdict's gap is clamped to 0, so it
# always lands in the least-severe band — there's no real failure to grade.
# Bounds picked from the labeled set's hand-assigned severities: the
# smallest real-failure gap observed (0.05 — one right/one wrong tool-call
# argument) was labeled "minor"; the largest (0.55 — a completely wrong tool
# call, or a task left entirely unaddressed) was labeled "critical"; the one
# in between (0.2 — a task left meaningfully, but not entirely, incomplete)
# was "major".
_MINOR_MAX_GAP = 0.15
_MAJOR_MAX_GAP = 0.4


def severity_from_score(score: float, *, threshold: float = METRIC_PASS_THRESHOLD) -> Severity:
    """Bucket a metric's continuous score into a severity: the further below
    ``threshold`` it falls, the more severe — a score that barely missed the
    bar is a minor slip, one near zero is critical. A passing score (or one
    exactly at the threshold) always gets "minor" (no real failure to grade)."""
    gap = max(0.0, threshold - score)
    if gap < _MINOR_MAX_GAP:
        return "minor"
    if gap < _MAJOR_MAX_GAP:
        return "major"
    return "critical"


@dataclass(frozen=True)
class LabeledCase:
    """One hand-labeled (probe/reply, expected verdict) example, reduced to
    the ``metric.score`` a real DeepEval metric actually produced for it
    (see ``tests/test_calibration.py``, which derives these scores by
    running the real metric classes against scripted responses — not
    hand-typed numbers standing in for them)."""

    description: str
    score: float
    expected_pass: bool
    expected_severity: Severity | None  # None when expected_pass is True


@dataclass(frozen=True)
class CalibrationResult:
    """One candidate threshold's precision/recall/F1 at detecting real
    failures (``expected_pass=False``) across a labeled set."""

    threshold: float
    precision: float
    recall: float
    f1: float


_DEFAULT_CANDIDATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def calibrate(
    cases: list[LabeledCase], *, candidates: list[float] | None = None
) -> CalibrationResult:
    """Sweep candidate thresholds against a labeled set and return the one
    with the best F1 at detecting real failures. Ties break toward the
    LOWER/more-sensitive candidate (see ``METRIC_PASS_THRESHOLD``'s
    docstring for why)."""
    best: CalibrationResult | None = None
    for threshold in candidates if candidates is not None else _DEFAULT_CANDIDATES:
        result = _score_threshold(cases, threshold)
        if (
            best is None
            or result.f1 > best.f1
            or (result.f1 == best.f1 and threshold < best.threshold)
        ):
            best = result
    return best


def _score_threshold(cases: list[LabeledCase], threshold: float) -> CalibrationResult:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for case in cases:
        predicted_fail = case.score < threshold
        actual_fail = not case.expected_pass
        if predicted_fail and actual_fail:
            true_positives += 1
        elif predicted_fail and not actual_fail:
            false_positives += 1
        elif not predicted_fail and actual_fail:
            false_negatives += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return CalibrationResult(threshold=threshold, precision=precision, recall=recall, f1=f1)
