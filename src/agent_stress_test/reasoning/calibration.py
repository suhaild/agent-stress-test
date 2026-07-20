"""Score -> pass/fail + severity calibration for the DeepEval metric judges
that have no human-configured severity to fall back on (``LLMJudge`` and
``ConversationRuleJudge`` use ``rule.severity`` instead and don't need this).
"""

from dataclasses import dataclass

from agent_stress_test.models import Severity

# Picked via calibrate() against the hand-labeled set in tests/test_calibration.py;
# ties break toward the lower/more-sensitive threshold since a missed real
# failure costs more than a false alarm.
METRIC_PASS_THRESHOLD = 0.55

# Bands keyed by how far a score falls below `threshold` ("gap"); a passing
# verdict's gap is clamped to 0. Bounds come from the labeled set's
# hand-assigned severities (see tests/test_calibration.py).
_MINOR_MAX_GAP = 0.15
_MAJOR_MAX_GAP = 0.4


def severity_from_score(score: float, *, threshold: float = METRIC_PASS_THRESHOLD) -> Severity:
    """The further a score falls below `threshold`, the more severe; a passing score is always "minor"."""
    gap = max(0.0, threshold - score)
    if gap < _MINOR_MAX_GAP:
        return "minor"
    if gap < _MAJOR_MAX_GAP:
        return "major"
    return "critical"


@dataclass(frozen=True)
class LabeledCase:
    """A hand-labeled example, reduced to the score a real DeepEval metric produced for it."""

    description: str
    score: float
    expected_pass: bool
    expected_severity: Severity | None  # None when expected_pass is True


@dataclass(frozen=True)
class CalibrationResult:
    """One candidate threshold's precision/recall/F1 at detecting real failures."""

    threshold: float
    precision: float
    recall: float
    f1: float


_DEFAULT_CANDIDATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def calibrate(
    cases: list[LabeledCase], *, candidates: list[float] | None = None
) -> CalibrationResult:
    """Sweeps candidate thresholds and returns the one with best F1 at
    detecting real failures; ties break toward the lower candidate."""
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
