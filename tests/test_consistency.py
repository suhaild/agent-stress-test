import pytest

from agent_stress_test.models import Message
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.consistency import ConsistencyScorer, instability_score

DISTINCT = [
    "Your order shipped on Monday.",
    "We have no record of that.",
    "A refund needs manager approval.",
    "That item is permanently discontinued.",
]


def msgs() -> list[Message]:
    return [Message(role="user", content="Where is my order?")]


# --- Identical -> stable, varied -> unstable -----------------------------


def test_identical_samples_are_stable():
    assert instability_score(["same", "same", "same"]) == 0.0


def test_scorer_identical_echoes_are_stable():
    # Default fake mode echoes the same reply every call -> all samples agree.
    score = ConsistencyScorer(FakeLLMProvider()).score(msgs(), 4)
    assert score == 0.0


def test_varied_samples_are_unstable():
    assert instability_score(DISTINCT) > 0.5


def test_scorer_varied_is_more_unstable_than_identical():
    varied = ConsistencyScorer(FakeLLMProvider(responses=DISTINCT)).score(msgs(), 4)
    identical = ConsistencyScorer(FakeLLMProvider()).score(msgs(), 4)
    assert varied > identical
    assert varied > 0.5


# --- Bounded 0-1 ---------------------------------------------------------


@pytest.mark.parametrize(
    "samples",
    [
        ["same", "same"],
        DISTINCT,
        ["", "hi"],
        ["a", "a", "b"],
        ["The order shipped.", "the order  shipped"],
    ],
)
def test_score_is_bounded(samples):
    score = instability_score(samples)
    assert 0.0 <= score <= 1.0


# --- Deterministic given fixed inputs ------------------------------------


def test_pure_score_is_deterministic():
    assert instability_score(DISTINCT) == instability_score(DISTINCT)


def test_scorer_is_deterministic_across_fresh_providers():
    first = ConsistencyScorer(FakeLLMProvider(responses=list(DISTINCT))).score(msgs(), 4)
    second = ConsistencyScorer(FakeLLMProvider(responses=list(DISTINCT))).score(msgs(), 4)
    assert first == second


# --- Smooth / monotonic (variance, not just exact match) -----------------


def test_partial_disagreement_is_between_extremes():
    identical = instability_score(["the order shipped today", "the order shipped today"])
    one_word = instability_score(["the order shipped today", "the order shipped tomorrow"])
    fully_distinct = instability_score(["the order shipped today", "completely different text"])
    assert identical == 0.0
    assert identical < one_word < fully_distinct


# --- Normalization -------------------------------------------------------


def test_normalization_ignores_case_and_whitespace():
    assert instability_score(["Hello  world", "hello world"]) == 0.0


# --- Edge cases ----------------------------------------------------------


def test_single_sample_is_stable():
    assert instability_score(["only"]) == 0.0
    assert ConsistencyScorer(FakeLLMProvider()).score(msgs(), 1) == 0.0


def test_all_empty_replies_are_stable():
    assert instability_score(["", "", ""]) == 0.0
    assert ConsistencyScorer(FakeLLMProvider(responses=["", ""])).score(msgs(), 2) == 0.0


def test_mixed_empty_and_nonempty_is_unstable_but_bounded():
    score = instability_score(["", "hello"])
    assert 0.0 < score <= 1.0


# --- Contract ------------------------------------------------------------


def test_score_rejects_non_positive_n():
    with pytest.raises(ValueError):
        ConsistencyScorer(FakeLLMProvider()).score(msgs(), 0)


def test_scorer_samples_exactly_n_times():
    provider = FakeLLMProvider()
    ConsistencyScorer(provider).score(msgs(), 5)
    assert len(provider.calls) == 5  # sample_n calls complete() once per sample
