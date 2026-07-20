"""Self-consistency scorer.

Samples the target agent N times for the same conversation and measures the
mean pairwise dissimilarity (via stdlib difflib) as a 0-1 instability score.
"""

import re
from difflib import SequenceMatcher

from agent_stress_test.models import Message
from agent_stress_test.ports import TargetAgent

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting isn't disagreement."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def _similarity(a: str, b: str) -> float:
    """Similarity of two strings in [0, 1] (1.0 = identical, incl. two empties)."""
    return SequenceMatcher(None, a, b).ratio()


def instability_score(samples: list[str]) -> float:
    """Mean pairwise dissimilarity of the samples, in [0, 1]; 0.0 if fewer than two."""
    normalized = [_normalize(sample) for sample in samples]
    n = len(normalized)
    if n < 2:
        return 0.0

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - _similarity(normalized[i], normalized[j])
            pairs += 1
    return total / pairs


class ConsistencyScorer:
    """Samples the target N times and scores how much the samples disagree.

    Calls the target itself rather than a side-channel LLM completion, so it
    measures the real target's variance regardless of what backs it.
    """

    def __init__(self, target: TargetAgent) -> None:
        self._target = target

    def score(self, messages: list[Message], n: int) -> float:
        if n < 1:
            raise ValueError("n must be >= 1")
        samples = [self._target.respond(messages).final_reply for _ in range(n)]
        return instability_score(samples)
