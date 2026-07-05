"""Deterministic fake LLMProvider (for tests)."""

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider


class FakeLLMProvider(LLMProvider):
    """Deterministic, no-network LLMProvider for tests.

    With no `responses` given, replies are a pure function of the last message's
    content (same input on a fresh instance always yields the same output). With
    `responses` given, they're returned in order; `cycle=True` wraps around once
    exhausted instead of raising.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        cycle: bool = False,
        default_reply_prefix: str = "fake-reply: ",
    ) -> None:
        self._responses = list(responses) if responses is not None else None
        self._cycle = cycle
        self._default_reply_prefix = default_reply_prefix
        self._next_index = 0
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        if self._responses is not None:
            if self._next_index >= len(self._responses):
                if not self._cycle:
                    raise IndexError("FakeLLMProvider: scripted responses exhausted")
                self._next_index = 0
            reply = self._responses[self._next_index]
            self._next_index += 1
            return reply
        last_content = messages[-1].content if messages else ""
        return f"{self._default_reply_prefix}{last_content}"

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return [self.complete(messages) for _ in range(n)]
