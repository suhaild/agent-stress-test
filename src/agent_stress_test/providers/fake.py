"""Deterministic fake LLMProvider (for tests)."""

import threading

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider


class FakeLLMProvider(LLMProvider):
    """Deterministic, no-network LLMProvider for tests.

    With no `responses`, replies are a pure function of the last message's
    content. With `responses`, they're returned in order; `cycle=True` wraps
    around instead of raising once exhausted.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        cycle: bool = False,
        default_reply_prefix: str = "fake-reply: ",
    ) -> None:
        super().__init__()
        self._responses = list(responses) if responses is not None else None
        self._cycle = cycle
        self._default_reply_prefix = default_reply_prefix
        self._next_index = 0
        self.calls: list[list[Message]] = []
        # Guards state — callers run tactic branches concurrently.
        self._lock = threading.Lock()

    def complete(self, messages: list[Message]) -> str:
        with self._lock:
            self.calls.append(list(messages))
            if self._responses is not None:
                if self._next_index >= len(self._responses):
                    if not self._cycle:
                        raise IndexError("FakeLLMProvider: scripted responses exhausted")
                    self._next_index = 0
                reply = self._responses[self._next_index]
                self._next_index += 1
            else:
                last_content = messages[-1].content if messages else ""
                reply = f"{self._default_reply_prefix}{last_content}"
        # No real call was made: word count stands in for token count, cost is 0.0.
        prompt_tokens = sum(len(m.content.split()) for m in messages if isinstance(m.content, str))
        completion_tokens = len(reply.split())
        self.meter.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,
        )
        return reply

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return [self.complete(messages) for _ in range(n)]
