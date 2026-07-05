"""litellm-backed LLMProvider (Claude + OpenAI via one call).

This is the only module in the codebase allowed to import litellm.
"""

import litellm

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider


class LiteLLMProvider(LLMProvider):
    """Thin wrapper over litellm exposing the LLMProvider contract."""

    def __init__(self, model: str, **default_kwargs: object) -> None:
        self._model = model
        self._default_kwargs = default_kwargs

    def complete(self, messages: list[Message]) -> str:
        response = litellm.completion(
            model=self._model,
            messages=[m.model_dump() for m in messages],
            **self._default_kwargs,
        )
        return response.choices[0].message.content

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return [self.complete(messages) for _ in range(n)]
