"""litellm-backed LLMProvider (Claude + OpenAI via one call).

This is the only module in the codebase allowed to import litellm.
"""

import litellm

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider


def _to_litellm_message(message: Message) -> dict:
    """Translate a Message to litellm's dict shape, adding cache_control
    (Anthropic prompt-caching) only for messages flagged as cache breakpoints.
    Messages with cache=False dump exactly as before, byte-for-byte.
    """
    data = message.model_dump(exclude={"cache"})
    if message.cache:
        data["cache_control"] = {"type": "ephemeral"}
    return data


class LiteLLMProvider(LLMProvider):
    """Thin wrapper over litellm exposing the LLMProvider contract."""

    def __init__(self, model: str, **default_kwargs: object) -> None:
        self._model = model
        self._default_kwargs = default_kwargs

    def complete(self, messages: list[Message]) -> str:
        response = litellm.completion(
            model=self._model,
            messages=[_to_litellm_message(m) for m in messages],
            **self._default_kwargs,
        )
        return response.choices[0].message.content

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return [self.complete(messages) for _ in range(n)]
