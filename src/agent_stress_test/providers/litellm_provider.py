"""litellm-backed LLMProvider (Claude + OpenAI via one call).

This is the only module in the codebase allowed to import litellm.
"""

from concurrent.futures import ThreadPoolExecutor

import litellm

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider

# Cap on concurrent in-flight requests for sample_n, independent of n itself,
# so a large --sample-n doesn't fan out an unbounded burst of requests.
_MAX_SAMPLE_WORKERS = 5


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
        # The n samples are identical, independent requests (same prompt, no
        # shared state) — run them concurrently instead of one after another.
        workers = min(n, _MAX_SAMPLE_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda _: self.complete(messages), range(n)))
