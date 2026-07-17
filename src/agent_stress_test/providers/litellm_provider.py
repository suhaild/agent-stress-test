"""litellm-backed LLMProvider (Claude + OpenAI via one call).

This is the only module in the codebase allowed to import litellm.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import litellm

from agent_stress_test.models import Message, ToolCall
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
        super().__init__()
        self._model = model
        self._default_kwargs = default_kwargs

    def _record_usage(self, response: object) -> None:
        """Side effect only — capture ``response.usage`` and
        ``litellm.completion_cost(response)`` into ``self.meter``. Doesn't
        touch what ``complete()``/``complete_with_tools()`` return.

        A stale/unrecognized model id makes ``completion_cost`` raise —
        litellm's pricing table is a static, not-always-current snapshot, and
        this must never crash a run over that; the call is recorded as
        "pricing unavailable" (cost stays 0.0) instead.
        """
        usage = getattr(response, "usage", None)
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = None
        self.meter.record(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            cost_usd=cost,
        )

    def complete(self, messages: list[Message]) -> str:
        response = litellm.completion(
            model=self._model,
            messages=[_to_litellm_message(m) for m in messages],
            **self._default_kwargs,
        )
        self._record_usage(response)
        return response.choices[0].message.content

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        # The n samples are identical, independent requests (same prompt, no
        # shared state) — run them concurrently instead of one after another.
        workers = min(n, _MAX_SAMPLE_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda _: self.complete(messages), range(n)))

    def complete_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> tuple[str, list[ToolCall]]:
        """Like ``complete``, but with native tool-calling: returns the
        assistant's text (``""`` if it only emitted tool calls) plus any
        ``ToolCall``s litellm's unified response reports — used by
        ``ProviderAgent`` (unlike ``SampleAgent``, which narrates tool use as
        parsed plain text instead)."""
        response = litellm.completion(
            model=self._model,
            messages=[_to_litellm_message(m) for m in messages],
            tools=tools,
            **self._default_kwargs,
        )
        self._record_usage(response)
        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input_parameters=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (message.tool_calls or [])
        ]
        return message.content or "", tool_calls
