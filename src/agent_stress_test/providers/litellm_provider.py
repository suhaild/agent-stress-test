"""litellm-backed LLMProvider (Claude + OpenAI via one call).

The only module allowed to import litellm; every litellm exception is
translated to ``ports.ProviderError`` (see ``_classify_error``) before it
crosses back out.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import litellm

from agent_stress_test.models import Message, ToolCall
from agent_stress_test.ports import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

# Caps sample_n concurrency so a large n doesn't fan out an unbounded burst.
_MAX_SAMPLE_WORKERS = 5

# litellm's own default (no timeout) is 6000s, indistinguishable from a hang;
# override via **default_kwargs if a setup is legitimately slower than this.
_DEFAULT_TIMEOUT_SECONDS = 60

# litellm.exceptions subclasses openai's hierarchy per provider, so mapping by
# type (not provider/status code) stays stable across whichever backend raised it.
_AUTH_ERRORS = (litellm.exceptions.AuthenticationError, litellm.exceptions.PermissionDeniedError)
_RATE_LIMIT_ERRORS = (litellm.exceptions.RateLimitError,)
_TIMEOUT_ERRORS = (litellm.exceptions.Timeout,)
_CONNECTION_ERRORS = (
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.BadGatewayError,
)


def _classify_error(model: str, exc: Exception) -> ProviderError:
    if isinstance(exc, _AUTH_ERRORS):
        return ProviderAuthError(
            f"Authentication failed for model '{model}' — check the provider API key in your .env."
        )
    if isinstance(exc, _RATE_LIMIT_ERRORS):
        return ProviderRateLimitError(
            f"Rate limited by the provider for model '{model}' — retry after a pause, "
            "or reduce concurrency (e.g. --sample-n)."
        )
    if isinstance(exc, _TIMEOUT_ERRORS):
        return ProviderTimeoutError(
            f"Request to model '{model}' timed out after {_DEFAULT_TIMEOUT_SECONDS}s — check "
            "the model/endpoint is reachable (e.g. a local model server actually running), or "
            "pass a longer `timeout` explicitly if it's just genuinely slow."
        )
    if isinstance(exc, _CONNECTION_ERRORS):
        return ProviderConnectionError(
            f"Could not reach the provider for model '{model}' — check network connectivity "
            "and the provider's status."
        )
    return ProviderError(f"Provider error for model '{model}': {exc}")


def _to_litellm_message(message: Message) -> dict:
    """Adds `cache_control` (Anthropic prompt-caching) only for messages flagged as cache breakpoints."""
    data = message.model_dump(exclude={"cache"})
    if message.cache:
        data["cache_control"] = {"type": "ephemeral"}
    return data


class LiteLLMProvider(LLMProvider):
    """Thin wrapper over litellm exposing the LLMProvider contract."""

    def __init__(self, model: str, **default_kwargs: object) -> None:
        super().__init__()
        self._model = model
        self._default_kwargs = {"timeout": _DEFAULT_TIMEOUT_SECONDS, **default_kwargs}

    def _record_usage(self, response: object) -> None:
        # completion_cost raises for a stale/unrecognized model id (litellm's
        # pricing table snapshot); treat that as cost-unavailable, not a crash.
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
        try:
            response = litellm.completion(
                model=self._model,
                messages=[_to_litellm_message(m) for m in messages],
                **self._default_kwargs,
            )
        except Exception as exc:
            # Broad on purpose: litellm.exceptions.* has no single common base.
            raise _classify_error(self._model, exc) from exc
        self._record_usage(response)
        return response.choices[0].message.content

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        workers = min(n, _MAX_SAMPLE_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda _: self.complete(messages), range(n)))

    def complete_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> tuple[str, list[ToolCall]]:
        """Like `complete`, but with native tool-calling; text is `""` if the model only emitted tool calls."""
        try:
            response = litellm.completion(
                model=self._model,
                messages=[_to_litellm_message(m) for m in messages],
                tools=tools,
                **self._default_kwargs,
            )
        except Exception as exc:
            # Broad on purpose: litellm.exceptions.* has no single common base.
            raise _classify_error(self._model, exc) from exc
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
