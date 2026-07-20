"""litellm-backed LLMProvider (Claude + OpenAI via one call).

This is the only module in the codebase allowed to import litellm — including
its exceptions: every provider-raised error is translated (see
``_classify_error``) into ``ports.ProviderError`` and its subclasses before
crossing back out of this module, so callers never need to import or
pattern-match litellm's own exception types.
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

# Cap on concurrent in-flight requests for sample_n, independent of n itself,
# so a large --sample-n doesn't fan out an unbounded burst of requests.
_MAX_SAMPLE_WORKERS = 5

# litellm's own default (no timeout passed at all) is 6000s -- indistinguishable
# from a genuine hang for a run stuck on an unresponsive endpoint (e.g. a local
# model server that isn't actually running). A real completion, even a slow
# one, finishes well under a minute; callers with a legitimately slower setup
# (a big local model on modest hardware) can still override via the
# constructor's **default_kwargs.
_DEFAULT_TIMEOUT_SECONDS = 60

# litellm.exceptions subclasses openai's hierarchy, one-to-one per provider,
# so this maps by exception *type* rather than by (provider, status code) --
# stable across whichever of Claude/OpenAI/etc. actually raised it.
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
    """Translate a raw litellm/provider-SDK exception into this codebase's
    own ``ProviderError`` hierarchy, with a friendly, human-readable message
    — so a caller (CLI, dashboard) never has to import litellm or pattern-
    match a raw stack trace to know what went wrong or whether retrying
    could help."""
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
        self._default_kwargs = {"timeout": _DEFAULT_TIMEOUT_SECONDS, **default_kwargs}

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
        try:
            response = litellm.completion(
                model=self._model,
                messages=[_to_litellm_message(m) for m in messages],
                **self._default_kwargs,
            )
        except Exception as exc:
            # Scoped to just the litellm.completion() call above — every
            # litellm.exceptions.* class litellm actually raises here (verified:
            # they subclass openai's exception hierarchy, not a single litellm
            # base), so a narrower except clause would miss some of them.
            raise _classify_error(self._model, exc) from exc
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
        try:
            response = litellm.completion(
                model=self._model,
                messages=[_to_litellm_message(m) for m in messages],
                tools=tools,
                **self._default_kwargs,
            )
        except Exception as exc:
            # Scoped to just the litellm.completion() call above — every
            # litellm.exceptions.* class litellm actually raises here (verified:
            # they subclass openai's exception hierarchy, not a single litellm
            # base), so a narrower except clause would miss some of them.
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
