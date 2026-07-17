"""Bring-your-own: wrap a Python callable as a TargetAgent."""

from typing import Callable

from agent_stress_test.models import AgentResponse, Capabilities, Message
from agent_stress_test.ports import TargetAgent


class PythonFunctionAgent(TargetAgent):
    """Wraps any `Callable[[list[Message]], str | AgentResponse]` as a TargetAgent.

    A callable that only knows how to return a final reply can keep returning a
    plain `str` — it's wrapped as `AgentResponse(final_reply=..., trace=None)`.
    A callable that already tracks its own reasoning steps can return a full
    `AgentResponse` and its trace is passed through unchanged.

    ``capabilities`` lets a caller who knows what the wrapped callable
    actually does declare it accurately (e.g. one that emits real
    ``ToolCall``s can pass ``Capabilities(tools=True)``) — this adapter can't
    infer it from an arbitrary function, so it defaults to claiming nothing.
    """

    def __init__(
        self,
        fn: Callable[[list[Message]], str | AgentResponse],
        *,
        capabilities: Capabilities | None = None,
    ) -> None:
        self._fn = fn
        self._capabilities = capabilities if capabilities is not None else Capabilities()

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def respond(self, conversation: list[Message]) -> AgentResponse:
        result = self._fn(list(conversation))
        if isinstance(result, AgentResponse):
            return result
        return AgentResponse(final_reply=result, trace=None)
