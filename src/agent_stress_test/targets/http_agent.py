"""Bring-your-own: wrap an HTTP endpoint as a TargetAgent.

This is the only module in the codebase allowed to import httpx.
"""

import httpx

from agent_stress_test.models import AgentResponse, Message, Step
from agent_stress_test.ports import TargetAgent


class HttpAgent(TargetAgent):
    """Wraps an HTTP/JSON endpoint as a TargetAgent.

    Sends `{"messages": [...]}` and expects `{"reply": "..."}` back, with an
    optional `"trace": [...]` list of step objects. A missing or null `trace`
    is returned as `trace=None` — never fabricated.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._headers = headers

    def respond(self, conversation: list[Message]) -> AgentResponse:
        payload = {"messages": [m.model_dump() for m in conversation]}
        response = httpx.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        trace_data = body.get("trace")
        trace = [Step(**step) for step in trace_data] if trace_data else None
        return AgentResponse(final_reply=body["reply"], trace=trace)
