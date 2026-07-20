"""Shared `{"messages": [...]} -> {"reply": ..., "trace": [...]}` JSON codec
for every bring-your-own TargetAgent that talks over a wire (HTTP, stdin/stdout)."""

from agent_stress_test.models import AgentResponse, Message, Step


def _build_wire_payload(conversation: list[Message]) -> dict:
    return {"messages": [m.model_dump(exclude={"cache"}) for m in conversation]}


def _parse_wire_response(body: dict) -> AgentResponse:
    """A missing or null ``trace`` is returned as ``trace=None`` — never fabricated."""
    trace_data = body.get("trace")
    trace = [Step(**step) for step in trace_data] if trace_data else None
    return AgentResponse(final_reply=body["reply"], trace=trace)
