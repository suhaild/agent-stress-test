"""A scripted target that emits a real ToolCall with a deliberately, deterministically
wrong order id — gives an argument-correctness judge something concrete to score against."""

import re

from agent_stress_test.models import AgentResponse, Message, ToolCall

_ORDER_ID_PATTERN = re.compile(r"\b\d{4,}\b")
_FALLBACK_ORDER_ID = "00000"


def _wrong_order_id(real_order_id: str) -> str:
    digits = [int(d) for d in real_order_id]
    digits[-1] = (digits[-1] + 1) % 10
    return "".join(str(d) for d in digits)


def tool_calling_verification_agent(conversation: list[Message]) -> AgentResponse:
    user_text = " ".join(
        m.content for m in conversation if m.role == "user" and isinstance(m.content, str)
    )
    match = _ORDER_ID_PATTERN.search(user_text)
    real_order_id = match.group(0) if match else _FALLBACK_ORDER_ID
    wrong_order_id = _wrong_order_id(real_order_id)

    call = ToolCall(
        id="call_1",
        name="lookup_order",
        input_parameters={"order_id": wrong_order_id},
        output="No order found with that ID.",
    )
    return AgentResponse(
        final_reply=(
            f"I looked up order {wrong_order_id}, but I couldn't find anything matching that."
        ),
        tool_calls=[call],
    )
