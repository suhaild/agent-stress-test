"""A scripted target that emits a real, deliberately-wrong ToolCall.

Verification target for Phase C's ArgumentCorrectnessMetric: the bundled
SampleAgent only narrates tool use as free-text reasoning (a ``Step`` trace)
and never populates ``AgentResponse.tool_calls``, so there's nothing for an
argument-correctness judge to actually score against. This target performs
the same "look up the order the customer mentioned" step a real tool-calling
agent would, but always calls ``lookup_order`` with the WRONG order id — a
deterministic corruption of whatever id the customer actually gave, not a
random one, so a test can assert exactly what's wrong.

Wired declaratively via
``target: {kind: python, import_path: "agent_stress_test.targets.tool_calling_verification_agent:tool_calling_verification_agent"}``
(see ``config/agents/example_tool_calling_verification.yaml``).
"""

import re

from agent_stress_test.models import AgentResponse, Message, ToolCall

_ORDER_ID_PATTERN = re.compile(r"\b\d{4,}\b")
_FALLBACK_ORDER_ID = "00000"


def _wrong_order_id(real_order_id: str) -> str:
    """A deterministically WRONG order id — never equal to ``real_order_id``."""
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
