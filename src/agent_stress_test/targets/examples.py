"""Example ``kind: python`` target: a minimal bring-your-own callable."""

from agent_stress_test.models import Message


def echo_target(conversation: list[Message]) -> str:
    last_user = next((m.content for m in reversed(conversation) if m.role == "user"), "")
    return f"You said: {last_user}"
