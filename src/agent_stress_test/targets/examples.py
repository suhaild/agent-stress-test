"""A minimal example ``kind: python`` target.

Referenced by ``config/agents/example_python_target.yaml`` via
``target: {kind: python, import_path: agent_stress_test.targets.examples:echo_target}``
— demonstrates the shape a bring-your-own Python callable takes (see
``composition.py``'s ``_load_python_target``): any
``Callable[[list[Message]], str | AgentResponse]``, importable as
``"module:attribute"``.
"""

from agent_stress_test.models import Message


def echo_target(conversation: list[Message]) -> str:
    last_user = next((m.content for m in reversed(conversation) if m.role == "user"), "")
    return f"You said: {last_user}"
