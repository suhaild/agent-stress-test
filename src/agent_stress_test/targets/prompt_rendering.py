"""Renders an AgentSpec's system prompt; shared by SampleAgent and ProviderAgent."""

from agent_stress_test.models import AgentSpec


def _render_system_prompt(agent_spec: AgentSpec) -> str:
    sections = [agent_spec.system_prompt]

    if agent_spec.tools:
        tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in agent_spec.tools)
        sections.append(f"Available tools:\n{tool_lines}")

    rule_lines = "\n".join(f"- {rule.text}" for rule in agent_spec.rules)
    sections.append(f"Rules:\n{rule_lines}")

    sections.append(
        "Think step by step. For each step, write a line starting with "
        "'Thought:', optionally followed by 'Action:', 'Action Input:', and "
        "'Observation:' lines. When ready to reply to the user, write a line "
        "starting with 'Final Answer:' followed by your reply."
    )

    return "\n\n".join(sections)
