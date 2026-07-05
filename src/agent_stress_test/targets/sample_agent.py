"""Bundled demo agent (built on an LLMProvider)."""

from agent_stress_test.models import AgentResponse, AgentSpec, Message, Step
from agent_stress_test.ports import LLMProvider, TargetAgent

_STEP_FIELD_BY_PREFIX = {
    "Thought:": "thought",
    "Action:": "action",
    "Action Input:": "action_input",
    "Observation:": "observation",
}
_FINAL_ANSWER_PREFIX = "Final Answer:"


def _render_system_prompt(spec: AgentSpec) -> str:
    sections = [spec.system_prompt]

    if spec.tools:
        tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in spec.tools)
        sections.append(f"Available tools:\n{tool_lines}")

    rule_lines = "\n".join(f"- {rule.text}" for rule in spec.rules)
    sections.append(f"Rules:\n{rule_lines}")

    sections.append(
        "Think step by step. For each step, write a line starting with "
        "'Thought:', optionally followed by 'Action:', 'Action Input:', and "
        "'Observation:' lines. When ready to reply to the user, write a line "
        "starting with 'Final Answer:' followed by your reply."
    )

    return "\n\n".join(sections)


def _parse_react_completion(text: str) -> AgentResponse:
    """Parse a ReAct-style completion into a trace and a final reply.

    Recognizes 'Thought:'/'Action:'/'Action Input:'/'Observation:'/'Final
    Answer:' line prefixes. Text using none of them is treated as a plain
    reply with no trace — this parser never fabricates steps.
    """
    steps: list[Step] = []
    current: dict[str, str] = {}
    final_reply = text.strip()

    for line in text.splitlines():
        stripped = line.strip()

        prefix_match = next(
            (p for p in _STEP_FIELD_BY_PREFIX if stripped.startswith(p)), None
        )
        if prefix_match is not None:
            field = _STEP_FIELD_BY_PREFIX[prefix_match]
            if field == "thought" and current:
                steps.append(Step(**current))
                current = {}
            current[field] = stripped[len(prefix_match) :].strip()
            continue

        if stripped.startswith(_FINAL_ANSWER_PREFIX):
            if current:
                steps.append(Step(**current))
                current = {}
            final_reply = stripped[len(_FINAL_ANSWER_PREFIX) :].strip()

    if current:
        steps.append(Step(**current))

    return AgentResponse(final_reply=final_reply, trace=steps or None)


class SampleAgent(TargetAgent):
    """A general tool-calling / ReAct-style demo agent driven by an LLMProvider.

    Describes its tools and rules to the LLM via its system prompt and asks it
    to narrate its reasoning in a recognizable Thought/Action/Observation/Final
    Answer format, which is then parsed into a trace — there are no tool
    backends to invoke in this bundled demo, so the LLM narrates rather than
    actually calling anything.
    """

    def __init__(self, agent_spec: AgentSpec, llm: LLMProvider) -> None:
        self._agent_spec = agent_spec
        self._llm = llm

    def respond(self, conversation: list[Message]) -> AgentResponse:
        system = Message(role="system", content=_render_system_prompt(self._agent_spec))
        completion = self._llm.complete([system, *conversation])
        return _parse_react_completion(completion)
