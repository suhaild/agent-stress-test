"""Bundled demo agent (built on an LLMProvider)."""

import re

from agent_stress_test.models import AgentResponse, AgentSpec, Message, Step
from agent_stress_test.ports import LLMProvider, TargetAgent
from agent_stress_test.targets.prompt_rendering import _render_system_prompt

_STEP_FIELD_BY_LABEL = {
    "Thought": "thought",
    "Action": "action",
    "Action Input": "action_input",
    "Observation": "observation",
}
_FINAL_ANSWER_LABEL = "Final Answer"
# "Action Input" must be tried before "Action" so it isn't matched short.
_LABELS = ("Thought", "Action Input", "Action", "Observation", _FINAL_ANSWER_LABEL)
# Tolerates markdown emphasis real models routinely wrap section labels in
# (`**Final Answer:**`) — matching only the bare "Final Answer:" prefix misses
# these and silently falls back to treating the whole completion, reasoning
# included, as the reply (see `_parse_react_completion`'s docstring).
_LABEL_PATTERN = re.compile(r"^[*_]{0,2}(" + "|".join(_LABELS) + r"):\s*[*_]{0,2}\s*")


def _parse_react_completion(text: str) -> AgentResponse:
    """Parse a ReAct-style completion into a trace and a final reply.

    Recognizes 'Thought:'/'Action:'/'Action Input:'/'Observation:'/'Final
    Answer:' line labels, tolerating markdown emphasis around them
    (`**Final Answer:**`) — real models (Claude included) format these labels
    that way constantly, and matching only the bare prefix would miss them
    entirely, silently falling back to treating the whole completion —
    reasoning included — as the reply. Once 'Final Answer:' is seen, every
    line after it is part of the reply, not just the rest of that one line —
    models write it as a multi-paragraph block, not a single line. Text using
    none of these labels at all is treated as a plain reply with no trace —
    this parser never fabricates steps.
    """
    steps: list[Step] = []
    current: dict[str, str] = {}
    final_reply = text.strip()
    final_answer_lines: list[str] | None = None

    for line in text.splitlines():
        if final_answer_lines is not None:
            final_answer_lines.append(line)
            continue

        stripped = line.strip()
        match = _LABEL_PATTERN.match(stripped)
        if match is None:
            continue
        label = match.group(1)
        remainder = stripped[match.end() :].strip()

        if label == _FINAL_ANSWER_LABEL:
            if current:
                steps.append(Step(**current))
                current = {}
            final_answer_lines = [remainder] if remainder else []
            continue

        field = _STEP_FIELD_BY_LABEL[label]
        if field == "thought" and current:
            steps.append(Step(**current))
            current = {}
        current[field] = remainder

    if current:
        steps.append(Step(**current))
    if final_answer_lines is not None:
        final_reply = "\n".join(final_answer_lines).strip()

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
        # Identical on every call within a run — a prime prompt-caching breakpoint.
        system = Message(
            role="system", content=_render_system_prompt(self._agent_spec), cache=True
        )
        completion = self._llm.complete([system, *conversation])
        return _parse_react_completion(completion)
