"""ReAct-style ``Thought:``/``Action:``/``Action Input:``/``Observation:``/
``Final Answer:`` narration parsing, shared by every target that asks its LLM
to narrate reasoning in this format:

- ``SampleAgent`` (see ``sample_agent.py``) asks for a whole turn — trace and
  final reply — in a single completion, and never executes anything; the
  narrated ``Observation:`` lines (if any) are the model's own invention.
- ``AdvancedSampleAgent`` (see ``sample_agent_advanced.py``) asks for one step
  at a time so it can execute the named tool for real and substitute a real
  ``Observation:`` before the next completion.
"""

import re

from agent_stress_test.models import AgentResponse, Step

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
# included, as the reply (see `parse_react_completion`'s docstring).
_LABEL_PATTERN = re.compile(r"^[*_]{0,2}(" + "|".join(_LABELS) + r"):\s*[*_]{0,2}\s*")


def parse_react_completion(text: str) -> AgentResponse:
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


def parse_react_step(text: str) -> tuple[Step | None, str | None]:
    """Parse ONE loop iteration's completion for an agent that executes tools
    for real (see ``AdvancedSampleAgent``), rather than a whole multi-step
    narration at once.

    Returns ``(step, None)`` when the model asked for a step (a Thought and/or
    an Action to execute) with no Final Answer yet; ``(step_or_None, reply)``
    once a 'Final Answer:' label is seen — any Thought/Action captured just
    before it is still returned alongside the reply, matching
    ``parse_react_completion``'s behavior of closing out a trailing thought
    rather than discarding it. Falls back to treating fully unlabeled text as
    a plain final reply, so a model that ignores the format still terminates
    the loop instead of hanging it for ``_MAX_TOOL_STEPS`` iterations.
    """
    lines = text.splitlines()
    current: dict[str, str] = {}

    for index, line in enumerate(lines):
        stripped = line.strip()
        match = _LABEL_PATTERN.match(stripped)
        if match is None:
            continue
        label = match.group(1)
        remainder = stripped[match.end() :].strip()

        if label == _FINAL_ANSWER_LABEL:
            tail = [remainder] if remainder else []
            tail.extend(lines[index + 1 :])
            final_step = Step(**current) if current else None
            return final_step, "\n".join(tail).strip()

        field = _STEP_FIELD_BY_LABEL[label]
        if field == "thought" and current:
            # A second Thought with no Final Answer in between closes out the
            # step in progress — this loop iteration is done narrating.
            break
        current[field] = remainder

    if current:
        return Step(**current), None
    return None, text.strip()
