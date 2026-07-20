"""ReAct-style Thought/Action/Action Input/Observation/Final Answer narration
parsing, shared by SampleAgent (whole-turn) and AdvancedSampleAgent (step-by-step)."""

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
# Tolerates markdown emphasis (`**Final Answer:**`) that real models add routinely.
_LABEL_PATTERN = re.compile(r"^[*_]{0,2}(" + "|".join(_LABELS) + r"):\s*[*_]{0,2}\s*")


def parse_react_completion(text: str) -> AgentResponse:
    """Parses ReAct narration into a trace + final reply; unlabeled text becomes a plain reply."""
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
    """Parses one narration step. Returns (step, None) mid-turn, (step_or_None, reply) on Final Answer.

    Unlabeled text falls back to a plain reply, so a model ignoring the format still terminates the loop.
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
            # A second Thought with no Final Answer between them ends this step.
            break
        current[field] = remainder

    if current:
        return Step(**current), None
    return None, text.strip()
