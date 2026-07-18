"""Writes a new ``system_prompt`` into an agent spec's YAML file, in place.

Split from ``config.py`` (plain YAML reading): this is real, self-contained
logic — an in-place block-splice that preserves every comment and unrelated
line — with a different caller (the dashboard's fix-apply flow) and a
different failure mode (roll back on a bad write) than a plain read.
"""

import re
from pathlib import Path

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentSpec

_SYSTEM_PROMPT_HEADER = re.compile(r"^(system_prompt:\s*)([|>][+-]?)\s*$")


def _replace_system_prompt_block(raw_yaml: str, new_system_prompt: str) -> str:
    """Splice a new value into the `system_prompt: |` block, byte-for-byte
    everywhere else.

    Not a `yaml.safe_load` + `yaml.dump` round trip — PyYAML has no concept
    of comments, so re-serializing the whole file would silently drop every
    one (e.g. the reasoning behind a rule's regex, written as a `#` comment
    next to it). This only touches the exact line span of the system_prompt
    block's content; every other line is untouched.
    """
    lines = raw_yaml.splitlines()
    start_idx = next(
        (i for i, line in enumerate(lines) if _SYSTEM_PROMPT_HEADER.match(line)), None
    )
    if start_idx is None:
        raise ValueError("No 'system_prompt: |' block found in the YAML.")

    indent = None
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue  # blank line inside (or trailing after) the block
        line_indent = len(line) - len(line.lstrip(" "))
        if indent is None:
            indent = line_indent
        if line_indent < indent:
            end_idx = i
            break
    indent = 2 if indent is None else indent

    # Trailing blank lines in the captured span are formatting between this
    # key and the next one, not part of the scalar's content — leave them
    # where they are instead of replacing them away.
    while end_idx > start_idx + 1 and not lines[end_idx - 1].strip():
        end_idx -= 1

    prefix = " " * indent
    new_block = [f"{prefix}{line}" if line else "" for line in new_system_prompt.splitlines()]

    return "\n".join(lines[: start_idx + 1] + new_block + lines[end_idx:]) + "\n"


def apply_system_prompt(path: str | Path, new_system_prompt: str) -> AgentSpec:
    """Write a new system_prompt into an agent spec's YAML file on disk.

    Validates the result by reloading it before returning; if the edit
    somehow produced invalid YAML or a schema violation, the original file
    content is restored — a failed edit should never leave the file broken.
    """
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    path.write_text(_replace_system_prompt_block(original, new_system_prompt), encoding="utf-8")
    try:
        return load_agent_spec(path)
    except Exception:
        path.write_text(original, encoding="utf-8")
        raise
