"""Writes a new ``system_prompt``, or one more ``rules:`` entry, into an agent
spec's YAML file, in place.

Split from ``config.py`` (plain YAML reading): this is real, self-contained
logic — in-place block-splices that preserve every comment and unrelated
line — with a different caller (the dashboard's fix-apply and
candidate-rule-apply flows) and a different failure mode (roll back on a bad
write) than a plain read.
"""

import re
from pathlib import Path

import yaml

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentSpec, Rule

_SYSTEM_PROMPT_HEADER = re.compile(r"^(system_prompt:\s*)([|>][+-]?)\s*$")
_RULES_HEADER = re.compile(r"^rules:\s*$")


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


def _append_rule_block(raw_yaml: str, rule: Rule) -> str:
    """Splice one more list item onto the end of the top-level `rules:`
    list, byte-for-byte everywhere else — the list-shaped analog of
    `_replace_system_prompt_block`'s scalar-block splice, for the same
    comment-preserving reason (a full `yaml.safe_load` + `yaml.dump` round
    trip would silently drop every comment in the file).
    """
    lines = raw_yaml.splitlines()
    start_idx = next((i for i, line in enumerate(lines) if _RULES_HEADER.match(line)), None)
    if start_idx is None:
        raise ValueError("No top-level 'rules:' key found in the YAML.")

    indent = None
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue  # blank line inside (or trailing after) the list
        line_indent = len(line) - len(line.lstrip(" "))
        if indent is None:
            indent = line_indent
        if line_indent < indent:
            end_idx = i
            break
    indent = 2 if indent is None else indent

    # Trailing blank lines in the captured span are formatting between this
    # key and the next one, not part of the list — leave them where they are.
    while end_idx > start_idx + 1 and not lines[end_idx - 1].strip():
        end_idx -= 1

    data: dict = {"id": rule.id, "text": rule.text, "severity": rule.severity}
    if rule.check_type is not None:
        data["check_type"] = rule.check_type
    if rule.params:
        data["params"] = rule.params
    dumped = yaml.safe_dump(
        data, sort_keys=False, default_flow_style=False, allow_unicode=True
    ).rstrip("\n")

    prefix_first = f"{' ' * indent}- "
    prefix_rest = " " * (indent + 2)
    new_block = [
        f"{prefix_first}{line}" if i == 0 else (f"{prefix_rest}{line}" if line else "")
        for i, line in enumerate(dumped.splitlines())
    ]

    return "\n".join(lines[:end_idx] + [""] + new_block + lines[end_idx:]) + "\n"


def apply_candidate_rule(path: str | Path, rule: Rule) -> AgentSpec:
    """Append one more rule into an agent spec's YAML `rules:` list, on disk.

    Refuses to create a duplicate `id` — the judge, verdicts, and regression
    cases all look rules up by id, so a collision would make that lookup
    ambiguous. Validates the result by reloading it before returning; if the
    edit somehow produced invalid YAML or a schema violation, the original
    file content is restored, same as `apply_system_prompt`.
    """
    path = Path(path)
    existing_spec = load_agent_spec(path)
    if any(existing.id == rule.id for existing in existing_spec.rules):
        raise ValueError(f"Rule id '{rule.id}' already exists on this agent spec.")

    original = path.read_text(encoding="utf-8")
    path.write_text(_append_rule_block(original, rule), encoding="utf-8")
    try:
        return load_agent_spec(path)
    except Exception:
        path.write_text(original, encoding="utf-8")
        raise
