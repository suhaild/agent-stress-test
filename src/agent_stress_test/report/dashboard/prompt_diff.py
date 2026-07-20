"""System-prompt diffing and content-addressed version history."""

import difflib

import pysbd

from agent_stress_test.models import SystemPromptVersion
from agent_stress_test.store.sqlite_store import SqliteStore

_SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def _normalize_for_diff(text: str) -> list[str]:
    """Segment by sentence, not line: the YAML's hard-wrap column rarely
    survives an LLM rewrite, so a line diff would flag whole unchanged
    paragraphs as rewritten."""
    collapsed = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SEGMENTER.segment(collapsed) if s.strip()]


def diff_blocks(old_text: str, new_text: str) -> list[dict]:
    """Groups a unified diff into template-ready blocks: unchanged runs as
    ``{"kind": "context", "text": ...}``, changed runs as ``{"kind":
    "change", "previous": [...], "suggested": [...]}``. Diff header lines are
    dropped."""
    lines = difflib.unified_diff(_normalize_for_diff(old_text), _normalize_for_diff(new_text), lineterm="")
    blocks: list[dict] = []
    previous: list[str] = []
    suggested: list[str] = []
    context: list[str] = []

    def flush_change() -> None:
        if previous or suggested:
            blocks.append({"kind": "change", "previous": list(previous), "suggested": list(suggested)})
            previous.clear()
            suggested.clear()

    def flush_context() -> None:
        if context:
            blocks.append({"kind": "context", "text": " ".join(context)})
            context.clear()

    for line in lines:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            flush_context()
            previous.append(line[1:])
        elif line.startswith("+"):
            flush_context()
            suggested.append(line[1:])
        else:
            flush_change()
            context.append(line[1:])
    flush_change()
    flush_context()
    return blocks


def record_prompt_version(
    store: SqliteStore, agent_spec_name: str, system_prompt: str
) -> None:
    """No-ops if an identical version is already on file, so restoring an
    old version never logs a duplicate row."""
    already_recorded = any(
        v.system_prompt == system_prompt for v in store.get_system_prompt_versions(agent_spec_name)
    )
    if not already_recorded:
        store.save_system_prompt_version(
            SystemPromptVersion(agent_spec_name=agent_spec_name, system_prompt=system_prompt)
        )


def prompt_version_history(
    current_system_prompt: str, prompt_versions: list[SystemPromptVersion]
) -> list[dict]:
    """One row per version (newest first), each diffed against its
    predecessor; the oldest row has no diff. "current" is whichever row's
    text matches what's live now, not necessarily the newest row."""
    rows = []
    total = len(prompt_versions)
    for i, version in enumerate(prompt_versions):
        predecessor = prompt_versions[i + 1] if i + 1 < total else None
        rows.append(
            {
                "version": version,
                "ordinal": total - i,
                "is_current": version.system_prompt == current_system_prompt,
                "diff_blocks": (
                    diff_blocks(predecessor.system_prompt, version.system_prompt)
                    if predecessor is not None
                    else None
                ),
            }
        )
    return rows
