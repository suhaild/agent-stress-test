"""System-prompt diffing and content-addressed version history — the
dashboard's "suggest a fix, show a diff, let a human apply/revert it" flow.

Pure text logic plus one ``Store``-backed history query; no HTTP/template
coupling, so it's usable (and testable) independently of ``server.py``'s
route handlers.
"""

import difflib

import pysbd

from agent_stress_test.models import SystemPromptVersion
from agent_stress_test.store.sqlite_store import SqliteStore

_SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def _normalize_for_diff(text: str) -> list[str]:
    """Split text into sentences for diffing, ignoring incidental line-wrap
    width. A raw line-based diff is misleading here: the YAML's system_prompt
    is hard-wrapped at a fixed column width, but an LLM's suggested
    replacement rarely reproduces that exact wrap point — so a plain
    ``str.splitlines()`` diff shows the whole paragraph as removed-and-re-added
    even when only one sentence actually changed. Collapsing whitespace first
    (so wrapping can't matter) and segmenting by sentence gives a diff that
    tracks meaning, not incidental formatting.
    """
    collapsed = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SEGMENTER.segment(collapsed) if s.strip()]


def _diff_blocks(old_text: str, new_text: str) -> list[dict]:
    """Groups a unified diff into template-ready blocks for a browser
    audience, not a `git diff` reader: each contiguous run of unchanged
    sentences becomes one ``{"kind": "context", "text": ...}`` row (labeled
    "unchanged" in the template — shown only for orientation, so it's clear
    it's surrounding prompt text, not part of the edit), and each run of
    removed/added sentences becomes one ``{"kind": "change", "previous":
    [...], "suggested": [...]}`` pair, labeled "previous"/"suggested"
    explicitly rather than relying on red/green coloring alone to say which
    side is which. The raw ``---``/``+++``/``@@ -3,4 +3,4 @@`` unified-diff
    header lines (meaningful line positions to a `git diff` reader, noise to
    everyone else) are dropped entirely.
    """
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


def _record_prompt_version(
    store: SqliteStore, agent_spec_name: str, system_prompt: str
) -> None:
    """Content-addressed history: records ``system_prompt`` as a version only
    if an identical one isn't already on file for this agent. Without this,
    restoring an old version (or re-applying one just undone) would log a
    fresh duplicate row every time — the history would grow on every click
    instead of being a genuine list of distinct versions. Restoring content
    that's already recorded just moves which row counts as "current"; it
    never mints a new one.
    """
    already_recorded = any(
        v.system_prompt == system_prompt for v in store.get_system_prompt_versions(agent_spec_name)
    )
    if not already_recorded:
        store.save_system_prompt_version(
            SystemPromptVersion(agent_spec_name=agent_spec_name, system_prompt=system_prompt)
        )


def _prompt_version_history(
    current_system_prompt: str, prompt_versions: list[SystemPromptVersion]
) -> list[dict]:
    """One row per distinct version ever recorded for this agent
    (most-recent-created first), each diffed against whichever version came
    immediately before it in time — the oldest entry is the prompt as it
    stood before any fix was ever applied, shown as-is with no diff. Because
    the version list is content-addressed (see ``_record_prompt_version``),
    every row — including ones already superseded and then brought back —
    is restorable via the same action, and "current" is whichever row's text
    matches what's live right now, not necessarily the newest row.
    """
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
                    _diff_blocks(predecessor.system_prompt, version.system_prompt)
                    if predecessor is not None
                    else None
                ),
            }
        )
    return rows
