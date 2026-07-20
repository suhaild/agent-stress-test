"""Opt-in LLM rephrasing of the deterministic run summary.

``orchestration/executive_summary.py``'s deterministic summary is always
computed and shown by default; this is the only place that spends an LLM
call turning those stats into prose, so callers must invoke it explicitly.
"""

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider

_SUMMARIZER_SYSTEM = (
    "You rewrite AI agent stress-test run summaries as punchy, skimmable prose "
    "for an engineer looking at a dashboard. Keep every number and name exactly "
    "as given in the input — never invent or drop a fact, never add a claim the "
    "input didn't already make. Respond with 2-3 sentences of plain prose only, "
    "no headers, no bullet points, no preamble."
)


class RunSummarizer:
    """Rephrases a deterministic run summary into punchier prose (Strategy)."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def summarize(self, deterministic_text: str) -> str:
        messages = [
            Message(role="system", content=_SUMMARIZER_SYSTEM, cache=True),
            Message(role="user", content=deterministic_text),
        ]
        return self._llm.complete(messages)
