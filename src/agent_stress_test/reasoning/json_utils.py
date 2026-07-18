"""Shared JSON-parsing helpers for every reasoning-layer component that asks
an LLM to respond with a bare JSON object.
"""

import re

# Real models routinely wrap their JSON output in a markdown code fence even
# when told to respond with ONLY the JSON object (Claude does this reliably,
# confirmed against a live model) — model_validate_json() can't parse the
# fence's backticks, so it's stripped before validation rather than treated
# as a parse failure. Every schema-constrained LLM call in this codebase
# (the tier-2 judge, the profiler, remediation) funnels through this one
# helper, so this is the single place that needs to handle it.
_JSON_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_json_fence(raw: str) -> str:
    match = _JSON_FENCE.match(raw.strip())
    return match.group(1).strip() if match else raw
