"""Shared JSON-parsing helpers for reasoning-layer components that expect a
bare JSON reply from an LLM.
"""

import re

# Models often wrap JSON output in a markdown code fence even when told not
# to; strip it before validation rather than treat it as a parse failure.
_JSON_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_json_fence(raw: str) -> str:
    match = _JSON_FENCE.match(raw.strip())
    return match.group(1).strip() if match else raw
