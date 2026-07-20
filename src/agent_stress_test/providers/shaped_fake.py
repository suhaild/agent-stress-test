"""Schema-aware fake LLMProvider for DeepEval-backed tests.

Detects the ``SCHEMA_MARKER`` embedded in the prompt (see
``reasoning/deepeval_bridge.py``) and fabricates a schema-valid JSON object
instead of plain text, so DeepEval metrics that validate the reply against a
Pydantic schema don't crash.
"""

import json
import threading
from typing import Any

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.deepeval_bridge import SCHEMA_MARKER

_BOOL_FALSE_NAME_MARKERS = ("complete", "stop", "end", "refus", "jailbroken")
_PLACEHOLDER_TEXT = "plausible text"


def _fabricate_value(name: str, node: dict, defs: dict) -> Any:
    if "$ref" in node:
        ref_name = node["$ref"].rsplit("/", 1)[-1]
        return _fabricate_value(name, defs.get(ref_name, {}), defs)
    if "anyOf" in node:
        options = [opt for opt in node["anyOf"] if opt.get("type") != "null"]
        return _fabricate_value(name, options[0], defs) if options else None
    if "enum" in node:
        return node["enum"][0]
    node_type = node.get("type")
    if node_type == "array":
        return []
    if node_type == "boolean":
        # Defaults True, but "done"-like field names default False, or every
        # fabricated reply would signal conversation-complete on turn 0.
        lowered = name.lower()
        return not any(marker in lowered for marker in _BOOL_FALSE_NAME_MARKERS)
    if node_type in ("integer", "number"):
        return 7
    if node_type == "object":
        return _fabricate_object(node, defs)
    return _PLACEHOLDER_TEXT


def _fabricate_object(node: dict, defs: dict) -> dict:
    return {
        field_name: _fabricate_value(field_name, prop, defs)
        for field_name, prop in node.get("properties", {}).items()
    }


def fabricate_from_json_schema(json_schema: dict) -> dict:
    """A plausible, schema-valid dict for a Pydantic ``model_json_schema()`` output."""
    return _fabricate_object(json_schema, json_schema.get("$defs", {}))


class ShapedFakeLLM(LLMProvider):
    """Falls back to plain "fake-reply: ..." when no schema marker is present."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[Message]] = []
        self._lock = threading.Lock()

    def complete(self, messages: list[Message]) -> str:
        with self._lock:
            self.calls.append(list(messages))
        last_content = messages[-1].content if messages else ""
        if isinstance(last_content, str) and SCHEMA_MARKER in last_content:
            _, _, schema_json = last_content.partition(SCHEMA_MARKER)
            reply = json.dumps(fabricate_from_json_schema(json.loads(schema_json)))
        else:
            reply = f"fake-reply: {last_content}"
        # No real call was made: word count stands in for token count, cost is 0.0.
        prompt_tokens = sum(len(m.content.split()) for m in messages if isinstance(m.content, str))
        completion_tokens = len(reply.split())
        self.meter.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,
        )
        return reply

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return [self.complete(messages) for _ in range(n)]
