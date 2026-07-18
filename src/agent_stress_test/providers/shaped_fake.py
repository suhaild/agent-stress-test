"""A schema-aware deterministic fake LLMProvider (for DeepEval-backed tests).

The plain ``FakeLLMProvider`` (``providers/fake.py``) always returns
"fake-reply: ..." — never valid JSON — so any DeepEval metric that calls
``LLMProviderAsDeepEvalLLM.generate(prompt, schema=SomeSchema)`` and then
``SomeSchema.model_validate_json(text)`` crashes immediately (GAP 1). This
fake recognizes the schema the shim embedded in the prompt (see
``reasoning/deepeval_bridge.py``'s ``SCHEMA_MARKER``) and fabricates a
plausible, schema-valid JSON object instead — every DeepEval schema this
codebase constructs through it must produce a valid instance, never a crash.

Fabrication rules, in priority order:
  - a ``$ref``/``anyOf`` wrapper resolves to its target / first non-null
    branch first
  - an ``enum`` (a ``Literal[...]`` field) -> its first allowed value
  - list fields -> ``[]`` (empty always validates — none of DeepEval's own
    schemas set a ``min_length``, and this sidesteps ever having to
    fabricate a nested item's own shape)
  - bool fields are SEMANTICS-AWARE, not blindly ``True``: a field whose
    name contains "complete"/"stop"/"end"/"refus"/"jailbroken" fabricates
    ``False``, everything else ``True``. This is deliberate, not a guess —
    an early spike fabricated every bool as ``True``, which made
    ``deepeval.simulator.schema.ConversationCompletion.is_complete`` come
    back ``True`` on the very first turn, ending every simulated
    conversation at turn 0 before it could do anything.
  - int/float fields -> ``7``
  - anything else (plain strings, untyped fields) -> placeholder text
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
    """A plausible, schema-shaped dict for any Pydantic model's
    ``model_json_schema()`` output — see the module docstring for the
    fabrication rules."""
    return _fabricate_object(json_schema, json_schema.get("$defs", {}))


class ShapedFakeLLM(LLMProvider):
    """Deterministic, schema-aware fake — the DeepEval-specific counterpart
    to ``FakeLLMProvider``. Falls back to the same plain "fake-reply: ..."
    behavior whenever no schema marker is present, so it's also a safe
    drop-in for a non-schema call.
    """

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
        # No real API call was made — see FakeLLMProvider's identical
        # word-count stand-in for why this is good enough offline.
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
