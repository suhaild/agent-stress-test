"""Bridges our LLMProvider port into DeepEval's own model interface.

This is the only module allowed to import ``deepeval`` directly — DeepEval is
reasoning-layer infrastructure (an evaluation framework this codebase builds
on, not an LLM provider SDK that needs its own port; see CLAUDE.md's Golden
Rule #1), the same way ``pysbd`` already is in ``reasoning/judge.py``.

``LLMProviderAsDeepEvalLLM`` embeds the requested Pydantic schema's JSON
Schema into the prompt text itself (behind ``SCHEMA_MARKER``) before calling
``provider.complete()`` — the ``LLMProvider`` port's ``complete(messages) ->
str`` signature never changes, so any provider (real or fake) can serve a
schema-constrained DeepEval call exactly the way it serves a plain one. A
real model benefits too: the appended schema is an extra, explicit
structured-output instruction alongside DeepEval's own prose-and-example
prompt. ``providers/shaped_fake.py``'s ``ShapedFakeLLM`` is the fake that
actually acts on this marker instead of ignoring it.
"""

import asyncio
import json

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider

SCHEMA_MARKER = (
    "\n\n=== RESPONSE_SCHEMA (JSON Schema — respond with ONLY a JSON object matching it) ===\n"
)


class LLMProviderAsDeepEvalLLM(DeepEvalBaseLLM):
    """Adapts any ``LLMProvider`` (real or fake) to DeepEval's model interface."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        super().__init__()

    def load_model(self) -> LLMProvider:
        return self._provider

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        full_prompt = prompt
        if schema is not None:
            full_prompt = f"{prompt}{SCHEMA_MARKER}{json.dumps(schema.model_json_schema())}"
        text = self._provider.complete([Message(role="user", content=full_prompt)])
        if schema is not None:
            return schema.model_validate_json(text)
        return text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return "agent-stress-test-llm-provider"
