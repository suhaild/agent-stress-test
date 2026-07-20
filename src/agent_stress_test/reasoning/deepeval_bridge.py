"""Bridges our LLMProvider port into DeepEval's own model interface.

Only module allowed to import ``deepeval`` directly — it's evaluation-
framework infrastructure, not an LLM provider SDK (see CLAUDE.md Golden Rule #1).
"""

import asyncio
import json

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from agent_stress_test.models import Message
from agent_stress_test.ports import LLMProvider
from agent_stress_test.reasoning.json_utils import _strip_json_fence

# Embeds the schema into the prompt text after this marker, since LLMProvider.complete()
# takes no schema param — providers/shaped_fake.py's ShapedFakeLLM is the fake that acts on it.
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
            return schema.model_validate_json(_strip_json_fence(text))
        return text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return "agent-stress-test-llm-provider"
