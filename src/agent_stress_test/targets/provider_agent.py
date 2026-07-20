"""Bring-your-own: a native tool-calling TargetAgent driven by a bare model id.

Unlike SampleAgent, issues tools via the provider's native tool-calling
interface (``ToolCallingLLM.complete_with_tools``) and records real ToolCalls.
"""

from agent_stress_test.models import (
    AgentResponse,
    AgentSpec,
    Capabilities,
    Message,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from agent_stress_test.ports import TargetAgent, ToolCallingLLM
from agent_stress_test.targets.prompt_rendering import _render_system_prompt

# ToolSpec carries no execution backend, so every tool_use gets this fixed stub.
_STUB_TOOL_RESULT = "[no execution backend configured for this tool in this stress test]"


def _tool_schemas(agent_spec: AgentSpec) -> list[dict]:
    """AgentSpec.tools as litellm/OpenAI native tool schemas (open object params — ToolSpec has no JSON schema)."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
            },
        }
        for tool in agent_spec.tools
    ]


class ProviderAgent(TargetAgent):
    """Wraps a ``ToolCallingLLM`` as a TargetAgent, bounded by ``max_tool_rounds``."""

    def __init__(
        self, provider: ToolCallingLLM, agent_spec: AgentSpec, *, max_tool_rounds: int = 3
    ) -> None:
        self._provider = provider
        self._agent_spec = agent_spec
        self._max_tool_rounds = max_tool_rounds

    def capabilities(self) -> Capabilities:
        return Capabilities(tools=True)

    def respond(self, conversation: list[Message]) -> AgentResponse:
        system = Message(role="system", content=_render_system_prompt(self._agent_spec), cache=True)
        tools = _tool_schemas(self._agent_spec)
        messages = [system, *conversation]
        all_tool_calls: list[ToolCall] = []
        text = ""

        for _ in range(self._max_tool_rounds):
            text, tool_calls = self._provider.complete_with_tools(messages, tools)
            if not tool_calls:
                break
            all_tool_calls.extend(
                tc.model_copy(update={"output": _STUB_TOOL_RESULT}) for tc in tool_calls
            )
            messages.append(
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(id=tc.id, name=tc.name, input=tc.input_parameters)
                        for tc in tool_calls
                    ],
                )
            )
            messages.append(
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(tool_use_id=tc.id, content=_STUB_TOOL_RESULT)
                        for tc in tool_calls
                    ],
                )
            )

        return AgentResponse(final_reply=text, tool_calls=all_tool_calls)
