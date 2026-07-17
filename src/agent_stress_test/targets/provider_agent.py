"""Bring-your-own: a native tool-calling TargetAgent driven by a bare model id.

Unlike ``SampleAgent`` (which narrates ReAct-style reasoning as plain text
and parses it back out), this issues the AgentSpec's tools to the model
through litellm's native tool-calling interface
(``LiteLLMProvider.complete_with_tools``) and records genuine ``ToolCall``
structures — useful for stress-testing how a target behaves with real
tool_use content blocks, not a text-parsed simulation of them.
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
from agent_stress_test.ports import TargetAgent
from agent_stress_test.providers.litellm_provider import LiteLLMProvider
from agent_stress_test.targets.sample_agent import _render_system_prompt

# No tool declared on an AgentSpec has a real backend here (ToolSpec only
# ever carries a name/description, never an implementation - see models.py),
# so every tool_use is resolved with this fixed stub result instead of
# actually executing anything.
_STUB_TOOL_RESULT = "[no execution backend configured for this tool in this stress test]"


def _tool_schemas(spec: AgentSpec) -> list[dict]:
    """AgentSpec.tools translated to litellm/OpenAI's native tool-schema shape.

    Every tool is declared as accepting an open object, since ToolSpec never
    carries a parameters JSON schema — the same assumption SampleAgent's
    system-prompt narration already makes about what tools can take.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
            },
        }
        for tool in spec.tools
    ]


class ProviderAgent(TargetAgent):
    """Wraps a bare model id (via ``LiteLLMProvider``) as a TargetAgent.

    Bounded by ``max_tool_rounds`` so a model that keeps calling tools can
    never loop forever — each round resolves every pending tool_use with the
    fixed stub result and asks the model once more, until it replies with no
    further tool calls or the round limit is hit.
    """

    def __init__(
        self, provider: LiteLLMProvider, agent_spec: AgentSpec, *, max_tool_rounds: int = 3
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
