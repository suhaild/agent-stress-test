"""SampleAgent's harder sibling: executes each narrated Action for real against
an in-memory fake tool backend instead of letting the model invent Observations."""

from collections.abc import Callable
from typing import Any

from agent_stress_test.models import AgentResponse, AgentSpec, Capabilities, Message, Step, ToolCall
from agent_stress_test.ports import LLMProvider, TargetAgent
from agent_stress_test.targets.prompt_rendering import _render_system_prompt
from agent_stress_test.targets.react_parsing import parse_react_step
from agent_stress_test.targets.tool_backends import parse_action_input

# Bounds the loop so a model that keeps calling tools can never run away.
_MAX_TOOL_STEPS = 4

_CONTINUE_NUDGE = (
    "Continue: either write an 'Action:' with an 'Action Input:' to use a "
    "tool, or write your 'Final Answer:' now."
)
_CONCLUDE_NUDGE = (
    "You are out of tool calls for this turn. Write your 'Final Answer:' "
    "now, based only on what the tools actually returned."
)


class AdvancedSampleAgent(TargetAgent):
    """Like ``SampleAgent``, but loops: narrate a step, execute its Action for real,
    feed back the real Observation, repeat until Final Answer or ``_MAX_TOOL_STEPS``."""

    def __init__(
        self,
        agent_spec: AgentSpec,
        llm: LLMProvider,
        tool_backend: dict[str, Callable[[dict[str, Any]], str]],
    ) -> None:
        self._agent_spec = agent_spec
        self._llm = llm
        self._tool_backend = tool_backend

    def capabilities(self) -> Capabilities:
        return Capabilities(tools=True)

    def respond(self, conversation: list[Message]) -> AgentResponse:
        system = Message(role="system", content=_render_system_prompt(self._agent_spec), cache=True)
        messages: list[Message] = [system, *conversation]
        steps: list[Step] = []
        tool_calls: list[ToolCall] = []

        for call_index in range(_MAX_TOOL_STEPS):
            completion = self._llm.complete(messages)
            step, final_reply = parse_react_step(completion)
            if step is not None:
                steps.append(step)
            messages.append(Message(role="assistant", content=completion))

            if final_reply is not None:
                return AgentResponse(
                    final_reply=final_reply, trace=steps or None, tool_calls=tool_calls
                )
            if step is None or not step.action:
                messages.append(Message(role="user", content=_CONTINUE_NUDGE))
                continue

            arguments = parse_action_input(step.action_input or "")
            backend_fn = self._tool_backend.get(step.action)
            observation = (
                backend_fn(arguments) if backend_fn else f"Tool '{step.action}' is not available."
            )
            if backend_fn is not None:
                tool_calls.append(
                    ToolCall(
                        id=f"call-{call_index}",
                        name=step.action,
                        input_parameters=arguments,
                        output=observation,
                    )
                )
            # role="user", not "tool": this is free-text narration with no tool_call_id,
            # and litellm's Anthropic transform requires one for role="tool" messages.
            messages.append(Message(role="user", content=f"Observation: {observation}"))

        final_completion = self._llm.complete(
            [*messages, Message(role="user", content=_CONCLUDE_NUDGE)]
        )
        _, forced_final = parse_react_step(final_completion)
        return AgentResponse(
            final_reply=forced_final or final_completion.strip(),
            trace=steps or None,
            tool_calls=tool_calls,
        )
