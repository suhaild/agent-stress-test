"""Bundled demo agent (built on an LLMProvider)."""

from agent_stress_test.models import AgentResponse, AgentSpec, Message
from agent_stress_test.ports import LLMProvider, TargetAgent
from agent_stress_test.targets.prompt_rendering import _render_system_prompt
from agent_stress_test.targets.react_parsing import parse_react_completion


class SampleAgent(TargetAgent):
    """A ReAct-style demo agent: narrates Thought/Action/Observation/Final Answer, no real tool execution."""

    def __init__(self, agent_spec: AgentSpec, llm: LLMProvider) -> None:
        self._agent_spec = agent_spec
        self._llm = llm

    def respond(self, conversation: list[Message]) -> AgentResponse:
        # Identical on every call within a run — a prime prompt-caching breakpoint.
        system = Message(role="system", content=_render_system_prompt(self._agent_spec), cache=True)
        completion = self._llm.complete([system, *conversation])
        return parse_react_completion(completion)
