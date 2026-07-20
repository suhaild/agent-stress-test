"""Bundled demo agent (built on an LLMProvider)."""

from agent_stress_test.models import AgentResponse, AgentSpec, Message
from agent_stress_test.ports import LLMProvider, TargetAgent
from agent_stress_test.targets.prompt_rendering import _render_system_prompt
from agent_stress_test.targets.react_parsing import parse_react_completion


class SampleAgent(TargetAgent):
    """A general tool-calling / ReAct-style demo agent driven by an LLMProvider.

    Describes its tools and rules to the LLM via its system prompt and asks it
    to narrate its reasoning in a recognizable Thought/Action/Observation/Final
    Answer format, which is then parsed into a trace — there are no tool
    backends to invoke in this bundled demo, so the LLM narrates rather than
    actually calling anything. See ``AdvancedSampleAgent``
    (``sample_agent_advanced.py``) for a harder sibling that executes tools
    for real against an in-memory fake backend.
    """

    def __init__(self, agent_spec: AgentSpec, llm: LLMProvider) -> None:
        self._agent_spec = agent_spec
        self._llm = llm

    def respond(self, conversation: list[Message]) -> AgentResponse:
        # Identical on every call within a run — a prime prompt-caching breakpoint.
        system = Message(role="system", content=_render_system_prompt(self._agent_spec), cache=True)
        completion = self._llm.complete([system, *conversation])
        return parse_react_completion(completion)
