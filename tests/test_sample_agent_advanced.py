from agent_stress_test.models import Capabilities, Message
from agent_stress_test.ports import TargetAgent
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.targets.sample_agent_advanced import (
    _CONCLUDE_NUDGE,
    _CONTINUE_NUDGE,
    _MAX_TOOL_STEPS,
    AdvancedSampleAgent,
)
from agent_stress_test.targets.tool_backends import build_northwind_tool_backend
from tests.conftest import make_agent_spec


def _agent(llm, tool_backend=None):
    return AdvancedSampleAgent(
        make_agent_spec(), llm, tool_backend or build_northwind_tool_backend()
    )


def test_advanced_sample_agent_is_a_target_agent():
    assert isinstance(_agent(FakeLLMProvider()), TargetAgent)


def test_advanced_sample_agent_reports_tools_capability():
    assert _agent(FakeLLMProvider()).capabilities() == Capabilities(tools=True)


def test_advanced_sample_agent_executes_a_real_tool_call_and_grounds_the_reply():
    llm = FakeLLMProvider(
        responses=[
            'Thought: I should look this up.\nAction: lookup_order\nAction Input: {"order_id": "NW-1001"}',
            "Thought: Now I can answer.\nFinal Answer: Tracking: 1Z999AA10123456789.",
        ]
    )
    agent = _agent(llm)

    response = agent.respond([Message(role="user", content="Where is order NW-1001?")])

    assert response.final_reply == "Tracking: 1Z999AA10123456789."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup_order"
    assert "1Z999AA10123456789" in response.tool_calls[0].output
    assert response.trace is not None
    assert response.trace[0].action == "lookup_order"


def test_advanced_sample_agent_feeds_the_real_observation_back_to_the_model():
    llm = FakeLLMProvider(
        responses=[
            'Action: lookup_order\nAction Input: {"order_id": "NW-1003"}',
            "Final Answer: ok",
        ]
    )
    agent = _agent(llm)

    agent.respond([Message(role="user", content="hi")])

    second_call = llm.calls[1]
    # role="user", not "tool" -- this is free-text ReAct narration, not the
    # provider's native tool-calling protocol (see sample_agent_advanced.py).
    observation_message = next(m for m in second_call if m.content.startswith("Observation:"))
    # The real backend, not the model, produced this text -- the model never
    # said anything about a tracking number, so this could only have come
    # from actually executing lookup_order.
    assert "Order NW-1003" in observation_message.content
    assert "1Z999AA10555555555" in observation_message.content


def test_advanced_sample_agent_nudges_when_no_action_or_final_answer_given():
    llm = FakeLLMProvider(
        responses=["Thought: just thinking, no action yet.", "Final Answer: done."]
    )
    agent = _agent(llm)

    response = agent.respond([Message(role="user", content="hi")])

    assert response.final_reply == "done."
    nudge_message = llm.calls[1][-1]
    assert nudge_message.content == _CONTINUE_NUDGE


def test_advanced_sample_agent_reports_unavailable_tool_without_crashing():
    llm = FakeLLMProvider(
        responses=[
            "Action: apply_loyalty_discount\nAction Input: 20%",
            "Final Answer: I can't apply a discount myself.",
        ]
    )
    agent = _agent(llm)

    response = agent.respond([Message(role="user", content="give me a discount")])

    assert response.final_reply == "I can't apply a discount myself."
    assert response.tool_calls == []  # unavailable tool is never recorded as a real call
    tool_message = llm.calls[1][-1]
    assert "not available" in tool_message.content


def test_advanced_sample_agent_bounds_the_loop_and_forces_a_conclusion():
    # No Action, no Final Answer, every turn -- must terminate within
    # _MAX_TOOL_STEPS nudges plus one forced-conclusion call, never hang.
    responses = ["Thought: still thinking."] * _MAX_TOOL_STEPS + [
        "Final Answer: giving up gracefully."
    ]
    llm = FakeLLMProvider(responses=responses)
    agent = _agent(llm)

    response = agent.respond([Message(role="user", content="hi")])

    assert response.final_reply == "giving up gracefully."
    assert len(llm.calls) == _MAX_TOOL_STEPS + 1
    assert llm.calls[-1][-1].content == _CONCLUDE_NUDGE


def test_advanced_sample_agent_falls_back_to_raw_text_if_forced_conclusion_has_no_label():
    responses = ["Thought: still thinking."] * _MAX_TOOL_STEPS + ["just some plain text"]
    llm = FakeLLMProvider(responses=responses)
    agent = _agent(llm)

    response = agent.respond([Message(role="user", content="hi")])

    assert response.final_reply == "just some plain text"
