from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent_stress_test.models import AgentResponse, AgentSpec, Message, Step, ToolSpec
from agent_stress_test.ports import TargetAgent
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.targets.http_agent import HttpAgent
from agent_stress_test.targets.python_fn import PythonFunctionAgent
from agent_stress_test.targets.sample_agent import SampleAgent, _render_system_prompt


def make_agent_spec(**overrides) -> AgentSpec:
    defaults = dict(
        name="test_agent",
        system_prompt="You are a helpful assistant.",
        tools=[
            ToolSpec(name="lookup_order", description="Look up an order by ID."),
            ToolSpec(name="initiate_return", description="Start a return."),
        ],
        rules=["Never invent data.", "Always be polite."],
    )
    defaults.update(overrides)
    return AgentSpec(**defaults)


# --- SampleAgent -------------------------------------------------------


def test_sample_agent_is_a_target_agent():
    assert isinstance(SampleAgent(make_agent_spec(), FakeLLMProvider()), TargetAgent)


def test_render_system_prompt_includes_system_prompt_tools_and_rules():
    spec = make_agent_spec()
    prompt = _render_system_prompt(spec)

    assert spec.system_prompt in prompt
    for tool in spec.tools:
        assert tool.name in prompt
        assert tool.description in prompt
    for rule in spec.rules:
        assert rule in prompt


def test_render_system_prompt_omits_tools_section_when_no_tools():
    spec = make_agent_spec(tools=[])
    prompt = _render_system_prompt(spec)

    assert "Available tools" not in prompt


def test_sample_agent_respond_prepends_system_message_and_forwards_history():
    llm = FakeLLMProvider()
    agent = SampleAgent(make_agent_spec(), llm)
    conversation = [Message(role="user", content="Where is my order?")]

    agent.respond(conversation)

    assert len(llm.calls) == 1
    sent = llm.calls[0]
    assert sent[0].role == "system"
    assert sent[1:] == conversation


def test_sample_agent_multi_turn_conversation_uses_scripted_replies():
    llm = FakeLLMProvider(responses=["first reply", "second reply"])
    agent = SampleAgent(make_agent_spec(), llm)

    response1 = agent.respond([Message(role="user", content="hi")])
    response2 = agent.respond(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content=response1.final_reply),
            Message(role="user", content="and then?"),
        ]
    )

    assert response1.final_reply == "first reply"
    assert response1.trace is None
    assert response2.final_reply == "second reply"
    assert len(llm.calls[1]) == 4  # system + 3 conversation messages


def test_sample_agent_returns_populated_trace_on_react_formatted_completion():
    react_completion = (
        "Thought: I should look up the order first.\n"
        "Action: lookup_order\n"
        "Action Input: 12345\n"
        "Observation: order shipped yesterday\n"
        "Thought: I now have enough information to answer.\n"
        "Final Answer: Your order shipped yesterday and is on its way."
    )
    llm = FakeLLMProvider(responses=[react_completion])
    agent = SampleAgent(make_agent_spec(), llm)

    response = agent.respond([Message(role="user", content="Where is my order?")])

    assert response.final_reply == "Your order shipped yesterday and is on its way."
    assert response.trace == [
        Step(
            thought="I should look up the order first.",
            action="lookup_order",
            action_input="12345",
            observation="order shipped yesterday",
        ),
        Step(thought="I now have enough information to answer."),
    ]


# --- PythonFunctionAgent -------------------------------------------------


def test_python_function_agent_is_a_target_agent():
    assert isinstance(PythonFunctionAgent(lambda conversation: "reply"), TargetAgent)


def test_python_function_agent_wraps_bare_string_with_no_trace():
    agent = PythonFunctionAgent(lambda conversation: "fixed reply")
    result = agent.respond([Message(role="user", content="hi")])
    assert result == AgentResponse(final_reply="fixed reply", trace=None)


def test_python_function_agent_passes_through_trace_when_provided():
    steps = [Step(thought="checked the order", action="lookup_order")]
    agent = PythonFunctionAgent(
        lambda conversation: AgentResponse(final_reply="shipped", trace=steps)
    )

    result = agent.respond([Message(role="user", content="hi")])

    assert result == AgentResponse(final_reply="shipped", trace=steps)


def test_python_function_agent_passes_conversation_through_in_order():
    received = []

    def fn(conversation):
        received.append(conversation)
        return "ok"

    agent = PythonFunctionAgent(fn)
    conversation = [
        Message(role="user", content="first"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="second"),
    ]

    agent.respond(conversation)

    assert received[0] == conversation


def test_python_function_agent_fn_cannot_mutate_callers_list():
    def fn(conversation):
        conversation.append(Message(role="user", content="injected"))
        return "ok"

    agent = PythonFunctionAgent(fn)
    conversation = [Message(role="user", content="hi")]

    agent.respond(conversation)

    assert len(conversation) == 1


# --- HttpAgent -----------------------------------------------------------


def make_http_response(json_body, status_ok: bool = True) -> SimpleNamespace:
    def raise_for_status():
        if not status_ok:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    return SimpleNamespace(json=lambda: json_body, raise_for_status=raise_for_status)


@pytest.fixture
def mock_post(monkeypatch):
    mock = MagicMock(return_value=make_http_response({"reply": "mocked reply"}))
    monkeypatch.setattr("agent_stress_test.targets.http_agent.httpx.post", mock)
    return mock


def test_http_agent_is_a_target_agent(mock_post):
    assert isinstance(HttpAgent(url="http://example.test/respond"), TargetAgent)


def test_http_agent_posts_conversation_and_returns_reply(mock_post):
    agent = HttpAgent(url="http://example.test/respond", timeout=5.0)
    conversation = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]

    result = agent.respond(conversation)

    assert result == AgentResponse(final_reply="mocked reply", trace=None)
    mock_post.assert_called_once_with(
        "http://example.test/respond",
        json={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        },
        headers=None,
        timeout=5.0,
    )


def test_http_agent_multi_turn_maps_growing_history(mock_post):
    agent = HttpAgent(url="http://example.test/respond")

    agent.respond([Message(role="user", content="first")])
    agent.respond(
        [
            Message(role="user", content="first"),
            Message(role="assistant", content="mocked reply"),
            Message(role="user", content="second"),
        ]
    )

    first_payload = mock_post.call_args_list[0].kwargs["json"]
    second_payload = mock_post.call_args_list[1].kwargs["json"]
    assert len(first_payload["messages"]) == 1
    assert len(second_payload["messages"]) == 3


def test_http_agent_passes_through_trace_when_present(monkeypatch):
    trace_payload = [
        {"thought": "checked the order", "action": "lookup_order", "observation": "shipped"},
        {"thought": "ready to answer"},
    ]
    mock = MagicMock(
        return_value=make_http_response({"reply": "it shipped", "trace": trace_payload})
    )
    monkeypatch.setattr("agent_stress_test.targets.http_agent.httpx.post", mock)
    agent = HttpAgent(url="http://example.test/respond")

    result = agent.respond([Message(role="user", content="hi")])

    assert result.final_reply == "it shipped"
    assert result.trace == [
        Step(thought="checked the order", action="lookup_order", observation="shipped"),
        Step(thought="ready to answer"),
    ]


def test_http_agent_returns_no_trace_when_endpoint_omits_it(mock_post):
    result = HttpAgent(url="http://example.test/respond").respond(
        [Message(role="user", content="hi")]
    )
    assert result.trace is None


def test_http_agent_propagates_non_2xx_status(monkeypatch):
    mock = MagicMock(return_value=make_http_response({"reply": "n/a"}, status_ok=False))
    monkeypatch.setattr("agent_stress_test.targets.http_agent.httpx.post", mock)
    agent = HttpAgent(url="http://example.test/respond")

    with pytest.raises(httpx.HTTPStatusError):
        agent.respond([Message(role="user", content="hi")])


def test_http_agent_missing_reply_key_raises(monkeypatch):
    mock = MagicMock(return_value=make_http_response({}))
    monkeypatch.setattr("agent_stress_test.targets.http_agent.httpx.post", mock)
    agent = HttpAgent(url="http://example.test/respond")

    with pytest.raises(KeyError):
        agent.respond([Message(role="user", content="hi")])
