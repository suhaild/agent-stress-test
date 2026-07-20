import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from agent_stress_test.composition import _build_target_from_spec, _load_python_target
from agent_stress_test.models import (
    AgentResponse,
    Capabilities,
    Message,
    Step,
    ToolCall,
)
from agent_stress_test.ports import TargetAgent
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.targets.http_agent import HttpAgent
from agent_stress_test.targets.provider_agent import _STUB_TOOL_RESULT, ProviderAgent, _tool_schemas
from agent_stress_test.targets.prompt_rendering import _render_system_prompt
from agent_stress_test.targets.python_fn import PythonFunctionAgent
from agent_stress_test.targets.sample_agent import SampleAgent
from agent_stress_test.targets.sample_agent_advanced import AdvancedSampleAgent
from agent_stress_test.targets.subprocess_agent import SubprocessAgent
from agent_stress_test.targets.tool_calling_verification_agent import (
    tool_calling_verification_agent,
)
from tests.conftest import make_agent_spec


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
        assert rule.text in prompt


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


def test_sample_agent_recognizes_markdown_bold_final_answer_label():
    # Real models (Claude Haiku included) routinely bold section labels —
    # matching only the bare "Final Answer:" prefix misses this entirely and
    # falls back to treating the whole completion, reasoning included, as
    # the reply.
    react_completion = (
        "Thought: Let me check that.\n"
        "**Final Answer:** Your order shipped yesterday and is on its way."
    )
    llm = FakeLLMProvider(responses=[react_completion])
    agent = SampleAgent(make_agent_spec(), llm)

    response = agent.respond([Message(role="user", content="Where is my order?")])

    assert response.final_reply == "Your order shipped yesterday and is on its way."
    assert response.trace == [Step(thought="Let me check that.")]


def test_sample_agent_captures_a_multi_paragraph_final_answer():
    # Models write the label on its own line, then the reply as several
    # paragraphs below it — the old parser only ever captured the rest of
    # the SAME line as the label, silently dropping everything after it.
    react_completion = (
        "Thought: The customer wants several things.\n"
        "\n"
        "**Final Answer:**\n"
        "\n"
        "I understand your frustration.\n"
        "\n"
        "Here's what I can do:\n"
        "- Look up one order at a time\n"
        "- Start a return once I have the order ID\n"
        "\n"
        "What's your order number?"
    )
    llm = FakeLLMProvider(responses=[react_completion])
    agent = SampleAgent(make_agent_spec(), llm)

    response = agent.respond([Message(role="user", content="Help with several orders")])

    assert response.final_reply == (
        "I understand your frustration.\n"
        "\n"
        "Here's what I can do:\n"
        "- Look up one order at a time\n"
        "- Start a return once I have the order ID\n"
        "\n"
        "What's your order number?"
    )
    assert response.trace == [Step(thought="The customer wants several things.")]


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


# --- Declarative target: AgentSpec.target ---------------------------------


def test_agent_spec_with_no_target_block_defaults_to_none():
    assert make_agent_spec().target is None


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "http", "url": "http://example.test/respond"},
        {"kind": "python", "import_path": "agent_stress_test.targets.examples:echo_target"},
        {"kind": "subprocess", "command": ["python", "run.py"]},
        {"kind": "provider", "model": "anthropic/claude-haiku-4-5-20251001"},
        {"kind": "sample_advanced"},
    ],
)
def test_agent_spec_validates_each_target_kind(target):
    spec = make_agent_spec(target=target)
    assert spec.target.kind == target["kind"]


def test_agent_spec_rejects_unknown_target_kind():
    with pytest.raises(ValidationError):
        make_agent_spec(target={"kind": "carrier-pigeon"})


# --- _build_target_from_spec: each kind builds and responds ---------------


def test_build_target_from_spec_with_no_target_returns_sample_agent():
    target = _build_target_from_spec(make_agent_spec(), FakeLLMProvider())
    assert isinstance(target, SampleAgent)


def test_build_target_from_spec_http_kind(monkeypatch):
    mock = MagicMock(return_value=make_http_response({"reply": "mocked reply"}))
    monkeypatch.setattr("agent_stress_test.targets.http_agent.httpx.post", mock)
    spec = make_agent_spec(target={"kind": "http", "url": "http://example.test/respond"})

    target = _build_target_from_spec(spec, FakeLLMProvider())

    assert isinstance(target, HttpAgent)
    result = target.respond([Message(role="user", content="hi")])
    assert result.final_reply == "mocked reply"


def test_build_target_from_spec_python_kind():
    spec = make_agent_spec(
        target={"kind": "python", "import_path": "agent_stress_test.targets.examples:echo_target"}
    )

    target = _build_target_from_spec(spec, FakeLLMProvider())

    assert isinstance(target, PythonFunctionAgent)
    result = target.respond([Message(role="user", content="hello there")])
    assert result.final_reply == "You said: hello there"


def test_load_python_target_rejects_a_path_with_no_attribute():
    with pytest.raises(ValueError, match="module:attribute"):
        _load_python_target("agent_stress_test.targets.examples")


def test_load_python_target_rejects_an_unknown_attribute():
    with pytest.raises(ValueError, match="no attribute"):
        _load_python_target("agent_stress_test.targets.examples:does_not_exist")


def test_build_target_from_spec_subprocess_kind(monkeypatch):
    fake_result = SimpleNamespace(
        returncode=0, stdout=json.dumps({"reply": "subprocess reply"}), stderr=""
    )
    mock_run = MagicMock(return_value=fake_result)
    monkeypatch.setattr("agent_stress_test.targets.subprocess_agent.subprocess.run", mock_run)
    spec = make_agent_spec(target={"kind": "subprocess", "command": ["echo-agent"]})

    target = _build_target_from_spec(spec, FakeLLMProvider())

    assert isinstance(target, SubprocessAgent)
    result = target.respond([Message(role="user", content="hi")])
    assert result.final_reply == "subprocess reply"
    mock_run.assert_called_once()


def test_build_target_from_spec_provider_kind(monkeypatch):
    mock_completion = MagicMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="final reply", tool_calls=None))
            ]
        )
    )
    monkeypatch.setattr(
        "agent_stress_test.providers.litellm_provider.litellm.completion", mock_completion
    )
    spec = make_agent_spec(
        target={"kind": "provider", "model": "anthropic/claude-haiku-4-5-20251001"}
    )

    target = _build_target_from_spec(spec, FakeLLMProvider())

    assert isinstance(target, ProviderAgent)
    result = target.respond([Message(role="user", content="hi")])
    assert result.final_reply == "final reply"


def test_build_target_from_spec_sample_advanced_kind_executes_a_real_tool():
    spec = make_agent_spec(target={"kind": "sample_advanced"})
    llm = FakeLLMProvider(
        responses=[
            'Thought: look it up.\nAction: lookup_order\nAction Input: {"order_id": "NW-1001"}',
            "Thought: done.\nFinal Answer: Your tracking number is 1Z999AA10123456789.",
        ]
    )

    target = _build_target_from_spec(spec, llm)

    assert isinstance(target, AdvancedSampleAgent)
    assert target.capabilities() == Capabilities(tools=True)
    result = target.respond([Message(role="user", content="Where is order NW-1001?")])
    assert result.final_reply == "Your tracking number is 1Z999AA10123456789."
    assert result.tool_calls[0].name == "lookup_order"
    assert "1Z999AA10123456789" in result.tool_calls[0].output


def test_build_target_from_spec_unknown_kind_raises_clean_error():
    # AgentSpec itself already rejects an unknown kind at validation time
    # (see test_agent_spec_rejects_unknown_target_kind); model_copy bypasses
    # that validation, letting this prove _build_target_from_spec's own
    # defensive fallback is a clean ValueError too, not a raw AttributeError.
    spec = make_agent_spec().model_copy(update={"target": SimpleNamespace(kind="carrier-pigeon")})

    with pytest.raises(ValueError, match="Unknown target kind"):
        _build_target_from_spec(spec, FakeLLMProvider())


# --- ProviderAgent ---------------------------------------------------------


class _FakeToolCallingProvider:
    def __init__(self, turns):
        self._turns = list(turns)
        self.calls: list[list[Message]] = []

    def complete_with_tools(self, messages, tools):
        self.calls.append(list(messages))
        return self._turns.pop(0)


def test_provider_agent_is_a_target_agent():
    agent = ProviderAgent(_FakeToolCallingProvider([("hi", [])]), make_agent_spec())
    assert isinstance(agent, TargetAgent)


def test_provider_agent_returns_reply_with_no_tool_calls():
    fake = _FakeToolCallingProvider([("just a plain reply", [])])
    agent = ProviderAgent(fake, make_agent_spec())

    result = agent.respond([Message(role="user", content="hi")])

    assert result == AgentResponse(final_reply="just a plain reply", tool_calls=[])
    assert len(fake.calls) == 1


def test_provider_agent_resolves_tool_calls_with_a_stub_and_asks_again():
    call = ToolCall(id="call_1", name="lookup_order", input_parameters={"order_id": "123"})
    fake = _FakeToolCallingProvider([("", [call]), ("Your order shipped.", [])])
    agent = ProviderAgent(fake, make_agent_spec())

    result = agent.respond([Message(role="user", content="Where's my order?")])

    assert result.final_reply == "Your order shipped."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].output == _STUB_TOOL_RESULT
    assert len(fake.calls) == 2
    assert [m.role for m in fake.calls[1][-2:]] == ["assistant", "tool"]


def test_provider_agent_stops_after_max_tool_rounds():
    call = ToolCall(id="call_1", name="lookup_order", input_parameters={})
    fake = _FakeToolCallingProvider([("", [call])] * 2)
    agent = ProviderAgent(fake, make_agent_spec(), max_tool_rounds=2)

    result = agent.respond([Message(role="user", content="hi")])

    assert len(fake.calls) == 2
    assert len(result.tool_calls) == 2  # one per round, never actually executed
    assert result.final_reply == ""


def test_tool_schemas_declares_an_open_object_per_tool():
    spec = make_agent_spec()
    schemas = _tool_schemas(spec)

    assert len(schemas) == len(spec.tools)
    assert schemas[0]["function"]["name"] == spec.tools[0].name
    assert schemas[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


# --- SubprocessAgent -------------------------------------------------------


def _fake_completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_subprocess_agent_is_a_target_agent():
    assert isinstance(SubprocessAgent(command=["true"]), TargetAgent)


def test_subprocess_agent_posts_conversation_json_and_returns_reply(monkeypatch):
    mock_run = MagicMock(return_value=_fake_completed_process(json.dumps({"reply": "hi back"})))
    monkeypatch.setattr("agent_stress_test.targets.subprocess_agent.subprocess.run", mock_run)
    agent = SubprocessAgent(command=["fake-agent"], timeout=5.0)

    result = agent.respond([Message(role="user", content="hello")])

    assert result == AgentResponse(final_reply="hi back", trace=None)
    _, kwargs = mock_run.call_args
    assert json.loads(kwargs["input"]) == {"messages": [{"role": "user", "content": "hello"}]}
    assert kwargs["timeout"] == 5.0


def test_subprocess_agent_passes_through_trace_when_present(monkeypatch):
    trace_payload = [{"thought": "checked the order", "action": "lookup_order"}]
    mock_run = MagicMock(
        return_value=_fake_completed_process(
            json.dumps({"reply": "shipped", "trace": trace_payload})
        )
    )
    monkeypatch.setattr("agent_stress_test.targets.subprocess_agent.subprocess.run", mock_run)
    agent = SubprocessAgent(command=["fake-agent"])

    result = agent.respond([Message(role="user", content="hi")])

    assert result.trace == [Step(thought="checked the order", action="lookup_order")]


def test_subprocess_agent_raises_on_non_zero_exit(monkeypatch):
    mock_run = MagicMock(return_value=_fake_completed_process("", returncode=1, stderr="boom"))
    monkeypatch.setattr("agent_stress_test.targets.subprocess_agent.subprocess.run", mock_run)
    agent = SubprocessAgent(command=["fake-agent"])

    with pytest.raises(RuntimeError, match="boom"):
        agent.respond([Message(role="user", content="hi")])


# --- Capabilities (A4) ------------------------------------------------------


@pytest.mark.parametrize(
    "build_agent",
    [
        lambda: SampleAgent(make_agent_spec(), FakeLLMProvider()),
        lambda: HttpAgent(url="http://example.test/respond"),
        lambda: PythonFunctionAgent(lambda conversation: "reply"),
        lambda: SubprocessAgent(command=["true"]),
    ],
    ids=["sample_agent", "http_agent", "python_function_agent", "subprocess_agent"],
)
def test_agent_capabilities_default_to_all_false(build_agent):
    assert build_agent().capabilities() == Capabilities()


def test_python_function_agent_capabilities_can_be_declared_explicitly():
    agent = PythonFunctionAgent(lambda conversation: "reply", capabilities=Capabilities(tools=True))
    assert agent.capabilities() == Capabilities(tools=True)


def test_provider_agent_reports_real_tool_calling_capability():
    agent = ProviderAgent(_FakeToolCallingProvider([("hi", [])]), make_agent_spec())
    assert agent.capabilities() == Capabilities(tools=True)


# --- tool_calling_verification_agent (A4 verification target) --------------


def test_tool_calling_verification_agent_calls_lookup_order_with_a_wrong_id():
    conversation = [Message(role="user", content="Where is order 12345?")]

    result = tool_calling_verification_agent(conversation)

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "lookup_order"
    assert call.input_parameters["order_id"] != "12345"


def test_tool_calling_verification_agent_is_deterministic():
    conversation = [Message(role="user", content="Where is order 12345?")]

    first = tool_calling_verification_agent(conversation)
    second = tool_calling_verification_agent(conversation)

    assert first.tool_calls[0].input_parameters == second.tool_calls[0].input_parameters


def test_tool_calling_verification_agent_falls_back_when_no_order_id_present():
    result = tool_calling_verification_agent([Message(role="user", content="Hi there")])
    assert result.tool_calls[0].input_parameters["order_id"] == "00001"


def test_build_target_from_spec_wires_the_tool_calling_verification_agent():
    spec = make_agent_spec(
        target={
            "kind": "python",
            "import_path": (
                "agent_stress_test.targets.tool_calling_verification_agent:"
                "tool_calling_verification_agent"
            ),
        }
    )

    target = _build_target_from_spec(spec, FakeLLMProvider())
    result = target.respond([Message(role="user", content="Where is order 98765?")])

    assert result.tool_calls[0].name == "lookup_order"
    assert result.tool_calls[0].input_parameters["order_id"] != "98765"
