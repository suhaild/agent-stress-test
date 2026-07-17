from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_stress_test.models import Message, TextBlock, ToolCall, ToolResultBlock, ToolUseBlock
from agent_stress_test.providers.litellm_provider import LiteLLMProvider


def make_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def mock_completion(monkeypatch):
    mock = MagicMock(return_value=make_response("mocked reply"))
    monkeypatch.setattr(
        "agent_stress_test.providers.litellm_provider.litellm.completion", mock
    )
    return mock


def test_complete_maps_messages_and_returns_content(mock_completion):
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")
    messages = [
        Message(role="system", content="be nice"),
        Message(role="user", content="hello"),
    ]

    result = provider.complete(messages)

    assert result == "mocked reply"
    mock_completion.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ],
    )


def test_sample_n_calls_complete_n_times(mock_completion):
    mock_completion.side_effect = [
        make_response("reply-1"),
        make_response("reply-2"),
        make_response("reply-3"),
        make_response("reply-4"),
    ]
    provider = LiteLLMProvider(model="gpt-4o")
    messages = [Message(role="user", content="probe")]

    results = provider.sample_n(messages, 4)

    # sample_n's contract is "n independent completions" — the samples run
    # concurrently, so submission order isn't preserved, only the full set.
    assert sorted(results) == ["reply-1", "reply-2", "reply-3", "reply-4"]
    assert mock_completion.call_count == 4


def test_sample_n_rejects_non_positive_n(mock_completion):
    provider = LiteLLMProvider(model="gpt-4o")
    with pytest.raises(ValueError):
        provider.sample_n([Message(role="user", content="hi")], 0)


def test_cache_flagged_message_gets_cache_control(mock_completion):
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")
    messages = [
        Message(role="system", content="be nice", cache=True),
        Message(role="user", content="hello"),
    ]

    provider.complete(messages)

    mock_completion.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "system",
                "content": "be nice",
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "hello"},
        ],
    )


def test_block_content_message_translates_to_litellm_content_blocks(mock_completion):
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")
    messages = [
        Message(
            role="assistant",
            content=[
                TextBlock(text="let me check"),
                ToolUseBlock(id="call_1", name="lookup_order", input={"order_id": "123"}),
            ],
        ),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="call_1", content="shipped")]),
    ]

    provider.complete(messages)

    mock_completion.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me check"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "lookup_order",
                        "input": {"order_id": "123"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "shipped",
                        "is_error": False,
                    }
                ],
            },
        ],
    )


# --- complete_with_tools ---------------------------------------------------


def make_tool_call_response(content, tool_calls):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def make_litellm_tool_call(call_id, name, arguments_json):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments_json)
    )


def test_complete_with_tools_passes_tools_through_and_returns_text_when_no_tool_call(
    mock_completion,
):
    mock_completion.return_value = make_tool_call_response("plain text reply", None)
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")
    tools = [{"type": "function", "function": {"name": "lookup_order", "description": "..."}}]

    text, tool_calls = provider.complete_with_tools([Message(role="user", content="hi")], tools)

    assert text == "plain text reply"
    assert tool_calls == []
    mock_completion.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )


def test_complete_with_tools_parses_tool_calls_from_the_response(mock_completion):
    mock_completion.return_value = make_tool_call_response(
        None,
        [make_litellm_tool_call("call_1", "lookup_order", '{"order_id": "123"}')],
    )
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")

    text, tool_calls = provider.complete_with_tools([Message(role="user", content="hi")], [])

    assert text == ""
    assert tool_calls == [
        ToolCall(id="call_1", name="lookup_order", input_parameters={"order_id": "123"})
    ]


def test_complete_with_tools_treats_empty_arguments_as_no_input(mock_completion):
    mock_completion.return_value = make_tool_call_response(
        None, [make_litellm_tool_call("call_1", "ping", "")]
    )
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")

    _text, tool_calls = provider.complete_with_tools([Message(role="user", content="hi")], [])

    assert tool_calls[0].input_parameters == {}


# --- Usage metering (A5) ----------------------------------------------------


def make_response_with_usage(content: str, *, prompt_tokens, completion_tokens, total_tokens):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def test_complete_records_real_usage_and_cost(monkeypatch, mock_completion):
    mock_completion.return_value = make_response_with_usage(
        "hi", prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    monkeypatch.setattr(
        "agent_stress_test.providers.litellm_provider.litellm.completion_cost",
        lambda completion_response: 0.0042,
    )
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")

    provider.complete([Message(role="user", content="hi")])

    usage = provider.meter.total()
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.cost_usd == pytest.approx(0.0042)
    assert usage.pricing_unavailable is False


def test_complete_usage_accumulates_across_calls(monkeypatch, mock_completion):
    mock_completion.return_value = make_response_with_usage(
        "hi", prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    monkeypatch.setattr(
        "agent_stress_test.providers.litellm_provider.litellm.completion_cost",
        lambda completion_response: 0.01,
    )
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")

    provider.complete([Message(role="user", content="hi")])
    provider.complete([Message(role="user", content="hi")])

    usage = provider.meter.total()
    assert usage.total_tokens == 30
    assert usage.cost_usd == pytest.approx(0.02)


def test_complete_marks_pricing_unavailable_when_completion_cost_raises(
    monkeypatch, mock_completion
):
    mock_completion.return_value = make_response_with_usage(
        "hi", prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    def _raise(completion_response):
        raise Exception("This model isn't mapped yet")

    monkeypatch.setattr(
        "agent_stress_test.providers.litellm_provider.litellm.completion_cost", _raise
    )
    provider = LiteLLMProvider(model="some-unrecognized-model-id")

    provider.complete([Message(role="user", content="hi")])

    usage = provider.meter.total()
    # Token counts are still captured even though pricing lookup failed —
    # only the cost itself is unknown, never fabricated as 0.0-and-silent.
    assert usage.total_tokens == 15
    assert usage.cost_usd == 0.0
    assert usage.pricing_unavailable is True


def test_complete_with_no_usage_on_the_response_records_zero_without_crashing(mock_completion):
    # The default mock_completion fixture's response has no `.usage` at all —
    # exercises the defensive getattr path, not just the happy path above.
    provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")

    provider.complete([Message(role="user", content="hi")])

    usage = provider.meter.total()
    assert usage.total_tokens == 0
