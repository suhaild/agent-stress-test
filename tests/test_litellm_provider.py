from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_stress_test.models import Message
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

    assert results == ["reply-1", "reply-2", "reply-3", "reply-4"]
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
