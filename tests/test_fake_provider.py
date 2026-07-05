import pytest

from agent_stress_test.models import Message
from agent_stress_test.providers.fake import FakeLLMProvider


def test_default_mode_is_pure_function_of_last_message():
    messages = [Message(role="user", content="break the agent")]
    reply1 = FakeLLMProvider().complete(messages)
    reply2 = FakeLLMProvider().complete(messages)
    assert reply1 == reply2
    assert "break the agent" in reply1


def test_default_mode_varies_with_input():
    provider = FakeLLMProvider()
    reply_a = provider.complete([Message(role="user", content="A")])
    reply_b = provider.complete([Message(role="user", content="B")])
    assert reply_a != reply_b


def test_scripted_mode_returns_in_order():
    provider = FakeLLMProvider(responses=["first", "second"])
    messages = [Message(role="user", content="hi")]
    assert provider.complete(messages) == "first"
    assert provider.complete(messages) == "second"


def test_scripted_mode_raises_when_exhausted():
    provider = FakeLLMProvider(responses=["only-one"])
    messages = [Message(role="user", content="hi")]
    provider.complete(messages)
    with pytest.raises(IndexError):
        provider.complete(messages)


def test_scripted_mode_cycles_when_requested():
    provider = FakeLLMProvider(responses=["a", "b"], cycle=True)
    messages = [Message(role="user", content="hi")]
    seen = [provider.complete(messages) for _ in range(4)]
    assert seen == ["a", "b", "a", "b"]


def test_sample_n_returns_n_items_and_logs_calls():
    provider = FakeLLMProvider()
    messages = [Message(role="user", content="probe")]
    results = provider.sample_n(messages, 3)
    assert len(results) == 3
    assert len(provider.calls) == 3


def test_sample_n_rejects_non_positive_n():
    provider = FakeLLMProvider()
    with pytest.raises(ValueError):
        provider.sample_n([Message(role="user", content="hi")], 0)
