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


# --- Usage metering (A5) ----------------------------------------------------


def test_complete_records_a_deterministic_token_count_at_zero_cost():
    provider = FakeLLMProvider()

    provider.complete([Message(role="user", content="two tokens here")])

    usage = provider.meter.total()
    assert usage.total_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.cost_usd == 0.0
    assert usage.pricing_unavailable is False


def test_complete_token_counts_accumulate_across_calls():
    provider = FakeLLMProvider()
    messages = [Message(role="user", content="a b c")]

    provider.complete(messages)
    after_one = provider.meter.total().total_tokens
    provider.complete(messages)
    after_two = provider.meter.total().total_tokens

    assert after_two == after_one * 2


def test_sample_n_records_usage_for_every_sample():
    provider = FakeLLMProvider()

    provider.sample_n([Message(role="user", content="probe")], 3)

    usage = provider.meter.total()
    assert usage.total_tokens > 0
    single_call_provider = FakeLLMProvider()
    single_call_provider.complete([Message(role="user", content="probe")])
    assert usage.total_tokens == 3 * single_call_provider.meter.total().total_tokens
