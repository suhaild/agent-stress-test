import pytest

from agent_stress_test.ports import LLMProvider, Store, TargetAgent
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.litellm_provider import LiteLLMProvider


def test_llm_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMProvider()


def test_target_agent_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        TargetAgent()


def test_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Store()


def test_fake_provider_is_an_llm_provider():
    assert isinstance(FakeLLMProvider(), LLMProvider)


def test_litellm_provider_is_an_llm_provider():
    assert isinstance(LiteLLMProvider(model="claude-3-5-sonnet-20241022"), LLMProvider)
