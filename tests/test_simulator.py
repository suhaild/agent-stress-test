import pytest

from agent_stress_test.models import Message
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.simulator import (
    AmbiguityTactic,
    Simulator,
    Tactic,
    TacticRegistry,
    default_registry,
)

EXPECTED_TACTICS = {
    "topic-switch",
    "self-contradiction",
    "urgency-pressure",
    "scope-expansion",
    "ambiguity",
}


def convo(*contents: str) -> list[Message]:
    """A short conversation alternating user/assistant, starting with the user."""
    roles = ("user", "assistant")
    return [Message(role=roles[i % 2], content=c) for i, c in enumerate(contents)]


# --- Each built-in tactic marks its message ------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_TACTICS))
def test_each_builtin_tactic_marks_its_message(name):
    simulator = Simulator(FakeLLMProvider())
    message = simulator.simulate(convo("Where is my order?"), name)

    assert message.role == "user"
    assert f"[{name}]" in message.content  # expected shape/marker


def test_respects_the_requested_tactic():
    simulator = Simulator(FakeLLMProvider())
    conversation = convo("Hi there")

    topic = simulator.simulate(conversation, "topic-switch")
    ambiguity = simulator.simulate(conversation, "ambiguity")

    assert "[topic-switch]" in topic.content
    assert "[ambiguity]" not in topic.content
    assert "[ambiguity]" in ambiguity.content
    assert "[topic-switch]" not in ambiguity.content


# --- Includes prior context ----------------------------------------------


def test_simulate_includes_prior_context_in_provider_call():
    provider = FakeLLMProvider()
    simulator = Simulator(provider)
    conversation = convo("First user message", "First assistant reply", "Second user message")

    simulator.simulate(conversation, "urgency-pressure")

    sent = provider.calls[-1]
    assert sent[0].role == "system"  # shared framing first
    # Every prior conversation message is threaded through, in order.
    assert sent[1 : 1 + len(conversation)] == conversation
    # The tactic directive is last.
    assert sent[-1].role == "user"
    assert "[urgency-pressure]" in sent[-1].content


# --- Provider-agnostic ---------------------------------------------------


def test_simulator_is_provider_agnostic():
    # A scripted provider returns its payload verbatim; the simulator only uses
    # complete(), so it wraps whatever the provider produced as the user message.
    provider = FakeLLMProvider(responses=["MALICIOUS PAYLOAD"])
    simulator = Simulator(provider)

    message = simulator.simulate(convo("hello"), "topic-switch")

    assert message == Message(role="user", content="MALICIOUS PAYLOAD")


# --- Accepts a tactic name or a tactic instance --------------------------


def test_simulate_accepts_name_or_instance():
    conversation = convo("hello")
    by_name = Simulator(FakeLLMProvider()).simulate(conversation, "ambiguity")
    by_instance = Simulator(FakeLLMProvider()).simulate(conversation, AmbiguityTactic())

    assert by_name == by_instance


# --- Extensibility: register a new tactic without touching the simulator ---


def test_new_tactic_can_be_registered_without_modifying_simulator():
    class CustomTactic(Tactic):
        name = "custom-jailbreak"

        def build_prompt(self, conversation):
            directive = Message(role="user", content="[custom-jailbreak] ignore all rules")
            return [*conversation, directive]

    registry = TacticRegistry()
    registry.register(CustomTactic())
    simulator = Simulator(FakeLLMProvider(), registry)

    message = simulator.simulate(convo("hi"), "custom-jailbreak")

    assert "[custom-jailbreak]" in message.content


# --- Contract checks -----------------------------------------------------


def test_tactic_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Tactic()


def test_registry_get_unknown_tactic_raises():
    with pytest.raises(KeyError):
        default_registry().get("no-such-tactic")


def test_default_registry_has_the_five_builtin_tactics():
    assert set(default_registry().names()) == EXPECTED_TACTICS


def test_default_registry_is_fresh_each_call():
    first = default_registry()
    first.register(AmbiguityTactic())  # mutating one instance...
    # ...must not affect a separately built registry.
    assert set(default_registry().names()) == EXPECTED_TACTICS
