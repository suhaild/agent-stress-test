import pytest
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn

from agent_stress_test.models import AgentResponse, Message, ToolCall
from agent_stress_test.ports import TargetAgent
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.deepeval_simulator import (
    PERSONAS,
    from_deepeval_tool_call,
    make_model_callback,
    simulate_personas,
    to_deepeval_tool_call,
)
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.targets.python_fn import PythonFunctionAgent


# --- PERSONAS: the 5 tactics, re-expressed -----------------------------


def test_personas_match_the_five_legacy_tactic_names_exactly():
    assert set(PERSONAS) == set(default_registry().names())


def test_every_persona_has_a_scenario_and_user_description():
    for name, golden in PERSONAS.items():
        assert golden.scenario, f"{name} has no scenario"
        assert golden.user_description, f"{name} has no user_description"


def test_personas_never_set_an_expected_outcome():
    # Leaving expected_outcome unset means DeepEval's default stopping
    # controller always returns "not complete" (see
    # deepeval.simulator.controller.controller.check_expected_outcome) —
    # confirmed against the installed version, not assumed — so a
    # simulated conversation reliably runs for exactly max_user_simulations
    # rounds every time, never stopping early.
    for golden in PERSONAS.values():
        assert golden.expected_outcome is None


# --- ToolCall <-> DeepEval ToolCall conversion --------------------------


def test_to_deepeval_tool_call_round_trips_name_input_and_output():
    call = ToolCall(
        id="call_1", name="lookup_order", input_parameters={"order_id": "123"}, output="shipped"
    )

    converted = to_deepeval_tool_call(call)

    assert converted.name == "lookup_order"
    assert converted.input_parameters == {"order_id": "123"}
    assert converted.output == "shipped"


def test_from_deepeval_tool_call_round_trips_name_input_and_output():
    deepeval_call = to_deepeval_tool_call(
        ToolCall(id="x", name="lookup_order", input_parameters={"order_id": "999"}, output="n/a")
    )

    converted = from_deepeval_tool_call(deepeval_call)

    assert converted.name == "lookup_order"
    assert converted.input_parameters == {"order_id": "999"}
    assert converted.output == "n/a"


# --- make_model_callback -------------------------------------------------


def test_model_callback_returns_a_turn_not_a_bare_string():
    target = PythonFunctionAgent(lambda conversation: "a target reply")
    callback = make_model_callback(target)

    result = callback([Turn(role="user", content="hi")])

    assert isinstance(result, Turn)
    assert result.role == "assistant"
    assert result.content == "a target reply"
    assert result.tools_called == []


def test_model_callback_converts_turns_to_messages_for_the_target():
    received: list[list[Message]] = []

    def fn(conversation: list[Message]) -> str:
        received.append(conversation)
        return "ok"

    callback = make_model_callback(PythonFunctionAgent(fn))
    turns = [
        Turn(role="user", content="first"),
        Turn(role="assistant", content="reply"),
        Turn(role="user", content="second"),
    ]

    callback(turns)

    assert received[0] == [
        Message(role="user", content="first"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="second"),
    ]


def test_model_callback_carries_tool_calls_onto_the_turn():
    class _ToolCallingTarget(TargetAgent):
        def respond(self, conversation: list[Message]) -> AgentResponse:
            return AgentResponse(
                final_reply="looked it up",
                tool_calls=[ToolCall(id="c1", name="lookup_order", input_parameters={})],
            )

    callback = make_model_callback(_ToolCallingTarget())

    result = callback([Turn(role="user", content="hi")])

    assert result.tools_called is not None
    assert len(result.tools_called) == 1
    assert result.tools_called[0].name == "lookup_order"


# --- simulate_personas: an offline run produces a multi-turn -------------
# --- ConversationalTestCase (the B2 mandatory test) ----------------------


def test_simulate_personas_produces_a_multi_turn_conversational_test_case():
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")

    test_cases = simulate_personas(
        target=target,
        sim_provider=ShapedFakeLLM(),
        persona_names=["hostile"],
        max_user_simulations=2,
    )

    assert len(test_cases) == 1
    turns = test_cases[0].turns
    assert len(turns) == 4  # 2 user turns + 2 assistant turns
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    assert all(t.content for t in turns)


def test_simulate_personas_runs_every_requested_persona_in_order():
    target = PythonFunctionAgent(lambda conversation: "ok")

    test_cases = simulate_personas(
        target=target,
        sim_provider=ShapedFakeLLM(),
        persona_names=["self-contradiction", "hostile"],
        max_user_simulations=1,
    )

    assert len(test_cases) == 2
    for test_case in test_cases:
        assert len(test_case.turns) == 2


def test_simulate_personas_with_a_personas_override_resolves_against_it_not_the_bundled_dict():
    # B4 profiler-consumption wiring: a caller can hand in a persona registry
    # that isn't the bundled PERSONAS dict at all — e.g. a merged
    # bundled+profile dict built by orchestration/deepeval_search.py.
    target = PythonFunctionAgent(lambda conversation: "ok")
    custom_golden = ConversationalGolden(
        scenario="A patient downplays a serious symptom.",
        user_description="A patient who minimizes their symptoms.",
    )

    test_cases = simulate_personas(
        target=target,
        sim_provider=ShapedFakeLLM(),
        persona_names=["symptom-minimizer"],
        max_user_simulations=1,
        personas={"symptom-minimizer": custom_golden},
    )

    assert len(test_cases) == 1
    assert len(test_cases[0].turns) == 2


def test_simulate_personas_override_still_raises_on_an_unknown_name():
    target = PythonFunctionAgent(lambda conversation: "ok")

    with pytest.raises(KeyError):
        simulate_personas(
            target=target,
            sim_provider=ShapedFakeLLM(),
            persona_names=["not-a-real-persona"],
            max_user_simulations=1,
            personas={"symptom-minimizer": PERSONAS["hostile"]},
        )
