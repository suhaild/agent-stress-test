import asyncio

from deepeval.dataset.golden import ConversationalGolden
from deepeval.metrics import ArgumentCorrectnessMetric, GEval
from deepeval.simulator.conversation_simulator import ConversationSimulator
from deepeval.simulator.schema import ConversationCompletion
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall, Turn

from agent_stress_test.models import Message
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.deepeval_bridge import LLMProviderAsDeepEvalLLM


def test_load_model_returns_the_wrapped_provider():
    provider = FakeLLMProvider()
    shim = LLMProviderAsDeepEvalLLM(provider)

    assert shim.load_model() is provider
    assert shim.model is provider  # DeepEvalBaseLLM.__init__ calls load_model()


def test_get_model_name_returns_a_fixed_string():
    shim = LLMProviderAsDeepEvalLLM(FakeLLMProvider())
    assert shim.get_model_name() == shim.get_model_name()
    assert isinstance(shim.get_model_name(), str)
    assert shim.get_model_name() != ""


def test_generate_without_schema_returns_the_providers_raw_text():
    provider = FakeLLMProvider()
    shim = LLMProviderAsDeepEvalLLM(provider)

    result = shim.generate("hello")

    assert result == provider.complete([Message(role="user", content="hello")])


def test_generate_with_schema_returns_a_validated_model_via_the_shaped_fake():
    shim = LLMProviderAsDeepEvalLLM(ShapedFakeLLM())

    result = shim.generate("Is this conversation done?", schema=ConversationCompletion)

    assert isinstance(result, ConversationCompletion)
    assert result.is_complete is False


# --- Fence-stripping (Phase C3): real models wrap JSON in ``` fences -------


def test_generate_with_schema_strips_a_markdown_code_fence():
    # Found live (Phase C3's cost-measurement run against real Haiku): every
    # schema-constrained DeepEval call funnels through this one shim, and a
    # real Claude call commonly wraps its JSON reply in a ```json fence --
    # model_validate_json() can't parse that directly. Before this was
    # fixed, every live call silently fell back to judge.py's "malformed
    # output, default to pass" path instead of ever actually judging.
    fenced = '```json\n{"is_complete": false, "reason": "not done"}\n```'
    provider = FakeLLMProvider(responses=[fenced])
    shim = LLMProviderAsDeepEvalLLM(provider)

    result = shim.generate("Is this conversation done?", schema=ConversationCompletion)

    assert isinstance(result, ConversationCompletion)
    assert result.is_complete is False


def test_generate_with_schema_strips_a_bare_triple_backtick_fence_with_no_language_tag():
    provider = FakeLLMProvider(responses=['```\n{"is_complete": true, "reason": "done"}\n```'])
    shim = LLMProviderAsDeepEvalLLM(provider)

    result = shim.generate("Is this conversation done?", schema=ConversationCompletion)

    assert result.is_complete is True


def test_generate_with_schema_leaves_unfenced_json_unchanged():
    # Every offline fake (ShapedFakeLLM/FakeLLMProvider) already returns raw,
    # unfenced JSON -- confirm fence-stripping is a no-op passthrough there,
    # not something that could corrupt an already-valid response.
    provider = FakeLLMProvider(responses=['{"is_complete": false, "reason": "not done"}'])
    shim = LLMProviderAsDeepEvalLLM(provider)

    result = shim.generate("Is this conversation done?", schema=ConversationCompletion)

    assert result.is_complete is False


def test_a_generate_awaited_returns_the_same_as_generate():
    shim = LLMProviderAsDeepEvalLLM(ShapedFakeLLM())

    result = asyncio.run(
        shim.a_generate("Is this conversation done?", schema=ConversationCompletion)
    )

    assert isinstance(result, ConversationCompletion)
    assert result.is_complete is False


def test_a_generate_without_schema_matches_generate_without_schema():
    provider = FakeLLMProvider(responses=["scripted reply"])
    shim = LLMProviderAsDeepEvalLLM(provider)

    result = asyncio.run(shim.a_generate("hi"))

    assert result == "scripted reply"


# --- GO check: the spike flow, fully offline (B1 checkpoint) ---------------


def _model_callback(input: str) -> Turn:
    return Turn(role="assistant", content=f"Target reply to: {input}")


def test_go_check_conversation_simulator_geval_and_argument_correctness_run_offline():
    """Re-runs the spike flow end to end against the schema-aware fake: no
    real API call, no crash. This is the mandatory GO check before any
    metric-backed reasoning code gets built on top of these two pieces.
    """
    shim = LLMProviderAsDeepEvalLLM(ShapedFakeLLM())

    simulator = ConversationSimulator(
        model_callback=_model_callback,
        simulator_model=shim,
        async_mode=False,
        max_concurrent=1,
    )
    golden = ConversationalGolden(
        scenario="A user asking about order status.",
        user_description="A customer checking on a recent order.",
    )
    test_cases = simulator.simulate([golden], max_user_simulations=2)

    assert len(test_cases) == 1
    # Bool semantics-awareness is load-bearing here: if ConversationCompletion
    # fabricated is_complete=True, the simulation would stop after turn 0 —
    # 2 user turns confirms it kept going the requested number of rounds.
    assert len(test_cases[0].turns) == 4

    test_case = LLMTestCase(
        input="Where is my order?",
        actual_output="Your order shipped yesterday.",
        tools_called=[ToolCall(name="lookup_order", input_parameters={"order_id": "123"})],
    )

    geval = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct.",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=shim,
        async_mode=False,
    )
    geval_score = geval.measure(test_case)
    assert 0.0 <= geval_score <= 1.0
    assert geval.reason

    arg_metric = ArgumentCorrectnessMetric(model=shim, async_mode=False)
    arg_score = arg_metric.measure(test_case)
    assert 0.0 <= arg_score <= 1.0
    assert arg_metric.reason
