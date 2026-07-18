import json

import pytest
from deepeval.metrics.argument_correctness.schema import (
    ArgumentCorrectnessScoreReason,
    ArgumentCorrectnessVerdict,
    Verdicts,
)
from deepeval.metrics.g_eval.schema import BestTestCase, ReasonScore, Steps
from deepeval.metrics.tool_use.schema import (
    ArgumentCorrectnessScore,
    Reason,
    ToolSelectionScore,
    UserInputAndTools,
)
from deepeval.simulator.schema import ConversationCompletion, EdgeChoice, SimulatedInput
from pydantic import BaseModel

from agent_stress_test.models import Message
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM, fabricate_from_json_schema
from agent_stress_test.reasoning.deepeval_bridge import SCHEMA_MARKER, LLMProviderAsDeepEvalLLM

# Every DeepEval schema the ConversationSimulator / GEval / ArgumentCorrectness
# flow actually constructs — see test_deepeval_bridge.py's GO check.
DEEPEVAL_SCHEMAS = [
    ArgumentCorrectnessVerdict,
    Verdicts,
    ArgumentCorrectnessScoreReason,
    ReasonScore,
    Steps,
    BestTestCase,
    ConversationCompletion,
    SimulatedInput,
    EdgeChoice,
    ArgumentCorrectnessScore,
    UserInputAndTools,
    ToolSelectionScore,
    Reason,
]


# --- fabricate_from_json_schema: the four rules + the required extras ----


class _Flags(BaseModel):
    is_complete: bool
    stopped: bool
    hit_end: bool
    refused: bool
    jailbroken_flag: bool
    tools_used: bool


def test_bool_fields_are_false_only_when_the_name_signals_completion():
    fabricated = fabricate_from_json_schema(_Flags.model_json_schema())

    assert fabricated == {
        "is_complete": False,
        "stopped": False,
        "hit_end": False,
        "refused": False,
        "jailbroken_flag": False,
        "tools_used": True,
    }
    _Flags.model_validate(fabricated)  # no crash


class _Shapes(BaseModel):
    names: list[str]
    count: int
    ratio: float
    label: str


def test_list_int_float_and_plain_string_rules():
    fabricated = fabricate_from_json_schema(_Shapes.model_json_schema())

    assert fabricated == {"names": [], "count": 7, "ratio": 7, "label": "plausible text"}
    _Shapes.model_validate(fabricated)


def test_enum_field_fabricates_its_first_allowed_value():
    from typing import Literal

    class Choice(BaseModel):
        verdict: Literal["yes", "no", "idk"]

    fabricated = fabricate_from_json_schema(Choice.model_json_schema())

    assert fabricated == {"verdict": "yes"}
    Choice.model_validate(fabricated)


class _Optional(BaseModel):
    maybe_count: int | None = None
    maybe_name: str | None = None


def test_anyof_optional_fields_resolve_to_the_non_null_branch():
    fabricated = fabricate_from_json_schema(_Optional.model_json_schema())

    assert fabricated == {"maybe_count": 7, "maybe_name": "plausible text"}
    _Optional.model_validate(fabricated)


class _Item(BaseModel):
    label: str


class _Nested(BaseModel):
    items: list[_Item]


def test_nested_model_list_field_still_fabricates_to_an_empty_list():
    fabricated = fabricate_from_json_schema(_Nested.model_json_schema())

    assert fabricated == {"items": []}
    _Nested.model_validate(fabricated)  # $ref inside `items` is never touched


@pytest.mark.parametrize("schema_cls", DEEPEVAL_SCHEMAS, ids=lambda c: c.__name__)
def test_every_deepeval_schema_fabricates_a_valid_instance(schema_cls):
    fabricated = fabricate_from_json_schema(schema_cls.model_json_schema())
    instance = schema_cls.model_validate(fabricated)
    assert isinstance(instance, schema_cls)


# --- ShapedFakeLLM ----------------------------------------------------------


def test_shaped_fake_falls_back_to_plain_fake_reply_when_no_schema_marker():
    provider = ShapedFakeLLM()

    reply = provider.complete([Message(role="user", content="hello there")])

    assert reply == "fake-reply: hello there"


def test_shaped_fake_fabricates_schema_valid_json_when_marker_present():
    provider = ShapedFakeLLM()
    schema_json = json.dumps(ConversationCompletion.model_json_schema())
    prompt = f"some prompt{SCHEMA_MARKER}{schema_json}"

    reply = provider.complete([Message(role="user", content=prompt)])

    parsed = ConversationCompletion.model_validate_json(reply)
    assert parsed.is_complete is False


def test_shaped_fake_records_calls_and_usage():
    provider = ShapedFakeLLM()

    provider.complete([Message(role="user", content="probe")])

    assert len(provider.calls) == 1
    assert provider.meter.total().total_tokens > 0
    assert provider.meter.total().cost_usd == 0.0


def test_shaped_fake_sample_n_rejects_non_positive_n():
    provider = ShapedFakeLLM()
    with pytest.raises(ValueError):
        provider.sample_n([Message(role="user", content="hi")], 0)


# --- LLMProviderAsDeepEvalLLM + ShapedFakeLLM, wired together --------------


def test_shim_generate_with_schema_returns_a_valid_instance_via_the_shaped_fake():
    shim = LLMProviderAsDeepEvalLLM(ShapedFakeLLM())

    result = shim.generate("Please respond.", schema=ConversationCompletion)

    assert isinstance(result, ConversationCompletion)
    assert result.is_complete is False
