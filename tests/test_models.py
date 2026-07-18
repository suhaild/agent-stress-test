import pytest
from pydantic import ValidationError

from agent_stress_test.models import (
    AgentResponse,
    AgentSpec,
    Cluster,
    ImageBlock,
    ImageSource,
    Message,
    Node,
    Rule,
    Run,
    RunUsage,
    Step,
    TextBlock,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Verdict,
)
from tests.conftest import make_agent_spec


def test_message_valid():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_message_str_content_round_trips():
    msg = Message(role="user", content="hello")
    reloaded = Message.model_validate_json(msg.model_dump_json())
    assert reloaded == msg


def test_message_accepts_all_content_block_types():
    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="let me check that"),
            ImageBlock(source=ImageSource(type="url", url="https://example.com/x.png")),
            ToolUseBlock(id="call_1", name="lookup_order", input={"order_id": "123"}),
            ToolResultBlock(tool_use_id="call_1", content="shipped", is_error=False),
        ],
    )
    reloaded = Message.model_validate_json(msg.model_dump_json())
    assert reloaded == msg
    assert [block.type for block in reloaded.content] == [
        "text",
        "image",
        "tool_use",
        "tool_result",
    ]


def test_message_accepts_tool_role():
    msg = Message(
        role="tool",
        content=[ToolResultBlock(tool_use_id="call_1", content="shipped")],
    )
    assert msg.role == "tool"


@pytest.mark.parametrize(
    "model_cls,bad_kwargs",
    [
        pytest.param(
            Message,
            dict(role="assistant", content=[{"type": "bogus", "text": "hi"}]),
            id="message-unknown-content-block-type",
        ),
        pytest.param(
            Message,
            dict(role="assistant", content=[{"type": "tool_use", "id": "call_1"}]),  # no name
            id="message-content-block-missing-required-field",
        ),
        pytest.param(
            Message, dict(role="wizard", content="hello"), id="message-invalid-role"
        ),
        pytest.param(
            Rule,
            dict(id="r1", text="Be nice.", severity="catastrophic"),
            id="rule-bad-severity",
        ),
        pytest.param(
            AgentSpec,
            dict(name="x", system_prompt="", tools=[], rules=[Rule(id="r", text="a rule")]),
            id="agent-spec-missing-system-prompt",
        ),
        pytest.param(
            AgentSpec,
            dict(name="x", system_prompt="hi", tools=[], rules=[]),
            id="agent-spec-empty-rules",
        ),
        pytest.param(
            Verdict,
            dict(
                run_id="run-1",
                node_id="node-1",
                passed=True,
                reason="fine",
                tier="tier-3-llm-council",
                confidence=1.0,
                severity="minor",
            ),
            id="verdict-bad-tier",
        ),
        pytest.param(
            Verdict,
            dict(
                run_id="run-1",
                node_id="node-1",
                passed=True,
                reason="fine",
                tier="rules",
                confidence=1.5,
                severity="minor",
            ),
            id="verdict-confidence-out-of-range",
        ),
        pytest.param(
            Verdict,
            dict(
                run_id="run-1",
                node_id="node-1",
                passed=True,
                reason="fine",
                tier="rules",
                confidence=1.0,
                severity="apocalyptic",
            ),
            id="verdict-bad-severity",
        ),
    ],
)
def test_model_rejects_invalid_input(model_cls, bad_kwargs):
    with pytest.raises(ValidationError):
        model_cls(**bad_kwargs)


def test_agent_spec_valid():
    spec = make_agent_spec(
        tools=[ToolSpec(name="lookup", description="Looks things up.")],
        rules=[Rule(id="no-lie", text="Never lie.", severity="major")],
    )
    assert spec.name == "test_agent"
    assert len(spec.tools) == 1
    assert spec.rules[0].text == "Never lie."
    assert spec.rules[0].severity == "major"


def test_rule_defaults_severity_to_major():
    rule = Rule(id="r1", text="Be nice.")
    assert rule.severity == "major"


def test_agent_spec_rejects_extra_field():
    with pytest.raises(ValidationError):
        AgentSpec(
            name="x",
            system_prompt="hi",
            tools=[],
            rules=[Rule(id="r", text="a rule")],
            unexpected_field="surprise",
        )


def test_run_defaults_and_unique_ids():
    spec = make_agent_spec()
    run1 = Run(agent_spec=spec, provider="fake")
    run2 = Run(agent_spec=spec, provider="fake")
    assert run1.id != run2.id
    assert run1.status == "pending"
    assert run1.final_score is None
    assert run1.usage == RunUsage()


def test_run_usage_round_trips():
    spec = make_agent_spec()
    run = Run(
        agent_spec=spec,
        provider="fake",
        usage=RunUsage(
            adversary=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            primary=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.0042,
            ),
        ),
    )

    reloaded = Run.model_validate_json(run.model_dump_json())

    assert reloaded.usage.adversary.total_tokens == 15
    assert reloaded.usage.primary.cost_usd == pytest.approx(0.0042)
    assert reloaded.usage.primary.pricing_unavailable is False


def test_token_usage_pricing_unavailable_defaults_to_false():
    assert TokenUsage().pricing_unavailable is False
    assert TokenUsage(pricing_unavailable=True).pricing_unavailable is True


def test_node_valid_and_unique_ids():
    node1 = Node(
        run_id="run-1",
        messages=[Message(role="user", content="hi")],
        target_reply="hello there",
    )
    node2 = Node(
        run_id="run-1",
        messages=[Message(role="user", content="hi")],
        target_reply="hello there",
    )
    assert node1.id != node2.id
    assert node1.parent_id is None
    assert node1.instability_score is None
    assert node1.verdict_id is None
    assert node1.tool_calls == []


def test_node_with_tool_calls_round_trips():
    node = Node(
        run_id="run-1",
        messages=[Message(role="user", content="hi")],
        target_reply="hello there",
        tool_calls=[
            ToolCall(id="call_1", name="lookup_order", input_parameters={"order_id": "123"})
        ],
    )
    reloaded = Node.model_validate_json(node.model_dump_json())
    assert reloaded == node
    assert reloaded.tool_calls[0].name == "lookup_order"


def test_verdict_valid():
    verdict = Verdict(
        run_id="run-1",
        node_id="node-1",
        passed=False,
        rule_id="no-refunds",
        reason="Agent promised a refund directly.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )
    assert verdict.tier == "rules"
    assert verdict.confidence == 1.0
    assert verdict.severity == "critical"


def test_cluster_valid():
    cluster = Cluster(run_id="run-1", label="topic-switch failures")
    assert cluster.member_node_ids == []
    assert cluster.representative_node_id is None


def test_tool_call_round_trips_and_defaults():
    call = ToolCall(id="call_1", name="lookup_order", input_parameters={"order_id": "123"})
    assert call.output is None
    reloaded = ToolCall.model_validate_json(call.model_dump_json())
    assert reloaded == call


def test_tool_result_round_trips():
    result = ToolResult(call_id="call_1", content="shipped")
    assert result.is_error is False
    reloaded = ToolResult.model_validate_json(result.model_dump_json())
    assert reloaded == result


def test_agent_response_without_trace():
    response = AgentResponse(final_reply="hello")
    assert response.final_reply == "hello"
    assert response.trace is None
    assert response.tool_calls == []


def test_agent_response_with_tool_calls():
    calls = [ToolCall(id="call_1", name="lookup_order", input_parameters={"order_id": "123"})]
    response = AgentResponse(final_reply="Your order shipped.", tool_calls=calls)
    assert response.tool_calls == calls
    assert response.trace is None  # trace and tool_calls are independent


def test_agent_response_with_trace():
    steps = [
        Step(thought="I should look this up.", action="lookup_order", observation="found it"),
        Step(thought="I have enough to answer."),
    ]
    response = AgentResponse(final_reply="Your order shipped.", trace=steps)
    assert response.trace == steps
    assert response.trace[0].action == "lookup_order"
    assert response.trace[1].action is None


def test_step_allows_extra_fields():
    step = Step(thought="hmm", tool_call_id="abc-123")
    assert step.thought == "hmm"
    assert step.model_dump()["tool_call_id"] == "abc-123"


def test_agent_response_rejects_extra_field():
    with pytest.raises(ValidationError):
        AgentResponse(final_reply="hi", unexpected_field="surprise")
