import pytest
from pydantic import ValidationError

from agent_stress_test.models import (
    AgentResponse,
    AgentSpec,
    Cluster,
    Message,
    Node,
    Run,
    Step,
    ToolSpec,
    Verdict,
)


def make_agent_spec(**overrides) -> AgentSpec:
    data = {
        "name": "test_agent",
        "system_prompt": "You are a helpful assistant.",
        "tools": [ToolSpec(name="lookup", description="Looks things up.")],
        "rules": ["Never lie."],
    }
    data.update(overrides)
    return AgentSpec(**data)


def test_message_valid():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_message_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="wizard", content="hello")


def test_agent_spec_valid():
    spec = make_agent_spec()
    assert spec.name == "test_agent"
    assert len(spec.tools) == 1
    assert spec.rules == ["Never lie."]


def test_agent_spec_missing_system_prompt():
    with pytest.raises(ValidationError):
        AgentSpec(name="x", system_prompt="", tools=[], rules=["a rule"])


def test_agent_spec_empty_rules():
    with pytest.raises(ValidationError):
        AgentSpec(name="x", system_prompt="hi", tools=[], rules=[])


def test_agent_spec_rejects_extra_field():
    with pytest.raises(ValidationError):
        AgentSpec(
            name="x",
            system_prompt="hi",
            tools=[],
            rules=["a rule"],
            unexpected_field="surprise",
        )


def test_run_defaults_and_unique_ids():
    spec = make_agent_spec()
    run1 = Run(agent_spec=spec, provider="fake")
    run2 = Run(agent_spec=spec, provider="fake")
    assert run1.id != run2.id
    assert run1.status == "pending"
    assert run1.final_score is None


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


def test_verdict_valid():
    verdict = Verdict(
        run_id="run-1",
        node_id="node-1",
        passed=False,
        rule_id="no-refunds",
        reason="Agent promised a refund directly.",
        tier="rules",
    )
    assert verdict.tier == "rules"


def test_verdict_rejects_bad_tier():
    with pytest.raises(ValidationError):
        Verdict(
            run_id="run-1",
            node_id="node-1",
            passed=True,
            reason="fine",
            tier="tier-3-llm-council",
        )


def test_cluster_valid():
    cluster = Cluster(run_id="run-1", label="topic-switch failures")
    assert cluster.member_node_ids == []
    assert cluster.representative_node_id is None


def test_agent_response_without_trace():
    response = AgentResponse(final_reply="hello")
    assert response.final_reply == "hello"
    assert response.trace is None


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
