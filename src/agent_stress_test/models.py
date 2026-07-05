"""Pydantic data models: Run, Node, Verdict, Cluster, AgentSpec."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One turn in a conversation, in the shape every port passes around."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class Step(BaseModel):
    """One reasoning step exposed by a ReAct-style agent.

    All fields are optional and extra fields are allowed: different agents
    expose different step shapes, so this stays loosely typed by design
    (the one exception to this file's usual `extra="forbid"` convention).
    """

    model_config = ConfigDict(extra="allow")

    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


class AgentResponse(BaseModel):
    """A target agent's reply, plus its reasoning trace if it exposed one."""

    model_config = ConfigDict(extra="forbid")

    final_reply: str
    trace: list[Step] | None = None


class ToolSpec(BaseModel):
    """A tool a target agent has available, as declared by its AgentSpec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class AgentSpec(BaseModel):
    """The declarative definition of the agent under test."""

    model_config = ConfigDict(extra="forbid")

    name: str
    system_prompt: str = Field(min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    rules: list[str] = Field(min_length=1)


class Run(BaseModel):
    """One full stress-test session."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_spec: AgentSpec
    provider: str
    budget: int = 20
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    final_score: float | None = None


class Node(BaseModel):
    """One point in a conversation tree."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    parent_id: str | None = None
    messages: list[Message]
    target_reply: str
    tactic: str | None = None
    instability_score: float | None = None
    verdict_id: str | None = None


class Verdict(BaseModel):
    """A judge result attached to a node."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    node_id: str
    passed: bool
    rule_id: str | None = None
    reason: str
    tier: Literal["rules", "llm"]


class Cluster(BaseModel):
    """A named group of failure nodes."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    label: str
    member_node_ids: list[str] = Field(default_factory=list)
    representative_node_id: str | None = None
