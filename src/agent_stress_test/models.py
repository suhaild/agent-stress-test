"""Pydantic data models: Run, Node, Verdict, Cluster, AgentSpec."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["minor", "major", "critical"]


class Message(BaseModel):
    """One turn in a conversation, in the shape every port passes around."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str
    # Provider-agnostic hint: this message ends a cacheable prompt prefix.
    # Adapters that support prompt caching (e.g. litellm_provider) may act on
    # it; others (e.g. the fake provider) ignore it.
    cache: bool = False


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


CheckType = Literal[
    "banned_tool_use",
    "forbidden_output",
    "required_disclaimer",
    "format_violation",
    "ungrounded_claim",
]


class Rule(BaseModel):
    """A single behavioral rule the agent under test must obey.

    Carries a stable `id` (referenced by the judge and by verdicts), the
    human-readable `text` (shown to the agent and to the tier-2 LLM judge),
    and a declared `severity`. Severity is configuration, not hardcoded in
    the judge — changing it here changes the severity carried by verdicts.

    `check_type` opts the rule into one of the deterministic tier-1
    `RuleCheck`s (see `reasoning/judge.py`'s check-builder registry), with
    `params` supplying that check's arguments (patterns, tool names, etc.).
    Left as `None`, the rule gets no tier-1 check and is judged by the tier-2
    LLM judge alone — this is the correct default for any rule whose
    violation isn't a simple pattern/trace match.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1)
    severity: Severity = "major"
    check_type: CheckType | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    """The declarative definition of the agent under test."""

    model_config = ConfigDict(extra="forbid")

    name: str
    system_prompt: str = Field(min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    rules: list[Rule] = Field(min_length=1)


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
    error: str | None = None


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
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity


class Cluster(BaseModel):
    """A named group of failure nodes."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    label: str
    member_node_ids: list[str] = Field(default_factory=list)
    representative_node_id: str | None = None


class RegressionCase(BaseModel):
    """A confirmed failure, locked in as a permanent replay target.

    Built from one failure cluster's representative node (see
    ``orchestration/regression.py``'s ``promote_clusters_to_cases``), so the
    corpus tracks distinct failure patterns, not every raw failing node.
    ``status`` is set by a human — ``"open"`` means a known, not-yet-fixed
    issue (replaying it and finding it still fails is expected, not an
    error); ``"resolved"`` means a fix was applied and confirmed, so a future
    replay finding it fails again is a genuine regression.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_spec_name: str
    messages: list[Message]
    tactic: str | None = None
    rule_id: str
    severity: Severity
    source_run_id: str
    source_cluster_id: str
    status: Literal["open", "resolved"] = "open"


class SystemPromptVersion(BaseModel):
    """A snapshot of an agent spec's ``system_prompt`` taken immediately
    before it gets overwritten by ``apply_system_prompt`` — the record that
    lets a fix be undone from the dashboard itself. Hosting this online means
    an operator may have no shell/git access to the deployed file, so ``git
    checkout`` isn't a real safety net there; this is the in-app substitute,
    persisted the same way as everything else in the ``Store``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_spec_name: str
    system_prompt: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
