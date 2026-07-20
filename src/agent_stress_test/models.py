"""Pydantic data models: Run, Node, Verdict, Cluster, AgentSpec."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["minor", "major", "critical"]


class TextBlock(BaseModel):
    """A plain-text content block, litellm/Anthropic-shaped."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    """An image's payload — either inline base64 or a fetchable URL."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ImageBlock(BaseModel):
    """An image content block, litellm/Anthropic-shaped."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    source: ImageSource


class ToolUseBlock(BaseModel):
    """An assistant-issued tool call, as a content block (litellm/Anthropic-shaped).

    Field is ``input`` (not ``input_parameters``) to mirror the litellm/Anthropic
    wire format; see ``ToolCall`` for our own domain-level record.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """A tool's result, as a content block (litellm/Anthropic-shaped)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """One turn in a conversation, in the shape every port passes around.

    ``content`` is usually plain text; it widens to a list of content blocks
    only for multimodal input or a tool call/result (the "tool" role).
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock]
    # Hint that this message ends a cacheable prompt prefix; adapters that
    # don't support prompt caching simply ignore it.
    cache: bool = False


class Step(BaseModel):
    """One reasoning step exposed by a ReAct-style agent.

    Loosely typed by design (`extra="allow"`, unlike the rest of this file):
    different agents expose different step shapes.
    """

    model_config = ConfigDict(extra="allow")

    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


class ToolCall(BaseModel):
    """A structured record of one tool invocation a target agent made.

    ``input_parameters`` (not ``arguments``/``input``) matches DeepEval's own
    ``ToolCall`` field name, so this can be handed to DeepEval's tool metrics
    without renaming.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    input_parameters: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None


class ToolResult(BaseModel):
    """A tool's result, correlated back to its call by ``call_id`` — used
    when an adapter resolves a call out of band, before folding it into
    ``ToolCall.output``."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    content: str
    is_error: bool = False


class AgentResponse(BaseModel):
    """A target agent's reply, plus its reasoning trace/tool calls if any.

    ``trace`` is free-text Step narration (used by the ReAct-style
    SampleAgent); ``tool_calls`` is a structured record a genuine
    tool-calling target reports instead. A target reports whichever fits.
    """

    model_config = ConfigDict(extra="forbid")

    final_reply: str
    trace: list[Step] | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """A tool a target agent has available, as declared by its AgentSpec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class HttpTargetConfig(BaseModel):
    """``target: {kind: http}`` — an HTTP/JSON endpoint (see ``targets/http_agent.py``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["http"] = "http"
    url: str
    headers: dict[str, str] | None = None
    timeout: float = 30.0


class PythonTargetConfig(BaseModel):
    """``target: {kind: python}`` — a bring-your-own Python callable.

    ``import_path`` is ``"module.path:attribute"``, wrapped as a
    ``PythonFunctionAgent`` (see ``composition.py``'s ``_load_python_target``).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["python"] = "python"
    import_path: str


class SubprocessTargetConfig(BaseModel):
    """``target: {kind: subprocess}`` — a command-line process speaking
    stdin/stdout JSON framing (see ``targets/subprocess_agent.py``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["subprocess"] = "subprocess"
    command: list[str]
    timeout: float = 30.0
    cwd: str | None = None


class ProviderTargetConfig(BaseModel):
    """``target: {kind: provider}`` — a bare model id driven directly through
    litellm's native tool-calling, using the spec's own tools/system prompt
    (see ``targets/provider_agent.py``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["provider"] = "provider"
    model: str


class SampleAdvancedTargetConfig(BaseModel):
    """``target: {kind: sample_advanced}`` — like the implicit SampleAgent,
    but every ``Action`` executes for real against an in-memory fake tool
    backend instead of the model inventing its own ``Observation``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sample_advanced"] = "sample_advanced"


TargetConfig = Annotated[
    HttpTargetConfig
    | PythonTargetConfig
    | SubprocessTargetConfig
    | ProviderTargetConfig
    | SampleAdvancedTargetConfig,
    Field(discriminator="kind"),
]


class Capabilities(BaseModel):
    """What a ``TargetAgent`` adapter actually supports, declared up front
    (see ``TargetAgent.capabilities()`` in ``ports.py``) so a probe can check
    before running rather than fail confusingly mid-run."""

    model_config = ConfigDict(extra="forbid")

    tools: bool = False
    sessions: bool = False
    streaming: bool = False
    multimodal: bool = False


class TokenUsage(BaseModel):
    """Token counts + dollar cost accumulated across a set of LLM calls —
    the immutable snapshot a ``UsageMeter`` produces via ``.total()``.
    ``pricing_unavailable`` distinguishes a real $0 cost from one litellm
    simply couldn't price."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    pricing_unavailable: bool = False


class RunUsage(BaseModel):
    """A run's spend, split by role: ``adversary`` covers the simulator (and,
    by default, the tier-2 judge); ``primary`` covers the target agent
    itself and the self-consistency scorer that resamples it."""

    model_config = ConfigDict(extra="forbid")

    adversary: TokenUsage = Field(default_factory=TokenUsage)
    primary: TokenUsage = Field(default_factory=TokenUsage)


CheckType = Literal[
    "banned_tool_use",
    "forbidden_output",
    "required_disclaimer",
    "format_violation",
    "ungrounded_claim",
]


class Rule(BaseModel):
    """A single behavioral rule the agent under test must obey.

    `check_type` opts the rule into a deterministic tier-1 `RuleCheck` (see
    `reasoning/judge.py`), with `params` as that check's arguments. Left
    `None` (the default), the rule is judged by the tier-2 LLM judge alone.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1)
    severity: Severity = "major"
    check_type: CheckType | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    """The declarative definition of the agent under test.

    ``name`` is a stable identifier every lookup keys off (regression cases,
    stress profiles, cross-run history) and must never change once runs
    exist against it. ``display_name`` is purely cosmetic, falling back to
    ``name`` when blank. ``target`` left unset falls back to the bundled
    ``SampleAgent``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str = ""
    purpose: str = ""
    domain: str = ""
    system_prompt: str = Field(min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    rules: list[Rule] = Field(min_length=1)
    target: TargetConfig | None = None


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
    usage: RunUsage = Field(default_factory=RunUsage)


class Node(BaseModel):
    """One point in a conversation tree."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    parent_id: str | None = None
    messages: list[Message]
    target_reply: str
    # Branch that produced this node: a tactic name (GreedyBestFirstSearch)
    # or a persona name (DeepEvalConversationSearch).
    tactic: str | None = None
    instability_score: float | None = None
    verdict_id: str | None = None
    # Persisted so a judge/metric or a replayed report can see the target's
    # tool calls later, not just at judging time.
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Verdict(BaseModel):
    """A judge result attached to a node.

    ``scope`` says what's judged: ``"rule"`` (default, keyed by ``rule_id``),
    ``"tool"``/``"task"`` (a tool-call or whole-node metric), or
    ``"conversation"`` (a whole persona conversation — ``node_id`` is the
    conversation's leaf node, since ``tree.path_to_root()`` from it
    reconstructs the conversation judged).

    ``applicable=False`` means the rule's subject matter never came up here
    (still ``passed=True``), distinct from genuinely holding up under real
    pressure (``passed=True, applicable=True``) — ``rule_coverage`` uses this
    to avoid crediting an untested rule. Always ``True`` for tier-1 checks.
    """

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
    scope: Literal["rule", "tool", "task", "conversation"] = "rule"
    applicable: bool = True


class Cluster(BaseModel):
    """A named group of failure nodes."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    label: str
    member_node_ids: list[str] = Field(default_factory=list)
    representative_node_id: str | None = None


class RegressionCase(BaseModel):
    """A confirmed failure, locked in as a permanent replay target — one per
    failure cluster's representative node, not every raw failing node.
    ``status`` is set by a human: ``"open"`` means still-failing is expected;
    ``"resolved"`` means a future failure is a genuine regression.
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
    before ``apply_system_prompt`` overwrites it, so a fix can be undone from
    the dashboard even when the operator has no git access to the file."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_spec_name: str
    system_prompt: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProfilePersona(BaseModel):
    """One profiler-generated adversarial persona for a specific AgentSpec.

    Mirrors the ``scenario``/``user_description`` shape DeepEval's own
    ``ConversationalGolden`` needs, kept as a plain model here so
    ``deepeval`` stays confined to the reasoning layer.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    user_description: str = Field(min_length=1)


class StressProfile(BaseModel):
    """A profiler-generated bundle of candidate personas and rules for one
    AgentSpec (see ``reasoning/profiler.py``).

    ``candidate_rules`` stay proposed, separate from ``AgentSpec.rules``,
    until a human reviews them — nothing here copies them in automatically.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_spec_name: str
    personas: list[ProfilePersona] = Field(default_factory=list)
    candidate_rules: list[Rule] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
