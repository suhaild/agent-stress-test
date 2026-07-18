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

    Field name is ``input`` (not ``input_parameters``) because this mirrors
    the wire format litellm/Anthropic expect on the message itself — see
    ``ToolCall`` for our own domain-level record of a tool call, which uses
    ``input_parameters`` to match DeepEval's field name instead.
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
    only for messages that carry multimodal input or a tool call/result —
    the "tool" role exists for the latter. Adapters that don't understand
    block content (the fake provider, today) only ever see plain-text
    messages, so nothing about them needs to change.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock]
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


class ToolCall(BaseModel):
    """A structured record of one tool invocation a target agent made.

    ``input_parameters`` (not ``arguments``/``input``) matches DeepEval's own
    ``ToolCall`` field name, so this can be handed to DeepEval's tool metrics
    (Phase C) without renaming. ``output`` holds the resolved result, if any —
    an adapter that resolves a call's result separately (see ``ToolResult``)
    folds it in here before the call is persisted.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    input_parameters: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None


class ToolResult(BaseModel):
    """A tool's result, correlated back to its call by ``call_id``.

    A standalone value an adapter can produce when it resolves a tool call
    out of band, before folding ``content`` into the matching
    ``ToolCall.output``.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str
    content: str
    is_error: bool = False


class AgentResponse(BaseModel):
    """A target agent's reply, plus its reasoning trace/tool calls if any.

    ``trace`` is the older free-text Step narration (still used by the
    bundled ReAct-style SampleAgent); ``tool_calls`` is the newer structured
    record a genuine tool-calling target (Phase A6) reports instead. Both are
    additive and independent — a target reports whichever fits how it works.
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

    ``import_path`` is ``"module.path:attribute"``: the module is imported
    and the named attribute, a ``Callable[[list[Message]], str |
    AgentResponse]``, is wrapped as a ``PythonFunctionAgent`` (see
    ``composition.py``'s ``_load_python_target``).
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


TargetConfig = Annotated[
    HttpTargetConfig | PythonTargetConfig | SubprocessTargetConfig | ProviderTargetConfig,
    Field(discriminator="kind"),
]


class Capabilities(BaseModel):
    """What a ``TargetAgent`` adapter actually supports, declared up front
    (see ``TargetAgent.capabilities()`` in ``ports.py``) so a probe that needs
    e.g. real tool-calling can ask before running, rather than discovering
    "this target can't do that" as a confusing mid-run failure. Deliberately
    just four booleans — no session lifecycle yet (see the build plan's
    Phase D).
    """

    model_config = ConfigDict(extra="forbid")

    tools: bool = False
    sessions: bool = False
    streaming: bool = False
    multimodal: bool = False


class TokenUsage(BaseModel):
    """Token counts + dollar cost accumulated across some set of LLM calls —
    the immutable snapshot a ``UsageMeter`` (see ``ports.py``) produces via
    ``.total()``. ``cost_usd`` stays ``0.0`` whenever a real cost couldn't be
    computed (the fake provider, or a model id litellm's pricing table
    doesn't recognize) — ``pricing_unavailable`` says why, rather than a
    silently-wrong ``0.0`` being mistaken for "this really was free".
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    pricing_unavailable: bool = False


class RunUsage(BaseModel):
    """A run's spend, split "adversary vs. rest" (see the build plan's Phase
    A5): ``adversary`` is whatever drove the adversarial simulator (and, by
    default, the tier-2 judge — see ``build_two_tier_judge``'s default
    provider); ``primary`` is whatever drives the target agent itself (and,
    by extension, the self-consistency scorer, which just resamples the
    target) — see ``Runner.run()``, which reads both meters' totals here.
    """

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
    """The declarative definition of the agent under test.

    ``target`` describes how to build the ``TargetAgent`` this spec runs
    against (see ``composition.py``'s ``_build_target_from_spec``) — left
    unset (the common case), the run falls back to the bundled, LLM-driven
    ``SampleAgent`` instead.

    ``purpose``/``domain`` are optional free text (blank by default, so every
    pre-existing spec still loads unchanged) describing what this agent is
    for and what field it operates in — read by ``reasoning/profiler.py``'s
    ``AgentProfiler`` so the personas/rules it proposes are grounded in this
    specific agent, not the bundled customer-support tactic library.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
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
    # Which branch produced this node — a bundled tactic name under
    # GreedyBestFirstSearch, or a persona name (bundled or profile-sourced)
    # under DeepEvalConversationSearch (see orchestration/deepeval_search.py's
    # DeepEvalConversationSearch docstring). The field predates the persona
    # engine and was never renamed since the two happen to share the same 5
    # bundled names by default.
    tactic: str | None = None
    instability_score: float | None = None
    verdict_id: str | None = None
    # The structured tool calls the target made producing target_reply, if
    # any (see AgentResponse.tool_calls) — persisted so a judge/metric or a
    # replayed report can see them later, not just at judging time.
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Verdict(BaseModel):
    """A judge result attached to a node.

    ``scope`` says what the verdict is *about*: ``"rule"`` (the default — an
    AgentSpec behavioral rule, tier-1 or tier-2, keyed by ``rule_id``),
    ``"tool"`` (a Phase-C tool-call metric, e.g. argument correctness — has no
    ``rule_id`` and is rendered inline with the node's tool-call block, not as
    a generic rule verdict), ``"task"`` (a Phase-C whole-node task-completion
    metric), or ``"conversation"`` (a Phase-C2 whole-conversation metric —
    scores a full root-to-leaf persona conversation, not one node; ``node_id``
    is the conversation's LEAF node, since ``tree.path_to_root()`` of that id
    reconstructs exactly the conversation judged. ``rule_id`` is set for these
    too — either a fixed metric name like ``"role_adherence"`` or, for the
    per-rule conversational GEval, the AgentSpec rule's own id — so failures
    stay distinguishable from one another and from plain node-scoped rule
    verdicts). Defaulting to ``"rule"`` keeps every pre-C verdict and every
    already-persisted row valid unchanged.
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


class ProfilePersona(BaseModel):
    """One profiler-generated adversarial persona for a specific AgentSpec.

    The same ``scenario``/``user_description`` shape DeepEval's own
    ``ConversationalGolden`` needs (see ``reasoning/profiler.py``'s
    ``to_conversational_golden``), kept as our own plain model here so
    ``deepeval`` stays confined to the reasoning layer (CLAUDE.md Golden
    Rule #1) — ``models.py`` is core and must never import it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    user_description: str = Field(min_length=1)


class StressProfile(BaseModel):
    """A profiler-generated bundle of candidate personas and rules for one
    AgentSpec — the hybrid gate's artifact (see ``reasoning/profiler.py``).

    ``personas`` are usable as soon as they're generated: picking one to run
    a simulated conversation against has no lasting effect on the spec.
    ``candidate_rules`` are different — silently merging a bad rule into
    ``AgentSpec.rules`` would misjudge every future run of this agent, so
    they stay PROPOSED here, separate from the spec's own ``rules``, until a
    human reviews (and edits, via the dashboard's profile screen) them.
    Nothing in this codebase copies ``candidate_rules`` into an AgentSpec
    automatically.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_spec_name: str
    personas: list[ProfilePersona] = Field(default_factory=list)
    candidate_rules: list[Rule] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
