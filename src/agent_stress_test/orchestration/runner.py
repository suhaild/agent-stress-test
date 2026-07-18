"""Ties everything into one run — the Orchestrator-Workers composition root.

This is the single wiring point: it injects the concrete workers (simulator,
target, judge, scorer, search) and drives one bounded ``Run`` end to end. The
ports it depends on — the target agent and the LLM providers — are handed in
from outside (the CLI or a test), so the core never constructs its own
dependencies. At the end of a run it computes the compounding reliability score
and, when a ``Store`` was injected, persists the whole run through that port —
the runner depends only on the abstract ``Store``, never on SQLite.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from agent_stress_test.models import AgentSpec, Message, Run, Verdict
from agent_stress_test.orchestration.deepeval_search import DeepEvalConversationSearch
from agent_stress_test.orchestration.reliability import ReliabilityReport, score_run
from agent_stress_test.orchestration.search import SearchStrategy
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import LLMProvider, Store, TargetAgent
from agent_stress_test.reasoning.consistency import ConsistencyScorer
from agent_stress_test.reasoning.judge import (
    CompositeJudge,
    Judge,
    TaskCompletionJudge,
    ToolArgumentJudge,
    build_conversation_judge,
    build_two_tier_judge,
)
from agent_stress_test.reasoning.profiler import to_conversational_golden

_DEFAULT_SEED: list[Message] = [
    Message(role="user", content="Hi, I need help with my recent order.")
]


@dataclass
class RunResult:
    """The in-memory outcome of a run: the Run record, tree, failures, score."""

    run: Run
    tree: ConversationTree
    reliability: ReliabilityReport
    failures: list[Verdict] = field(default_factory=list)


class Runner:
    """Drives one stress-test run over an injected search strategy."""

    def __init__(
        self,
        agent_spec: AgentSpec,
        strategy: SearchStrategy,
        store: Store | None = None,
        *,
        sim_provider: LLMProvider | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._agent_spec = agent_spec
        self._strategy = strategy
        self._store = store
        # Metering only — read once the run finishes (see .run()) and
        # attached to Run.usage; never used to drive any decision here.
        self._sim_provider = sim_provider
        self._llm = llm

    def run(
        self,
        *,
        provider_name: str,
        budget: int,
        seed_messages: list[Message] | None = None,
        run_id: str | None = None,
        tree: ConversationTree | None = None,
    ) -> RunResult:
        """Run one stress test end to end.

        ``run_id``/``tree`` let a caller pre-generate the run's id and hand in a
        ``ConversationTree`` it already holds a reference to — the search
        mutates that exact tree in place, so the caller can read live progress
        (``tree.nodes()``/``tree.all_verdicts()``) from another thread while
        this call is still running. Both default to ``None``, so every existing
        caller is unaffected.
        """
        run = Run(
            id=run_id if run_id is not None else str(uuid4()),
            agent_spec=self._agent_spec,
            provider=provider_name,
            budget=budget,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        tree = tree if tree is not None else ConversationTree(run.id)

        seed = seed_messages if seed_messages is not None else list(_DEFAULT_SEED)
        result = self._strategy.search(tree, seed, budget=budget)

        reliability = score_run(tree.nodes(), tree.all_verdicts())
        run.final_score = reliability.score
        if self._sim_provider is not None:
            run.usage.adversary = self._sim_provider.meter.total()
        if self._llm is not None:
            run.usage.primary = self._llm.meter.total()
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)

        if self._store is not None:
            self._persist(run, tree)

        return RunResult(
            run=run, tree=tree, reliability=reliability, failures=result.failures
        )

    def _persist(self, run: Run, tree: ConversationTree) -> None:
        # Nodes/verdicts must land before the run row flips to "completed" —
        # a concurrent reader (the dashboard's SSE loop) treats that status as
        # the signal that the full tree is safe to reload, and each save here
        # commits immediately (see SqliteStore._upsert).
        for node in tree.nodes():
            self._store.save_node(node)
        for verdict in tree.all_verdicts():
            self._store.save_verdict(verdict)
        self._store.save_run(run)


def _profile_extra_personas(store: Store | None, agent_spec_name: str) -> dict[str, object]:
    """This agent's own approved ``StressProfile`` personas, converted to
    DeepEval's persona shape — or ``{}`` if there's no store or no profile
    yet. Kept loosely typed (see ``DeepEvalConversationSearch.__init__``'s
    ``extra_personas`` docstring) so this module never needs to import
    ``deepeval`` itself; ``to_conversational_golden`` is the one reasoning-
    layer function that actually does.
    """
    if store is None:
        return {}
    profile = store.get_stress_profile(agent_spec_name)
    if profile is None:
        return {}
    return {persona.name: to_conversational_golden(persona) for persona in profile.personas}


def build_runner(
    *,
    agent_spec: AgentSpec,
    target: TargetAgent,
    sim_provider: LLMProvider,
    llm: LLMProvider | None = None,
    judge: Judge | None = None,
    store: Store | None = None,
    tactics: list[str] | None = None,
    sample_n: int = 3,
    argument_correctness: bool = True,
    task_completion: bool = False,
    conversation_metrics: bool = False,
) -> Runner:
    """Wire the workers into a ready-to-run Runner (the composition root).

    ``sim_provider`` drives the adversarial simulator, and — when ``judge`` is
    not given — also backs the tier-2 LLM judge, so ambiguous cases tier 1's
    deterministic checks can't resolve on their own (e.g. a reply that
    generically lists "returns" as a capability vs. one that's actually
    processing a customer's return) get a real judgment instead of an
    ever-more-elaborate regex. Tier 1 still decides first and wins whenever it
    fires; the LLM is only consulted when every deterministic check passes
    (see ``TwoTierJudge``). The self-consistency scorer resamples ``target``
    itself (see ``ConsistencyScorer``), so it's built automatically whenever
    ``sample_n >= 2`` — a single sample can only ever score 0.0, so there's no
    point paying for the extra calls below that threshold; a caller wanting to
    skip the extra cost/latency entirely can just pass ``sample_n=1``.
    ``store``, when given, persists the finished run through the ``Store``
    port.

    ``llm``, when given, is read for metering only (see ``Run.usage`` and
    ``Runner.run()``) — it's whichever ``LLMProvider`` instance backs the
    target agent (and, by extension, the self-consistency scorer, which just
    resamples ``target``), never used to build anything here. Omit it for a
    target that isn't LLM-backed at all (``HttpAgent``, a scripted Python
    callable, ...) and ``Run.usage.primary`` simply stays at zero.

    The search strategy is ``DeepEvalConversationSearch`` (see
    ``orchestration/deepeval_search.py``): one full DeepEval-simulated
    conversation per persona (``budget`` = turns per conversation), not the
    old per-node judge-driven ``GreedyBestFirstSearch`` (still in
    ``orchestration/search.py``, kept for direct use/testing, just no longer
    what this composition root wires by default).

    When ``store`` holds an approved ``StressProfile`` for ``agent_spec``
    (see ``reasoning/profiler.py``), its personas are automatically merged in
    as extra tactics alongside the bundled 5 — no separate opt-in needed,
    since picking a persona to run against has no lasting effect on the spec
    (unlike its candidate rules, which stay proposed until a human copies
    them into ``agent_spec.rules`` by hand). ``tactics``, when given
    explicitly, may name either a bundled tactic or one of this profile's
    own persona names.

    The Phase-C node-level metrics layer on top of the rule judge (as extra,
    independent verdict axes — see ``CompositeJudge``): ``argument_correctness``
    (DeepEval's ``ArgumentCorrectnessMetric``) is on by default because it
    costs nothing on nodes without tool calls and only does real work when
    there's a tool call to judge; ``task_completion`` (``TaskCompletionMetric``)
    is off by default because it costs 2 LLM calls per node regardless of
    tool use. Both attach to whichever judge is in use (the default two-tier
    judge or an explicitly-passed ``judge``).

    ``conversation_metrics`` (Phase C2) wires ``build_conversation_judge`` —
    RoleAdherence/KnowledgeRetention/ConversationCompleteness/TurnRelevancy
    plus a per-rule conversational GEval, scored once per persona's WHOLE
    conversation rather than per node. Off by default: every one of those
    metrics costs its own LLM call(s) per persona, on top of the per-node
    judge already running on every turn.

    Phase C3 measured these real per-metric costs (``scripts/
    measure_metric_costs.py``, claude-haiku-4-5, one real live run over a
    short 4-turn support conversation with one tool call, after fixing a
    real bug the very same measurement run surfaced — see
    ``reasoning/deepeval_bridge.py``'s fence-stripping fix):

        metric                      calls  tokens   cost
        tool_argument_correctness     2     1490   $0.0021
        task_completion                2     1428   $0.0022
        role_adherence                 2     1763   $0.0023
        knowledge_retention             5     3811   $0.0052
        conversation_completeness      5     3349   $0.0042
        turn_relevancy                  3     1972   $0.0024
        conversation_rule_geval (x4)    4     2713   $0.0046
        TOTAL (one full pass)          23    16526   $0.0230

    None of these are individually "wildly expensive" on Haiku (a couple of
    cents for the whole stack over one conversation) — but the two defaults
    above are still correct, now backed by evidence instead of a guess:
    ``task_completion`` fires on EVERY node regardless of content (the only
    per-node metric here with no cheap early-out, unlike
    ``argument_correctness``, which skips nodes with no tool calls), so its
    cost compounds with node count across a run. ``conversation_metrics``
    is ~19 calls (~$0.02) PER PERSONA — with the bundled 5-tactic registry
    alone that's ~95 extra calls layered on top of the per-node judge
    already running every turn. Both stay opt-in.
    """
    resolved_judge = judge if judge is not None else build_two_tier_judge(agent_spec, sim_provider)
    metric_judges: list[Judge] = []
    if argument_correctness:
        metric_judges.append(ToolArgumentJudge(sim_provider))
    if task_completion:
        metric_judges.append(TaskCompletionJudge(sim_provider))
    if metric_judges:
        resolved_judge = CompositeJudge([resolved_judge, *metric_judges])
    scorer = ConsistencyScorer(target) if sample_n >= 2 else None
    extra_personas = _profile_extra_personas(store, agent_spec.name)
    conversation_judge = (
        build_conversation_judge(sim_provider, agent_spec) if conversation_metrics else None
    )
    strategy = DeepEvalConversationSearch(
        target,
        sim_provider,
        resolved_judge,
        scorer,
        tactics=tactics,
        sample_n=sample_n,
        extra_personas=extra_personas,
        conversation_judge=conversation_judge,
    )
    return Runner(agent_spec, strategy, store, sim_provider=sim_provider, llm=llm)
