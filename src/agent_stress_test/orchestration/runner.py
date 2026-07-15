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
from agent_stress_test.orchestration.reliability import ReliabilityReport, score_run
from agent_stress_test.orchestration.search import GreedyBestFirstSearch, SearchStrategy
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import LLMProvider, Store, TargetAgent
from agent_stress_test.reasoning.consistency import ConsistencyScorer
from agent_stress_test.reasoning.judge import Judge, build_two_tier_judge
from agent_stress_test.reasoning.simulator import Simulator

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
        self, agent_spec: AgentSpec, strategy: SearchStrategy, store: Store | None = None
    ) -> None:
        self._agent_spec = agent_spec
        self._strategy = strategy
        self._store = store

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


def build_runner(
    *,
    agent_spec: AgentSpec,
    target: TargetAgent,
    sim_provider: LLMProvider,
    judge: Judge | None = None,
    store: Store | None = None,
    tactics: list[str] | None = None,
    sample_n: int = 3,
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
    """
    simulator = Simulator(sim_provider)
    resolved_judge = judge if judge is not None else build_two_tier_judge(agent_spec, sim_provider)
    scorer = ConsistencyScorer(target) if sample_n >= 2 else None
    strategy = GreedyBestFirstSearch(
        simulator,
        target,
        resolved_judge,
        scorer,
        tactics=tactics,
        sample_n=sample_n,
    )
    return Runner(agent_spec, strategy, store)
