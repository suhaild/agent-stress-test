"""Ties everything into one run — the Orchestrator-Workers composition root.

Injects the concrete workers (simulator, target, judge, scorer, search) and
drives one bounded ``Run`` end to end, then scores and (if a ``Store`` was
injected) persists it through that port.
"""

import logging
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

logger = logging.getLogger(__name__)

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
        # Metering only — read once the run finishes and attached to Run.usage.
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

        ``run_id``/``tree`` let a caller pre-generate the run id and pass in
        a tree it already holds a reference to, so it can read live progress
        from another thread while the search is still writing to it.
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

        logger.info(
            "run started run_id=%s agent=%s provider=%s budget=%d",
            run.id, self._agent_spec.name, provider_name, budget,
        )

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

        logger.info(
            "run completed run_id=%s score=%.3f nodes=%d failures=%d",
            run.id, reliability.score, len(tree.nodes()), len(result.failures),
        )

        if self._store is not None:
            self._persist(run, tree)

        return RunResult(
            run=run, tree=tree, reliability=reliability, failures=result.failures
        )

    def _persist(self, run: Run, tree: ConversationTree) -> None:
        # Nodes/verdicts must land before the run row flips to "completed" —
        # the dashboard's SSE loop treats that status as the signal the
        # tree is safe to reload.
        self._store.save_nodes(tree.nodes())
        self._store.save_verdicts(tree.all_verdicts())
        self._store.save_run(run)


def _profile_extra_personas(store: Store | None, agent_spec_name: str) -> dict[str, object]:
    """This agent's approved ``StressProfile`` personas, converted to
    DeepEval's persona shape — or ``{}`` if there's no store or no profile
    yet."""
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

    ``sim_provider`` drives the adversarial simulator and, when ``judge`` is
    not given, also backs the tier-2 LLM judge that resolves cases tier 1's
    deterministic checks can't. ``llm``, when given, is read for metering
    only (``Run.usage``) — it's whichever ``LLMProvider`` backs the target
    agent; omit it for a target that isn't LLM-backed and
    ``Run.usage.primary`` stays at zero.

    When ``store`` holds an approved ``StressProfile`` for ``agent_spec``,
    its personas are merged in as extra tactics alongside the bundled ones.
    ``tactics``, when given explicitly, may name either a bundled tactic or
    one of the profile's own persona names.

    ``argument_correctness`` and ``task_completion`` layer Phase-C node-level
    metrics on top of the rule judge as extra, independent verdict axes.
    ``conversation_metrics`` wires a conversation-level judge (role
    adherence, knowledge retention, etc.), scored once per persona's whole
    conversation rather than per node; off by default since each metric
    costs its own LLM call(s) per persona.
    """
    resolved_judge = judge if judge is not None else build_two_tier_judge(agent_spec, sim_provider)
    metric_judges: list[Judge] = []
    if argument_correctness:
        # No-ops on nodes without tool calls, so safe to leave on by default.
        metric_judges.append(ToolArgumentJudge(sim_provider))
    if task_completion:
        # Costs 2 LLM calls per node regardless of tool use, so off by default.
        metric_judges.append(TaskCompletionJudge(sim_provider))
    if metric_judges:
        resolved_judge = CompositeJudge([resolved_judge, *metric_judges])
    # A single sample can't measure agreement, so skip the extra calls below sample_n=2.
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
        personas=tactics,
        sample_n=sample_n,
        extra_personas=extra_personas,
        conversation_judge=conversation_judge,
    )
    return Runner(agent_spec, strategy, store, sim_provider=sim_provider, llm=llm)
