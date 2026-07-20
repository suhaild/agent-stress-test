"""Greedy best-first search — the Evaluator-Optimizer loop.

The core engine cycle: the simulator generates an adversarial turn, the
target replies, the judge evaluates it, the scorer estimates instability,
and the search steers toward the most promising failures. Sits behind a
``SearchStrategy`` interface (Strategy pattern) so other strategies (e.g.
MCTS) can slot in without touching the runner.

The frontier is ordered by ``instability + judge proximity-to-failure``, so
nodes whose replies are shaky or already show (or nearly show) a rule
violation are expanded first. A deflecting reply (refusal/non-answer/
redirect) is tracked as its own signal, distinct from a genuine rule pass.
The closest near-miss across the whole search is reported as a first-class
result rather than left buried in the tree.
"""

import heapq
import itertools
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agent_stress_test.models import AgentResponse, Message, Node, Severity, Verdict
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import TargetAgent
from agent_stress_test.reasoning.consistency import ConsistencyScorer
from agent_stress_test.reasoning.judge import Judge, is_deflection
from agent_stress_test.reasoning.simulator import Simulator, default_registry

# How "fertile" a failure of each severity makes a node's region for further expansion.
SEVERITY_WEIGHT: dict[Severity, float] = {"minor": 0.34, "major": 0.67, "critical": 1.0}

# Deflection alone raises priority roughly on par with a "major" failure.
DEFLECTION_WEIGHT = 0.67


def failure_proximity(verdicts: list[Verdict]) -> float:
    """Max severity weight among *failed* verdicts, or 0.0 if none failed, in
    [0, 1]. Kept separate from ``graded_proximity`` below because
    ``reliability.py``'s ``SeverityWeightedModel`` reuses this exact
    discrete formula for the reliability score itself."""
    weights = [SEVERITY_WEIGHT[v.severity] for v in verdicts if not v.passed]
    return max(weights, default=0.0)


def graded_proximity(verdicts: list[Verdict]) -> float:
    """Like ``failure_proximity``, but a passing LLM/metric-tier verdict that
    barely cleared its own pass/fail threshold (low ``confidence``) also
    counts as a near-miss instead of a clean 0. Deterministic tier-1 checks
    always report ``confidence=1.0``, so a passing one still contributes 0.0
    — a regex match has no "almost"."""
    best = failure_proximity(verdicts)
    for verdict in verdicts:
        if verdict.passed and verdict.tier == "llm":
            best = max(best, 1.0 - verdict.confidence)
    return best


def node_priority(node: Node, verdicts: list[Verdict]) -> float:
    """Search priority for a node, in [0, 2] (higher = expand sooner)."""
    proximity = graded_proximity(verdicts)
    if is_deflection(node.target_reply):
        proximity = max(proximity, DEFLECTION_WEIGHT)
    return (node.instability_score or 0.0) + proximity


def score_and_judge(
    node: Node,
    response: AgentResponse,
    *,
    run_id: str,
    judge: Judge,
    scorer: ConsistencyScorer | None,
    sample_n: int,
) -> list[Verdict]:
    """Instability-score and judge a node's reply — the shared step both
    ``GreedyBestFirstSearch`` and ``DeepEvalConversationSearch`` run once a
    node's target reply is known. Sets ``node.instability_score`` in place
    (0.0 without a scorer)."""
    node.instability_score = scorer.score(node.messages, sample_n) if scorer is not None else 0.0
    return judge.judge(response, run_id=run_id, node_id=node.id, conversation=node.messages)


class Frontier:
    """A max-priority queue of node ids (higher priority pops first).

    Built on a min-heap of ``(-priority, insertion_counter, node_id)`` so
    ties break FIFO, keeping expansion deterministic among equally
    promising nodes.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str]] = []
        self._counter = itertools.count()

    def push(self, node_id: str, priority: float) -> None:
        heapq.heappush(self._heap, (-priority, next(self._counter), node_id))

    def pop(self) -> str:
        return heapq.heappop(self._heap)[2]

    def is_empty(self) -> bool:
        return not self._heap

    def __len__(self) -> int:
        return len(self._heap)


@dataclass(frozen=True)
class ExplorationDetail:
    """Frontier-exploration signals only a priority-driven search over a
    shared tree can produce.

    ``near_miss`` is the passing node with the highest ``graded_proximity``
    seen anywhere in the search, or ``None`` if every node was either a
    clean pass or an outright failure. ``deflections`` is tracked
    separately from a genuine rule pass.
    """

    near_miss: Node | None = None
    deflections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    """Summary of one search. The tree (Blackboard) holds the full detail.

    ``exploration_detail`` is populated only by strategies with a shared
    priority frontier (``GreedyBestFirstSearch``); it is ``None`` — not an
    empty ``ExplorationDetail`` — for strategies that don't track this at
    all (``DeepEvalConversationSearch``). ``None`` means "not measured",
    distinct from a populated result with ``near_miss=None`` ("measured,
    found none").
    """

    expansions: int
    nodes_created: int
    failures: list[Verdict] = field(default_factory=list)
    exploration_detail: ExplorationDetail | None = None


class SearchStrategy(ABC):
    """A strategy for exploring the conversation tree (Strategy pattern).

    Every implementation must reliably populate ``SearchResult.expansions``/
    ``.nodes_created``/``.failures``. ``budget`` and ``seed_messages`` may
    carry a strategy-specific meaning despite sharing this one signature
    (see e.g. ``DeepEvalConversationSearch``).
    """

    @abstractmethod
    def search(
        self, tree: ConversationTree, seed_messages: list[Message], *, budget: int
    ) -> SearchResult: ...


class GreedyBestFirstSearch(SearchStrategy):
    """Expand the highest-priority frontier node until the budget is spent.

    The four workers are injected; this strategy owns only the loop.
    ``scorer`` is optional — without it instability is treated as 0.0 and
    priority is driven by judge proximity-to-failure alone.

    ``prune_floor`` drops a node from the frontier (it is still created,
    judged, and counted, just never expanded further) once its
    ``node_priority`` falls below the floor. Defaults to ``0.0``, which
    prunes nothing.
    """

    def __init__(
        self,
        simulator: Simulator,
        target: TargetAgent,
        judge: Judge,
        scorer: ConsistencyScorer | None = None,
        *,
        tactics: list[str] | None = None,
        sample_n: int = 3,
        max_concurrency: int = 5,
        prune_floor: float = 0.0,
    ) -> None:
        self._simulator = simulator
        self._target = target
        self._judge = judge
        self._scorer = scorer
        self._tactics = tactics if tactics is not None else default_registry().names()
        self._sample_n = sample_n
        self._max_concurrency = max_concurrency
        self._prune_floor = prune_floor

    def search(
        self, tree: ConversationTree, seed_messages: list[Message], *, budget: int
    ) -> SearchResult:
        frontier = Frontier()
        failures: list[Verdict] = []
        deflections: list[str] = []
        nodes_created = 0
        near_miss: Node | None = None
        near_miss_proximity = 0.0

        def track(candidate: Node, candidate_verdicts: list[Verdict]) -> None:
            nonlocal near_miss, near_miss_proximity
            if is_deflection(candidate.target_reply):
                deflections.append(candidate.id)
            if any(not v.passed for v in candidate_verdicts):
                return  # an outright failure, not a near-miss
            proximity = graded_proximity(candidate_verdicts)
            if proximity > near_miss_proximity:
                near_miss, near_miss_proximity = candidate, proximity

        root_node, root_verdicts = self._compute(
            tree.run_id, seed_messages, parent_id=None, tactic=None
        )
        root = self._commit(tree, root_node, root_verdicts, failures=failures)
        nodes_created += 1
        track(root, tree.verdicts(root.id))
        frontier.push(root.id, node_priority(root, tree.verdicts(root.id)))

        expansions = 0
        while not frontier.is_empty() and expansions < budget:
            node = tree.get(frontier.pop())
            children = self._expand(tree, node, failures=failures)
            expansions += 1
            nodes_created += len(children)
            for child in children:
                child_verdicts = tree.verdicts(child.id)
                track(child, child_verdicts)
                priority = node_priority(child, child_verdicts)
                if priority >= self._prune_floor:
                    frontier.push(child.id, priority)

        return SearchResult(
            expansions=expansions,
            nodes_created=nodes_created,
            failures=failures,
            exploration_detail=ExplorationDetail(near_miss=near_miss, deflections=deflections),
        )

    def _expand(
        self, tree: ConversationTree, node: Node, *, failures: list[Verdict]
    ) -> list[Node]:
        """Generate one adversarial child per tactic and evaluate each.

        Tactic branches run concurrently, bounded to ``max_concurrency`` in
        flight. Only the compute step runs off the main thread; committing
        results to the shared tree happens back on the main thread, so the
        tree is never mutated from more than one thread at a time.
        """
        # Drop any cache breakpoint inherited from an ancestor turn before
        # adding a new one — otherwise breakpoints accumulate one per tree
        # level and blow past Anthropic's max of 4 per request as the search
        # goes deeper.
        base = [
            *(m.model_copy(update={"cache": False}) if m.cache else m for m in node.messages),
            Message(role="assistant", content=node.target_reply, cache=True),
        ]

        def compute_for(tactic: str) -> tuple[Node, list[Verdict]]:
            probe = self._simulator.simulate(base, tactic)
            return self._compute(tree.run_id, [*base, probe], parent_id=node.id, tactic=tactic)

        workers = min(self._max_concurrency, len(self._tactics))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            computed = list(executor.map(compute_for, self._tactics))

        return [
            self._commit(tree, child_node, verdicts, failures=failures)
            for child_node, verdicts in computed
        ]

    def _compute(
        self,
        run_id: str,
        messages: list[Message],
        *,
        parent_id: str | None,
        tactic: str | None,
    ) -> tuple[Node, list[Verdict]]:
        """Target replies, judge evaluates, scorer scores. Safe to call
        concurrently: builds a Node and its Verdicts purely from its
        arguments, without touching the tree."""
        response = self._target.respond(messages)
        node = Node(
            run_id=run_id,
            parent_id=parent_id,
            messages=messages,
            target_reply=response.final_reply,
            tactic=tactic,
            tool_calls=response.tool_calls,
        )
        verdicts = score_and_judge(
            node,
            response,
            run_id=run_id,
            judge=self._judge,
            scorer=self._scorer,
            sample_n=self._sample_n,
        )
        return node, verdicts

    def _commit(
        self,
        tree: ConversationTree,
        node: Node,
        verdicts: list[Verdict],
        *,
        failures: list[Verdict],
    ) -> Node:
        tree.add(node)
        tree.attach_verdicts(node.id, verdicts)
        failures.extend(v for v in verdicts if not v.passed)
        return node
