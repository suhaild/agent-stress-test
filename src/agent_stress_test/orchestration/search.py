"""Greedy best-first search — the Evaluator-Optimizer loop.

This is the core engine cycle: the simulator generates an adversarial turn, the
target replies, the judge evaluates it, the scorer estimates instability, and
the search steers toward the most promising failures. Search sits behind a
``SearchStrategy`` interface (Strategy pattern) so an MCTS strategy can slot in
later (Phase 11) without touching the runner.

The frontier is ordered by ``instability + judge proximity-to-failure``: nodes
whose replies are shaky or already show (or nearly show) a rule violation are
expanded first, because that is where more failures tend to live.
"""

import heapq
import itertools
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agent_stress_test.models import Message, Node, Severity, Verdict
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import TargetAgent
from agent_stress_test.reasoning.consistency import ConsistencyScorer
from agent_stress_test.reasoning.judge import Judge
from agent_stress_test.reasoning.simulator import Simulator, default_registry

# How "fertile" a failure of each severity makes a node's region. Critical
# failures point at the most valuable branches to keep expanding.
SEVERITY_WEIGHT: dict[Severity, float] = {"minor": 0.34, "major": 0.67, "critical": 1.0}


def failure_proximity(verdicts: list[Verdict]) -> float:
    """How close this node is to (or into) failure, in [0, 1].

    The max severity weight among *failed* verdicts, or 0.0 if none failed. A
    node that already carries a critical failure scores 1.0 — the most promising
    region to keep probing.
    """
    weights = [SEVERITY_WEIGHT[v.severity] for v in verdicts if not v.passed]
    return max(weights, default=0.0)


def node_priority(node: Node, verdicts: list[Verdict]) -> float:
    """Search priority for a node, in [0, 2] (higher = expand sooner)."""
    return (node.instability_score or 0.0) + failure_proximity(verdicts)


class Frontier:
    """A max-priority queue of node ids (higher priority pops first).

    Built on a min-heap of ``(-priority, insertion_counter, node_id)`` so that
    ties break FIFO — expansion stays deterministic and BFS-like among equally
    promising nodes, which keeps failure discovery reliable.
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
class SearchResult:
    """Summary of one search. The tree (Blackboard) holds the full detail."""

    expansions: int
    nodes_created: int
    failures: list[Verdict] = field(default_factory=list)


class SearchStrategy(ABC):
    """A strategy for exploring the conversation tree (Strategy pattern)."""

    @abstractmethod
    def search(
        self, tree: ConversationTree, seed_messages: list[Message], *, budget: int
    ) -> SearchResult: ...


class GreedyBestFirstSearch(SearchStrategy):
    """Expand the highest-priority frontier node until the budget is spent.

    The four workers are injected (Dependency Injection); this strategy owns
    only the loop and never constructs them. ``scorer`` is optional: without it
    (e.g. non-LLM targets) instability is treated as 0.0 and priority is driven
    by judge proximity-to-failure alone.
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
    ) -> None:
        self._simulator = simulator
        self._target = target
        self._judge = judge
        self._scorer = scorer
        self._tactics = tactics if tactics is not None else default_registry().names()
        self._sample_n = sample_n
        self._max_concurrency = max_concurrency

    def search(
        self, tree: ConversationTree, seed_messages: list[Message], *, budget: int
    ) -> SearchResult:
        frontier = Frontier()
        failures: list[Verdict] = []
        nodes_created = 0

        root_node, root_verdicts = self._compute(
            tree.run_id, seed_messages, parent_id=None, tactic=None
        )
        root = self._commit(tree, root_node, root_verdicts, failures=failures)
        nodes_created += 1
        frontier.push(root.id, node_priority(root, tree.verdicts(root.id)))

        expansions = 0
        while not frontier.is_empty() and expansions < budget:
            node = tree.get(frontier.pop())
            children = self._expand(tree, node, failures=failures)
            expansions += 1
            nodes_created += len(children)
            for child in children:
                frontier.push(child.id, node_priority(child, tree.verdicts(child.id)))

        return SearchResult(
            expansions=expansions, nodes_created=nodes_created, failures=failures
        )

    def _expand(
        self, tree: ConversationTree, node: Node, *, failures: list[Verdict]
    ) -> list[Node]:
        """Generate one adversarial child per tactic and evaluate each.

        The tactic branches are independent network calls (simulate + target +
        optional scorer), so they run concurrently, bounded to
        ``max_concurrency`` in flight at once. Only the compute step runs off
        the main thread; committing results to the shared tree happens back
        on the main thread afterward, so the tree is never mutated from more
        than one thread at a time.
        """
        # Clear any cache breakpoint inherited from an ancestor turn before
        # adding a new one — otherwise breakpoints accumulate one per tree
        # level and blow past Anthropic's max of 4 per request as the search
        # goes deeper. The cache lookback matches prior cached spans by content
        # regardless of whether they're still explicitly marked, so only the
        # newest breakpoint needs to be live.
        base = [
            *(m.model_copy(update={"cache": False}) if m.cache else m for m in node.messages),
            Message(role="assistant", content=node.target_reply, cache=True),
        ]
        # `base` is identical across every tactic branch below (and across each
        # branch's self-consistency samples) — the trailing cache=True is the
        # shared-prefix breakpoint so only the first call pays full price for it.

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
        """Target replies, judge evaluates, scorer scores — no shared state touched.

        Safe to call concurrently: builds a Node and its Verdicts purely from
        its arguments, without reading or writing the tree.
        """
        response = self._target.respond(messages)
        node = Node(
            run_id=run_id,
            parent_id=parent_id,
            messages=messages,
            target_reply=response.final_reply,
            tactic=tactic,
        )
        if self._scorer is not None:
            node.instability_score = self._scorer.score(messages, self._sample_n)
        else:
            node.instability_score = 0.0

        verdicts = self._judge.judge(response, run_id=run_id, node_id=node.id)
        return node, verdicts

    def _commit(
        self,
        tree: ConversationTree,
        node: Node,
        verdicts: list[Verdict],
        *,
        failures: list[Verdict],
    ) -> Node:
        """Write a computed node and its verdicts onto the shared tree (Blackboard)."""
        tree.add(node)
        tree.attach_verdicts(node.id, verdicts)
        failures.extend(v for v in verdicts if not v.passed)
        return node
