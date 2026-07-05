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
    ) -> None:
        self._simulator = simulator
        self._target = target
        self._judge = judge
        self._scorer = scorer
        self._tactics = tactics if tactics is not None else default_registry().names()
        self._sample_n = sample_n

    def search(
        self, tree: ConversationTree, seed_messages: list[Message], *, budget: int
    ) -> SearchResult:
        frontier = Frontier()
        failures: list[Verdict] = []
        nodes_created = 0

        root = self._evaluate(
            tree, seed_messages, parent_id=None, tactic=None, failures=failures
        )
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
        """Generate one adversarial child per tactic and evaluate each."""
        base = [*node.messages, Message(role="assistant", content=node.target_reply)]
        children: list[Node] = []
        for tactic in self._tactics:
            probe = self._simulator.simulate(base, tactic)
            child = self._evaluate(
                tree, [*base, probe], parent_id=node.id, tactic=tactic, failures=failures
            )
            children.append(child)
        return children

    def _evaluate(
        self,
        tree: ConversationTree,
        messages: list[Message],
        *,
        parent_id: str | None,
        tactic: str | None,
        failures: list[Verdict],
    ) -> Node:
        """Target replies, judge evaluates, scorer scores — one loop iteration.

        This is the single point where the four workers meet, writing their
        results onto the shared tree.
        """
        response = self._target.respond(messages)
        node = Node(
            run_id=tree.run_id,
            parent_id=parent_id,
            messages=messages,
            target_reply=response.final_reply,
            tactic=tactic,
        )
        if self._scorer is not None:
            node.instability_score = self._scorer.score(messages, self._sample_n)
        else:
            node.instability_score = 0.0

        tree.add(node)
        verdicts = self._judge.judge(response, run_id=tree.run_id, node_id=node.id)
        tree.attach_verdicts(node.id, verdicts)
        failures.extend(v for v in verdicts if not v.passed)
        return node
