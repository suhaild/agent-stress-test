"""Greedy best-first search — the Evaluator-Optimizer loop.

This is the core engine cycle: the simulator generates an adversarial turn, the
target replies, the judge evaluates it, the scorer estimates instability, and
the search steers toward the most promising failures. Search sits behind a
``SearchStrategy`` interface (Strategy pattern) so an MCTS strategy can slot in
later (Phase 11) without touching the runner.

The frontier is ordered by ``instability + judge proximity-to-failure``: nodes
whose replies are shaky or already show (or nearly show) a rule violation are
expanded first, because that is where more failures tend to live.

Phase C5 sharpens that steering signal without adding any new dependency
(``ours``, no library import):
  1. ``node_priority`` folds in a GRADED proximity, not just the discrete
     severity of an outright failure — a passing LLM/metric-tier verdict that
     barely cleared its own pass/fail threshold (low ``confidence``, see
     ``graded_proximity``) is still worth probing further, not a clean 0.
  2. the frontier explicitly PRUNES nodes whose priority falls below
     ``prune_floor`` — off by default (``0.0``, nothing pruned, identical to
     pre-C5 behavior) so a caller opts in deliberately rather than every
     existing run's breadth silently changing.
  3. a deflecting reply (refusal/non-answer/redirect — see
     ``reasoning/judge.py``'s ``is_deflection``) is tracked as its OWN signal,
     both nudging priority and reported on
     ``SearchResult.exploration_detail.deflections`` — distinct from a
     genuine rule pass, since dodging the question isn't the same as
     answering it correctly.
  4. the single closest near-miss (the highest-graded-proximity node that
     never actually failed) is tracked across the whole search and reported
     as ``SearchResult.exploration_detail.near_miss`` — a first-class
     result, not just a number buried in the tree.
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

# How "fertile" a failure of each severity makes a node's region. Critical
# failures point at the most valuable branches to keep expanding.
SEVERITY_WEIGHT: dict[Severity, float] = {"minor": 0.34, "major": 0.67, "critical": 1.0}

# How much a deflecting reply alone raises priority — a deflection hasn't
# broken any specific rule, but an agent that dodges under pressure is still
# worth probing further, roughly on par with a "major" outright failure.
DEFLECTION_WEIGHT = 0.67


def failure_proximity(verdicts: list[Verdict]) -> float:
    """How close this node is to (or into) failure, in [0, 1].

    The max severity weight among *failed* verdicts, or 0.0 if none failed. A
    node that already carries a critical failure scores 1.0 — the most promising
    region to keep probing. Kept as its own function (rather than folded into
    ``graded_proximity`` below) because ``orchestration/reliability.py``'s
    ``SeverityWeightedModel`` reuses this exact discrete formula for the
    reliability score, which C5 deliberately leaves unchanged — only search
    steering (``node_priority``) gets the graded near-miss upgrade.
    """
    weights = [SEVERITY_WEIGHT[v.severity] for v in verdicts if not v.passed]
    return max(weights, default=0.0)


def graded_proximity(verdicts: list[Verdict]) -> float:
    """How close this node is to failing, in [0, 1] — like ``failure_proximity``,
    but a passing LLM/metric-tier verdict sitting close to its own pass/fail
    threshold also counts as a near-miss instead of a clean 0.

    A verdict's ``confidence`` is already a distance-from-threshold signal
    (see ``reasoning/judge.py``'s ``_confidence_from_score``): low confidence
    on a PASSING verdict means it barely cleared the bar. Deterministic
    tier-1 rule checks always report ``confidence=1.0`` (see
    ``DETERMINISTIC_CONFIDENCE``), so a passing tier-1 verdict correctly
    contributes 0.0 here — a regex match has no "almost".
    """
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
    (0.0 without a scorer, to skip its extra target calls) and returns the
    judge's verdicts for the caller to commit onto the tree.
    """
    node.instability_score = scorer.score(node.messages, sample_n) if scorer is not None else 0.0
    return judge.judge(response, run_id=run_id, node_id=node.id, conversation=node.messages)


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
class ExplorationDetail:
    """Frontier-exploration signals only a priority-driven search over a
    shared tree can produce — see ``SearchResult.exploration_detail``.

    ``near_miss`` (Phase C5) is the passing node with the highest
    ``graded_proximity`` seen anywhere in the search — the closest the target
    came to failing without actually failing — or ``None`` if every node was
    either a clean pass (0 proximity) or an outright failure (already in
    ``SearchResult.failures``). ``deflections`` is every node id whose reply
    was flagged by ``reasoning/judge.py``'s ``is_deflection`` — its own
    signal, tracked separately so a deflection is never mistaken for a
    genuine rule pass.
    """

    near_miss: Node | None = None
    deflections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    """Summary of one search. The tree (Blackboard) holds the full detail.

    ``exploration_detail`` is populated only by strategies that maintain a
    priority frontier over a shared tree (``GreedyBestFirstSearch``) — it is
    ``None``, not an empty ``ExplorationDetail``, for strategies that don't
    track this concept at all (``DeepEvalConversationSearch``, which ingests
    independent per-persona conversations with no shared frontier to rank).
    That distinction matters: ``None`` means "this strategy doesn't measure
    near-misses/deflections," while a populated ``ExplorationDetail`` with
    ``near_miss=None`` means "measured, and found none" — collapsing the two
    into shared top-level fields defaulting to "nothing found" would make
    them indistinguishable, which is exactly the trap a caller relying on
    ``SearchStrategy``'s contract could fall into.
    """

    expansions: int
    nodes_created: int
    failures: list[Verdict] = field(default_factory=list)
    exploration_detail: ExplorationDetail | None = None


class SearchStrategy(ABC):
    """A strategy for exploring the conversation tree (Strategy pattern).

    Every implementation must reliably populate ``SearchResult.expansions``/
    ``.nodes_created``/``.failures`` — the only fields a caller may read
    without checking which concrete strategy produced the result.
    ``exploration_detail`` is optional and strategy-specific (see its
    docstring); ``budget`` and ``seed_messages`` may also carry a
    strategy-specific meaning (see e.g. ``DeepEvalConversationSearch``'s
    docstring) despite sharing this one method signature.
    """

    @abstractmethod
    def search(
        self, tree: ConversationTree, seed_messages: list[Message], *, budget: int
    ) -> SearchResult: ...


class GreedyBestFirstSearch(SearchStrategy):
    """Expand the highest-priority frontier node until the budget is spent.

    The four workers are injected (Dependency Injection); this strategy owns
    only the loop and never constructs them. ``scorer`` is optional — it costs
    extra target calls per node, so a caller may skip it to save cost/latency
    regardless of target type; without it instability is treated as 0.0 and
    priority is driven by judge proximity-to-failure alone.

    ``prune_floor`` (Phase C5) drops a node from the frontier — it is still
    created, judged, and counted in ``nodes_created``/``failures``, just never
    expanded further — once its ``node_priority`` falls below the floor.
    Defaults to ``0.0``, which prunes nothing (every priority is already
    ``>= 0.0``): a target with zero signal anywhere still gets its full
    budget of expansions by default, identical to pre-C5 behavior. Raise it
    to actually skip low-value branches once ``node_priority`` is graded
    finely enough (see ``graded_proximity``) to make that judgment call.
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
        """Write a computed node and its verdicts onto the shared tree (Blackboard)."""
        tree.add(node)
        tree.attach_verdicts(node.id, verdicts)
        failures.extend(v for v in verdicts if not v.passed)
        return node
