"""Failure clustering + naming.

Confirmed failures are embedded, grouped by similarity, and each group is named
so a run can say things like "breaks under topic-switching". Embeddings come in
through the injected ``Embedder`` port; clustering is a small in-memory
single-linkage agglomerative pass (union any two failures whose cosine distance
is below a threshold) — no vector database, no heavyweight clustering library.
"""

from collections import Counter
from dataclasses import dataclass

from agent_stress_test.models import Cluster, Node, Verdict
from agent_stress_test.ports import Embedder

# "topic-switch"/"ambiguity" are kept for reports on runs stored before those
# tactics were renamed in simulator.py.
_TACTIC_LABELS = {
    "self-contradiction": "breaks under self-contradiction",
    "urgency-pressure": "breaks under urgency/pressure",
    "scope-expansion": "breaks under scope expansion",
    "hostile": "breaks under hostility",
    "stale-recall": "breaks under stale-context reliance",
    "topic-switch": "breaks under topic-switching",
    "ambiguity": "breaks under ambiguity",
}


@dataclass(frozen=True)
class _FailurePoint:
    """A failing node reduced to what clustering needs."""

    node_id: str
    text: str
    tactic: str | None
    rule_ids: tuple[str, ...]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self._parent[self.find(i)] = self.find(j)


class FailureClusterer:
    """Groups confirmed failures into named clusters (Blackboard consumer)."""

    def __init__(self, embedder: Embedder, *, distance_threshold: float = 0.5) -> None:
        self._embedder = embedder
        self._threshold = distance_threshold

    def cluster(
        self, nodes: list[Node], verdicts: list[Verdict], *, run_id: str
    ) -> list[Cluster]:
        points = self._failure_points(nodes, verdicts)
        if not points:
            return []

        vectors = self._embedder.embed([point.text for point in points])
        components = self._connected_components(vectors)

        clusters: list[Cluster] = []
        for member_indices in components:
            members = [points[i] for i in member_indices]
            representative = self._representative(member_indices, vectors, points)
            clusters.append(
                Cluster(
                    run_id=run_id,
                    label=self._label(members),
                    member_node_ids=[point.node_id for point in members],
                    representative_node_id=representative,
                )
            )
        return clusters

    # --- failure selection -----------------------------------------------

    @staticmethod
    def _failure_points(nodes: list[Node], verdicts: list[Verdict]) -> list[_FailurePoint]:
        failing_rules: dict[str, list[str]] = {}
        for verdict in verdicts:
            if not verdict.passed:
                failing_rules.setdefault(verdict.node_id, []).append(verdict.rule_id or "")

        points: list[_FailurePoint] = []
        for node in nodes:
            rule_ids = failing_rules.get(node.id)
            if not rule_ids:
                continue
            text = f"{node.tactic or ''} {' '.join(sorted(rule_ids))} {node.target_reply}"
            points.append(
                _FailurePoint(
                    node_id=node.id,
                    text=text,
                    tactic=node.tactic,
                    rule_ids=tuple(sorted(rule_ids)),
                )
            )
        return points

    # --- clustering ------------------------------------------------------

    def _connected_components(self, vectors: list[list[float]]) -> list[list[int]]:
        n = len(vectors)
        uf = _UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if 1.0 - _dot(vectors[i], vectors[j]) < self._threshold:
                    uf.union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(i)
        # Order clusters by first member index for deterministic output.
        return [members for _, members in sorted(groups.items(), key=lambda kv: kv[1][0])]

    @staticmethod
    def _representative(
        member_indices: list[int], vectors: list[list[float]], points: list[_FailurePoint]
    ) -> str:
        """The medoid: the member with the highest mean similarity to the group."""
        best_index = member_indices[0]
        best_score = -1.0
        for i in member_indices:
            score = sum(_dot(vectors[i], vectors[j]) for j in member_indices if j != i)
            if score > best_score:
                best_score = score
                best_index = i
        return points[best_index].node_id

    # --- naming ----------------------------------------------------------

    @staticmethod
    def _label(members: list[_FailurePoint]) -> str:
        tactics = [point.tactic for point in members if point.tactic]
        if tactics:
            dominant, _ = Counter(tactics).most_common(1)[0]
            return _TACTIC_LABELS.get(dominant, f"breaks under {dominant}")

        rule_ids = [rule_id for point in members for rule_id in point.rule_ids if rule_id]
        if rule_ids:
            dominant, _ = Counter(rule_ids).most_common(1)[0]
            return f"repeated {dominant} failures"
        return "uncategorized failures"
