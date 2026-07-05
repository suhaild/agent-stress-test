"""Conversation tree structure — the Blackboard.

The tree is the shared knowledge space the workers collaborate through: the
simulator writes probes (as user messages on new nodes), the target writes
replies, the judge writes verdicts, and the search reads scores/verdicts to
pick the next node to expand. Components never call each other directly; they
read and write here. This module is a pure data structure — it imports only
the data models and holds no provider, port, or search logic.
"""

from agent_stress_test.models import Node, Verdict


class ConversationTree:
    """Nodes with parent/child links plus the verdicts attached to each node.

    Keyed by ``node.id``. Roots (nodes with no parent) are kept in insertion
    order so a run's seeds stay stable and reproducible.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._nodes: dict[str, Node] = {}
        self._children: dict[str, list[str]] = {}
        self._verdicts: dict[str, list[Verdict]] = {}
        self._root_ids: list[str] = []

    def add(self, node: Node) -> Node:
        """Insert a node and register it under its parent (or as a root)."""
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id} is already in the tree.")
        self._nodes[node.id] = node
        self._children.setdefault(node.id, [])
        if node.parent_id is None:
            self._root_ids.append(node.id)
        else:
            if node.parent_id not in self._nodes:
                raise KeyError(f"Unknown parent {node.parent_id} for node {node.id}.")
            self._children[node.parent_id].append(node.id)
        return node

    def attach_verdicts(self, node_id: str, verdicts: list[Verdict]) -> None:
        """Record the verdicts a judge produced for a node (a blackboard write).

        Also points ``node.verdict_id`` at the first failing verdict, if any, so
        a failing node carries a direct handle to its failure. This is a plain
        attribute assignment on an already-defined, nullable field — no model
        change.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node {node_id}.")
        self._verdicts[node_id] = list(verdicts)
        failing = [v for v in verdicts if not v.passed]
        if failing:
            self._nodes[node_id].verdict_id = failing[0].id

    def get(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def children(self, node_id: str) -> list[Node]:
        return [self._nodes[cid] for cid in self._children.get(node_id, [])]

    def roots(self) -> list[Node]:
        return [self._nodes[rid] for rid in self._root_ids]

    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def verdicts(self, node_id: str) -> list[Verdict]:
        return list(self._verdicts.get(node_id, []))

    def all_verdicts(self) -> list[Verdict]:
        return [verdict for verdicts in self._verdicts.values() for verdict in verdicts]

    def failures(self) -> list[Verdict]:
        """Every failed verdict across the tree."""
        return [verdict for verdict in self.all_verdicts() if not verdict.passed]

    def path_to_root(self, node_id: str) -> list[Node]:
        """The lineage from the root down to ``node_id`` (root first)."""
        lineage: list[Node] = []
        current: str | None = node_id
        while current is not None:
            node = self._nodes[current]
            lineage.append(node)
            current = node.parent_id
        lineage.reverse()
        return lineage
