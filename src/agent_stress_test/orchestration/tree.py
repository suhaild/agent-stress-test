"""Conversation tree structure — the Blackboard the simulator, target, judge,
and search collaborate through instead of calling each other directly.

The dashboard reads a run's tree from a different thread than the one the
search writes it from, so every access below takes ``_lock`` — a plain
``dict``/``list`` mutation isn't atomic across bytecode boundaries in
CPython. An ``RLock`` (not a plain ``Lock``) is required because some
methods below call other locked methods on ``self``.
"""

import threading

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
        self._lock = threading.RLock()

    def add(self, node: Node) -> Node:
        """Insert a node and register it under its parent (or as a root)."""
        with self._lock:
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
        """Record the verdicts a judge produced for a node (a blackboard
        write). Accumulates rather than replaces — a conversation-level
        judge attaches a second batch to a leaf node that already holds its
        own per-turn verdicts. Also sets ``node.verdict_id`` to the first
        failing verdict ever received, once only."""
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Unknown node {node_id}.")
            self._verdicts.setdefault(node_id, []).extend(verdicts)
            if self._nodes[node_id].verdict_id is None:
                failing = [v for v in verdicts if not v.passed]
                if failing:
                    self._nodes[node_id].verdict_id = failing[0].id

    def get(self, node_id: str) -> Node:
        with self._lock:
            return self._nodes[node_id]

    def children(self, node_id: str) -> list[Node]:
        with self._lock:
            return [self._nodes[cid] for cid in self._children.get(node_id, [])]

    def roots(self) -> list[Node]:
        with self._lock:
            return [self._nodes[rid] for rid in self._root_ids]

    def nodes(self) -> list[Node]:
        with self._lock:
            return list(self._nodes.values())

    def verdicts(self, node_id: str) -> list[Verdict]:
        with self._lock:
            return list(self._verdicts.get(node_id, []))

    def all_verdicts(self) -> list[Verdict]:
        with self._lock:
            return [verdict for verdicts in self._verdicts.values() for verdict in verdicts]

    def failures(self) -> list[Verdict]:
        """Every failed verdict across the tree."""
        with self._lock:
            return [verdict for verdict in self.all_verdicts() if not verdict.passed]

    def path_to_root(self, node_id: str) -> list[Node]:
        """The lineage from the root down to ``node_id`` (root first)."""
        with self._lock:
            lineage: list[Node] = []
            current: str | None = node_id
            while current is not None:
                node = self._nodes[current]
                lineage.append(node)
                current = node.parent_id
            lineage.reverse()
            return lineage
