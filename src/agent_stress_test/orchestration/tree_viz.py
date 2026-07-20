"""A simple color-coded vertical conversation-tree visualization.

One lane per leaf node, each a straight root-to-leaf line — correct for any
tree shape without graph-layout math, at the cost of repeating a shared
prefix across lanes wherever the tree actually branches.
"""

from dataclasses import dataclass
from typing import Literal

from agent_stress_test.models import Verdict
from agent_stress_test.orchestration.search import graded_proximity
from agent_stress_test.orchestration.tree import ConversationTree

NodeStatus = Literal["fail", "near_miss", "pass"]


@dataclass(frozen=True)
class TreeVizNode:
    node_id: str
    status: NodeStatus
    tactic: str | None


@dataclass(frozen=True)
class TreeVizLane:
    """One root-to-leaf path, root first."""

    leaf_node_id: str
    label: str
    nodes: list[TreeVizNode]


def _node_status(node_verdicts: list[Verdict]) -> NodeStatus:
    if any(not verdict.passed for verdict in node_verdicts):
        return "fail"
    if graded_proximity(node_verdicts) > 0:
        return "near_miss"
    return "pass"


def build_tree_viz(tree: ConversationTree, verdicts: list[Verdict]) -> list[TreeVizLane]:
    """One lane per leaf (a node no other node names as its parent)."""
    by_node: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        by_node.setdefault(verdict.node_id, []).append(verdict)

    nodes = tree.nodes()
    parent_ids = {node.parent_id for node in nodes if node.parent_id is not None}
    leaves = [node for node in nodes if node.id not in parent_ids]

    lanes = []
    for leaf in leaves:
        path = tree.path_to_root(leaf.id)
        viz_nodes = [
            TreeVizNode(
                node_id=node.id,
                status=_node_status(by_node.get(node.id, [])),
                tactic=node.tactic,
            )
            for node in path
        ]
        lanes.append(
            TreeVizLane(
                leaf_node_id=leaf.id,
                label=path[-1].tactic or leaf.id[:8],
                nodes=viz_nodes,
            )
        )
    return lanes
