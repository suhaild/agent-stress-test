from agent_stress_test.models import Message, Node, Verdict
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.orchestration.tree_viz import TreeVizNode, build_tree_viz


def _node(node_id: str, parent_id: str | None, tactic: str | None = None) -> Node:
    return Node(
        id=node_id,
        run_id="r",
        parent_id=parent_id,
        messages=[Message(role="user", content="hi")],
        target_reply="ok",
        tactic=tactic,
    )


def _verdict(node_id: str, *, passed: bool, tier="rules", confidence=1.0) -> Verdict:
    return Verdict(
        run_id="r",
        node_id=node_id,
        passed=passed,
        rule_id="r1",
        reason="x",
        tier=tier,
        confidence=confidence,
        severity="major",
    )


def test_build_tree_viz_empty_tree():
    assert build_tree_viz(ConversationTree("r"), []) == []


def test_build_tree_viz_one_lane_per_leaf_root_to_leaf_order():
    tree = ConversationTree("r")
    root = _node("root", None, tactic="hostile")
    child = _node("child", "root", tactic="hostile")
    tree.add(root)
    tree.add(child)

    lanes = build_tree_viz(tree, [])

    assert len(lanes) == 1
    lane = lanes[0]
    assert lane.leaf_node_id == "child"
    assert lane.label == "hostile"
    assert [n.node_id for n in lane.nodes] == ["root", "child"]
    assert all(n.status == "pass" for n in lane.nodes)


def test_build_tree_viz_one_lane_per_persona_forest():
    tree = ConversationTree("r")
    tree.add(_node("a-root", None, tactic="hostile"))
    tree.add(_node("b-root", None, tactic="urgency-pressure"))

    lanes = build_tree_viz(tree, [])

    assert {lane.label for lane in lanes} == {"hostile", "urgency-pressure"}
    assert all(len(lane.nodes) == 1 for lane in lanes)


def test_build_tree_viz_colors_a_failing_node_as_fail():
    tree = ConversationTree("r")
    tree.add(_node("root", None, tactic="hostile"))
    lanes = build_tree_viz(tree, [_verdict("root", passed=False)])

    assert lanes[0].nodes[0] == TreeVizNode(node_id="root", status="fail", tactic="hostile")


def test_build_tree_viz_colors_a_low_confidence_pass_as_near_miss():
    tree = ConversationTree("r")
    tree.add(_node("root", None, tactic="hostile"))
    lanes = build_tree_viz(tree, [_verdict("root", passed=True, tier="llm", confidence=0.3)])

    assert lanes[0].nodes[0].status == "near_miss"


def test_build_tree_viz_branching_draws_a_lane_per_leaf():
    tree = ConversationTree("r")
    tree.add(_node("root", None, tactic="hostile"))
    tree.add(_node("child-1", "root", tactic="hostile"))
    tree.add(_node("child-2", "root", tactic="hostile"))

    lanes = build_tree_viz(tree, [])

    assert len(lanes) == 2
    leaf_ids = {lane.leaf_node_id for lane in lanes}
    assert leaf_ids == {"child-1", "child-2"}
    for lane in lanes:
        assert [n.node_id for n in lane.nodes][0] == "root"
