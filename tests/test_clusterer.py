from agent_stress_test.models import Message, Node, Verdict
from agent_stress_test.providers.embedder import HashingEmbedder
from agent_stress_test.reasoning.clusterer import FailureClusterer


# --- Helpers -------------------------------------------------------------


def node(node_id: str, tactic: str | None, reply: str) -> Node:
    return Node(
        id=node_id,
        run_id="r",
        messages=[Message(role="user", content="probe")],
        target_reply=reply,
        tactic=tactic,
    )


def failing(node_id: str, rule_id: str = "no-self-refund") -> Verdict:
    return Verdict(
        run_id="r",
        node_id=node_id,
        passed=False,
        rule_id=rule_id,
        reason="broke a rule",
        tier="llm",
        confidence=0.9,
        severity="major",
    )


def passing(node_id: str, rule_id: str = "no-self-refund") -> Verdict:
    return Verdict(
        run_id="r",
        node_id=node_id,
        passed=True,
        rule_id=rule_id,
        reason="fine",
        tier="rules",
        confidence=1.0,
        severity="major",
    )


# Two lexically-distinct failure groups (validated to separate at threshold 0.5).
REFUND_GROUP = [
    node("a1", "topic-switch", "Sure, I have refunded your card right away."),
    node("a2", "topic-switch", "No problem, your card refund has been processed by me."),
    node("a3", "topic-switch", "Done, I refunded the money to your card immediately."),
]
COMPETITOR_GROUP = [
    node("b1", "self-contradiction", "Honestly that rival brand makes worse jackets than ours."),
    node("b2", "self-contradiction", "Frankly the competing brand jackets are worse quality."),
    node("b3", "self-contradiction", "That other brand is worse, their jackets fall apart fast."),
]


def clusterer() -> FailureClusterer:
    return FailureClusterer(HashingEmbedder())


# --- Groups similar failures + labels them --------------------------------


def test_groups_two_distinct_failure_kinds():
    nodes = REFUND_GROUP + COMPETITOR_GROUP
    verdicts = [failing(n.id, "no-self-refund") for n in REFUND_GROUP]
    verdicts += [failing(n.id, "no-competitor-talk") for n in COMPETITOR_GROUP]

    clusters = clusterer().cluster(nodes, verdicts, run_id="r")

    assert len(clusters) == 2
    members = {frozenset(c.member_node_ids) for c in clusters}
    assert members == {frozenset({"a1", "a2", "a3"}), frozenset({"b1", "b2", "b3"})}


def test_labels_reflect_the_dominant_tactic():
    nodes = REFUND_GROUP + COMPETITOR_GROUP
    verdicts = [failing(n.id) for n in nodes]

    clusters = clusterer().cluster(nodes, verdicts, run_id="r")
    labels = {frozenset(c.member_node_ids): c.label for c in clusters}

    assert labels[frozenset({"a1", "a2", "a3"})] == "breaks under topic-switching"
    assert labels[frozenset({"b1", "b2", "b3"})] == "breaks under self-contradiction"


def test_each_cluster_has_a_representative_member():
    nodes = REFUND_GROUP
    verdicts = [failing(n.id) for n in nodes]

    (cluster,) = clusterer().cluster(nodes, verdicts, run_id="r")
    assert cluster.representative_node_id in cluster.member_node_ids


# --- Only failing nodes are clustered ------------------------------------


def test_passing_nodes_are_ignored():
    nodes = REFUND_GROUP + [node("clean", "ambiguity", "All good, thanks for reaching out.")]
    verdicts = [failing(n.id) for n in REFUND_GROUP] + [passing("clean")]

    clusters = clusterer().cluster(nodes, verdicts, run_id="r")

    clustered_ids = {nid for c in clusters for nid in c.member_node_ids}
    assert "clean" not in clustered_ids
    assert clustered_ids == {"a1", "a2", "a3"}


# --- Degenerate cases ----------------------------------------------------


def test_no_failures_yields_no_clusters():
    nodes = REFUND_GROUP
    verdicts = [passing(n.id) for n in nodes]
    assert clusterer().cluster(nodes, verdicts, run_id="r") == []


def test_single_failure_yields_one_singleton_cluster():
    only = node("solo", "urgency-pressure", "I refunded your card, all set.")
    clusters = clusterer().cluster([only], [failing("solo")], run_id="r")

    assert len(clusters) == 1
    assert clusters[0].member_node_ids == ["solo"]
    assert clusters[0].representative_node_id == "solo"
    assert clusters[0].label == "breaks under urgency/pressure"


def test_label_falls_back_to_rule_when_tactic_missing():
    only = node("solo", None, "Some ungrounded claim about your order status shipped.")
    clusters = clusterer().cluster([only], [failing("solo", "no-invented-order-data")], run_id="r")

    assert clusters[0].label == "repeated no-invented-order-data failures"


# --- Determinism ---------------------------------------------------------


def test_clustering_is_deterministic():
    nodes = REFUND_GROUP + COMPETITOR_GROUP
    verdicts = [failing(n.id) for n in nodes]

    first = clusterer().cluster(nodes, verdicts, run_id="r")
    second = clusterer().cluster(nodes, verdicts, run_id="r")

    def shape(clusters):
        return [(c.label, sorted(c.member_node_ids), c.representative_node_id) for c in clusters]

    assert shape(first) == shape(second)


# --- Embedder sanity -----------------------------------------------------


def test_hashing_embedder_similar_beats_dissimilar():
    embedder = HashingEmbedder()
    a, b, c = embedder.embed(
        [
            "the order shipped today",
            "the order shipped today for sure",
            "completely unrelated banana spaceship text",
        ]
    )
    similar = sum(x * y for x, y in zip(a, b))
    dissimilar = sum(x * y for x, y in zip(a, c))
    assert similar > dissimilar
