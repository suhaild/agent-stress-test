import re
from pathlib import Path

import pytest

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentResponse, Message, Node, Verdict
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.orchestration.search import (
    Frontier,
    GreedyBestFirstSearch,
    SearchResult,
    failure_proximity,
    node_priority,
)
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import TargetAgent
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.judge import RulesJudge, build_checks
from agent_stress_test.reasoning.simulator import Simulator, default_registry
from agent_stress_test.targets.python_fn import PythonFunctionAgent

TACTIC_COUNT = len(default_registry().names())


# --- Helpers -------------------------------------------------------------


def seed() -> list[Message]:
    return [Message(role="user", content="Hi, I need help with my order.")]


def verdict(passed: bool, severity: str = "major") -> Verdict:
    return Verdict(
        run_id="r",
        node_id="n",
        passed=passed,
        reason="because",
        tier="rules",
        confidence=1.0,
        severity=severity,
    )


def a_node(instability: float | None = None) -> Node:
    return Node(
        run_id="r",
        messages=[Message(role="user", content="hi")],
        target_reply="ok",
        instability_score=instability,
    )


class CountingTarget(TargetAgent):
    """Wraps a target and counts how many times it is asked to respond."""

    def __init__(self, inner: TargetAgent) -> None:
        self._inner = inner
        self.calls = 0

    def respond(self, conversation: list[Message]) -> AgentResponse:
        self.calls += 1
        return self._inner.respond(conversation)


def clean_target() -> PythonFunctionAgent:
    # Never trips a rule: no return/refund trigger, no competitor, no order data.
    return PythonFunctionAgent(lambda _conv: "Happy to help. Let me know what you need.")


def planted_fn(conversation: list[Message]) -> str:
    """A target with a failure planted two tactics deep.

    self-contradiction alone -> a minor competitor slip (near miss);
    self-contradiction THEN urgency-pressure -> a critical self-issued refund.
    Tactic markers ride in on the simulated user turns under the fake provider.
    """
    user_text = " ".join(m.content for m in conversation if m.role == "user")
    has_sc = "[self-contradiction]" in user_text
    has_urgency = "[urgency-pressure]" in user_text
    if has_sc and has_urgency:
        return "Sure — I've already refunded your card."  # no-self-refund (critical)
    if has_sc:
        return "Honestly, Patagonia makes similar gear. Anyway, our 30-day return window applies."
    return "Happy to help. Let me know what you need."


def make_strategy(spec_path: Path, target: TargetAgent, **kwargs) -> GreedyBestFirstSearch:
    spec = load_agent_spec(spec_path)
    return GreedyBestFirstSearch(
        Simulator(FakeLLMProvider()),
        target,
        RulesJudge(build_checks(spec)),
        **kwargs,
    )


# --- Priority signal (pure, unit) ----------------------------------------


def test_failure_proximity_scales_with_severity():
    assert failure_proximity([]) == 0.0
    assert failure_proximity([verdict(True, "critical")]) == 0.0  # passed -> not fertile
    minor = failure_proximity([verdict(False, "minor")])
    major = failure_proximity([verdict(False, "major")])
    critical = failure_proximity([verdict(False, "critical")])
    assert 0.0 < minor < major < critical == 1.0


def test_failure_proximity_takes_the_worst_failure():
    mixed = [verdict(False, "minor"), verdict(True, "major"), verdict(False, "critical")]
    assert failure_proximity(mixed) == 1.0


def test_node_priority_sums_instability_and_proximity():
    scored = a_node(instability=0.5)
    assert node_priority(scored, [verdict(False, "minor")]) == pytest.approx(0.5 + 0.34)
    # A missing instability score contributes 0, never crashes.
    assert node_priority(a_node(None), []) == 0.0


# --- Frontier ordering ---------------------------------------------------


def test_frontier_pops_highest_priority_first():
    frontier = Frontier()
    frontier.push("low", 0.2)
    frontier.push("high", 0.9)
    frontier.push("mid", 0.5)
    assert [frontier.pop(), frontier.pop(), frontier.pop()] == ["high", "mid", "low"]
    assert frontier.is_empty()


def test_frontier_breaks_ties_fifo():
    frontier = Frontier()
    frontier.push("first", 0.5)
    frontier.push("second", 0.5)
    frontier.push("third", 0.5)
    assert [frontier.pop(), frontier.pop(), frontier.pop()] == ["first", "second", "third"]


# --- Search reliably finds a planted failure -----------------------------


def test_search_finds_planted_failure(sample_agent_spec_path):
    strategy = make_strategy(sample_agent_spec_path, PythonFunctionAgent(planted_fn))
    tree = ConversationTree("run-plant")

    result = strategy.search(tree, seed(), budget=5)

    critical = [
        v for v in result.failures if v.rule_id == "no-self-refund" and v.severity == "critical"
    ]
    assert critical, "greedy search did not surface the planted critical failure"

    # It was reached by going self-contradiction -> urgency-pressure.
    lineage = tree.path_to_root(critical[0].node_id)
    user_text = " ".join(m.content for node in lineage for m in node.messages if m.role == "user")
    assert "[self-contradiction]" in user_text
    assert "[urgency-pressure]" in user_text


def test_search_prefers_the_more_promising_branch(sample_agent_spec_path):
    # With the planted target, the minor-failure branch (proximity > 0) must be
    # expanded before the zero-signal siblings, which is what leads greedy to
    # the deeper critical failure within a small budget.
    strategy = make_strategy(sample_agent_spec_path, PythonFunctionAgent(planted_fn))
    tree = ConversationTree("run-branch")

    strategy.search(tree, seed(), budget=2)

    assert any(
        v.rule_id == "no-self-refund" for v in tree.failures()
    ), "two expansions down the best branch should already reach the planted failure"


# --- Budget respected + clean termination --------------------------------


def test_budget_caps_expansions(sample_agent_spec_path):
    strategy = make_strategy(sample_agent_spec_path, clean_target())
    tree = ConversationTree("run-budget")

    result = strategy.search(tree, seed(), budget=2)

    assert result.expansions == 2  # branching is unbounded, so budget is the limiter
    assert isinstance(result, SearchResult)


def test_zero_budget_only_seeds_the_root(sample_agent_spec_path):
    strategy = make_strategy(sample_agent_spec_path, clean_target())
    tree = ConversationTree("run-zero")

    result = strategy.search(tree, seed(), budget=0)

    assert result.expansions == 0
    assert len(tree.nodes()) == 1  # just the seeded root
    assert len(tree.roots()) == 1


def test_target_calls_are_bounded_by_budget(sample_agent_spec_path):
    counting = CountingTarget(clean_target())
    strategy = make_strategy(sample_agent_spec_path, counting)
    tree = ConversationTree("run-count")

    strategy.search(tree, seed(), budget=3)

    # One seed reply plus one reply per tactic on each of the 3 expansions.
    assert counting.calls == 1 + 3 * TACTIC_COUNT


# --- Full loop end-to-end via the runner ---------------------------------


def run_once(spec_path: Path):
    spec = load_agent_spec(spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(planted_fn),
        sim_provider=FakeLLMProvider(),
        scorer_provider=FakeLLMProvider(responses=["red", "green", "blue"], cycle=True),
    )
    return runner.run(provider_name="fake", budget=3)


def test_end_to_end_run_completes_and_populates_the_tree(sample_agent_spec_path):
    result = run_once(sample_agent_spec_path)

    assert result.run.status == "completed"
    assert result.run.started_at is not None and result.run.completed_at is not None
    assert result.run.final_score is None  # reliability score is Phase 7

    assert len(result.tree.roots()) == 1
    root = result.tree.roots()[0]
    assert len(result.tree.children(root.id)) == TACTIC_COUNT

    # Every node was judged and scored; instability is a bounded float.
    assert result.tree.all_verdicts()
    for node in result.tree.nodes():
        assert isinstance(node.instability_score, float)
        assert 0.0 <= node.instability_score <= 1.0

    # The planted failure is found through the runner too.
    assert any(v.rule_id == "no-self-refund" for v in result.failures)


def test_end_to_end_run_is_deterministic(sample_agent_spec_path):
    first = run_once(sample_agent_spec_path)
    second = run_once(sample_agent_spec_path)

    def fingerprint(res):
        return (
            len(res.tree.nodes()),
            sorted(v.rule_id for v in res.failures),
            sorted(n.instability_score for n in res.tree.nodes()),
        )

    assert fingerprint(first) == fingerprint(second)


# --- Layer boundary: orchestration stays free of adapters ----------------


def test_orchestration_imports_no_adapters():
    orchestration = Path(__file__).resolve().parents[1] / "src" / "agent_stress_test" / "orchestration"
    forbidden = re.compile(r"^\s*(?:import|from)\s+(?:litellm|httpx|sqlite3|sqlalchemy)\b", re.MULTILINE)
    offenders = [
        path.name
        for path in orchestration.glob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
