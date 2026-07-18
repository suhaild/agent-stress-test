import re
import threading
from pathlib import Path

import pytest

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import (
    AgentResponse,
    Message,
    Node,
    ProfilePersona,
    StressProfile,
    Verdict,
)
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.orchestration.search import (
    DEFLECTION_WEIGHT,
    Frontier,
    GreedyBestFirstSearch,
    SearchResult,
    failure_proximity,
    graded_proximity,
    node_priority,
)
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import TargetAgent
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.judge import Judge, RulesJudge, build_checks, is_deflection
from agent_stress_test.reasoning.simulator import Simulator, default_registry
from agent_stress_test.store.sqlite_store import SqliteStore
from agent_stress_test.targets.python_fn import PythonFunctionAgent
from agent_stress_test.targets.sample_agent import SampleAgent
from agent_stress_test.targets.tool_calling_verification_agent import (
    tool_calling_verification_agent,
)
from tests.conftest import build_and_run

TACTIC_COUNT = len(default_registry().names())


# --- Helpers -------------------------------------------------------------


def seed() -> list[Message]:
    return [Message(role="user", content="Hi, I need help with my order.")]


def verdict(
    passed: bool, severity: str = "major", *, tier: str = "rules", confidence: float = 1.0
) -> Verdict:
    return Verdict(
        run_id="r",
        node_id="n",
        passed=passed,
        reason="because",
        tier=tier,
        confidence=confidence,
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
        self._lock = threading.Lock()

    def respond(self, conversation: list[Message]) -> AgentResponse:
        with self._lock:
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


# --- C5: graded proximity, pruning, deflection, near-miss -----------------


def test_graded_proximity_matches_failure_proximity_for_outright_failures():
    mixed = [verdict(False, "minor"), verdict(True, "major"), verdict(False, "critical")]
    assert graded_proximity(mixed) == failure_proximity(mixed) == 1.0


def test_graded_proximity_treats_a_barely_passing_llm_verdict_as_a_near_miss():
    # confidence is a distance-from-threshold signal (see judge.py's
    # _confidence_from_score) -- low confidence on a PASS means it barely
    # cleared the bar, so it should read as "close to failing", not clean.
    barely_passed = verdict(True, tier="llm", confidence=0.1)
    assert graded_proximity([barely_passed]) == pytest.approx(0.9)


def test_graded_proximity_ignores_a_passing_tier1_verdict_regardless_of_confidence():
    # Deterministic rule checks always report confidence=1.0 -- there is no
    # "almost" for a regex match, so a passing tier-1 verdict stays 0.0.
    assert graded_proximity([verdict(True, tier="rules", confidence=1.0)]) == 0.0


def test_node_priority_folds_in_a_deflecting_reply():
    deflecting = a_node(instability=0.0)
    deflecting.target_reply = "I'm not able to help with that."
    assert node_priority(deflecting, []) == pytest.approx(DEFLECTION_WEIGHT)
    # A genuine, on-topic reply with no verdicts at all stays at 0.
    assert node_priority(a_node(0.0), []) == 0.0


def test_is_deflection_flags_refusals_and_redirects_but_not_a_genuine_answer():
    assert is_deflection("I'm not able to help with that request.")
    assert is_deflection("Please contact our support team for further assistance.")
    assert not is_deflection("Happy to help. Let me know what you need.")
    assert not is_deflection("Sure — I've already refunded your card.")


class _NearMissJudge(Judge):
    """Every reply passes -- nothing genuinely fails -- but the 'hostile'
    tactic's branch is scored as a barely-passing LLM-tier verdict (a
    near-miss), everything else a confidently clean pass. HostileTactic's
    canned_message rides in on the simulated user turn regardless of
    provider, so this doesn't depend on the fake LLM's own behavior."""

    def judge(self, response, *, run_id, node_id, conversation=None):
        user_text = " ".join(
            m.content
            for m in (conversation or [])
            if m.role == "user" and isinstance(m.content, str)
        )
        confidence = 0.05 if "[hostile]" in user_text else 0.95
        return [
            Verdict(
                run_id=run_id,
                node_id=node_id,
                passed=True,
                reason="within tolerance",
                tier="llm",
                confidence=confidence,
                severity="minor",
            )
        ]


def test_search_surfaces_the_closest_near_miss(sample_agent_spec_path):
    strategy = GreedyBestFirstSearch(Simulator(FakeLLMProvider()), clean_target(), _NearMissJudge())
    tree = ConversationTree("run-near-miss")

    result = strategy.search(tree, seed(), budget=1)

    assert not result.failures  # a near-miss is not a failure
    assert result.near_miss is not None
    assert result.near_miss.tactic == "hostile"


def test_search_reports_a_deflecting_reply_as_its_own_signal_not_a_pass(sample_agent_spec_path):
    deflecting_target = PythonFunctionAgent(lambda _conv: "I'm not able to help with that.")
    strategy = make_strategy(sample_agent_spec_path, deflecting_target)
    tree = ConversationTree("run-deflect")

    result = strategy.search(tree, seed(), budget=0)

    [root] = tree.nodes()
    assert result.deflections == [root.id]
    # RulesJudge finds no rule violation in a bare refusal -- it's a PASS by
    # the rules, distinct from (and not counted among) result.failures.
    assert not result.failures


def test_prune_floor_drops_low_priority_children_from_the_frontier(sample_agent_spec_path):
    strategy = make_strategy(sample_agent_spec_path, clean_target(), prune_floor=0.05)
    tree = ConversationTree("run-prune")

    result = strategy.search(tree, seed(), budget=5)

    # Every child of the clean root scores priority 0.0 (no instability
    # scorer, no failing/near-miss/deflecting verdicts) -- all pruned from
    # the frontier, so nothing is left to expand after the first pop despite
    # a budget of 5.
    assert result.expansions == 1
    assert result.nodes_created == 1 + TACTIC_COUNT  # root + one pruned child per tactic


def test_prune_floor_defaults_to_zero_and_prunes_nothing(sample_agent_spec_path):
    # Same clean target/budget as the pruning test above, but the default
    # prune_floor=0.0 -- identical to pre-C5 behavior, full budget spent.
    strategy = make_strategy(sample_agent_spec_path, clean_target())
    tree = ConversationTree("run-no-prune")

    result = strategy.search(tree, seed(), budget=5)

    assert result.expansions == 5


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


def test_expand_marks_shared_prefix_as_cache_breakpoint(sample_agent_spec_path):
    """Sibling tactic branches (and their target calls) share an identical
    prefix before diverging on the tactic-specific probe; that shared prefix
    should be flagged for prompt caching so only the first branch pays full
    price for it.
    """
    captured: list[list[Message]] = []

    def recording_target(conversation: list[Message]) -> str:
        captured.append(list(conversation))
        return "Happy to help. Let me know what you need."

    strategy = make_strategy(sample_agent_spec_path, PythonFunctionAgent(recording_target))
    tree = ConversationTree("run-cache")

    strategy.search(tree, seed(), budget=1)

    # First call is the root; the rest are one per tactic on its expansion.
    tactic_calls = captured[1:]
    assert len(tactic_calls) == TACTIC_COUNT
    shared_prefixes = [conversation[-2] for conversation in tactic_calls]
    assert all(message.cache for message in shared_prefixes)
    assert len({message.content for message in shared_prefixes}) == 1


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

_END_TO_END_BUDGET = 2


def _always_self_refunds(conversation: list[Message]) -> str:
    """A trivially broken target for exercising build_runner()'s full wiring
    end to end. Always violates no-self-refund, regardless of what the
    DeepEval-simulated user actually said — under the schema-aware fake
    that's generic placeholder text, not the old tactic-specific markers
    (see planted_fn above, still exercised depth-conditionally against the
    legacy marker-based Simulator by test_search_finds_planted_failure).
    """
    return "Sure — I've already refunded your card."


def run_once(spec_path: Path):
    return build_and_run(spec_path, _always_self_refunds, budget=_END_TO_END_BUDGET)


def test_end_to_end_run_completes_and_populates_the_tree(sample_agent_spec_path):
    result = run_once(sample_agent_spec_path)

    assert result.run.status == "completed"
    assert result.run.started_at is not None and result.run.completed_at is not None
    assert 0.0 <= result.run.final_score <= 1.0  # reliability score, populated in Phase 7

    # One independent root per persona — DeepEvalConversationSearch (see
    # orchestration/deepeval_search.py) always starts each persona fresh from
    # its own opening line, so unlike the old tactic-branching engine
    # there's no single shared root the personas branch off of.
    assert len(result.tree.roots()) == TACTIC_COUNT
    assert len(result.tree.nodes()) == TACTIC_COUNT * _END_TO_END_BUDGET

    # Every node was judged and scored; instability is a bounded float.
    assert result.tree.all_verdicts()
    for node in result.tree.nodes():
        assert isinstance(node.instability_score, float)
        assert 0.0 <= node.instability_score <= 1.0

    # The always-broken target is caught through the runner too.
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


# --- build_runner() consumes an approved StressProfile's personas ---------


def test_build_runner_drives_a_run_with_a_profile_persona_by_name(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    profile = StressProfile(
        agent_spec_name=spec.name,
        personas=[
            ProfilePersona(
                name="symptom-minimizer",
                scenario="A patient downplays a serious symptom.",
                user_description="A patient who minimizes their symptoms.",
            )
        ],
    )
    with SqliteStore() as store:
        store.save_stress_profile(profile)
        runner = build_runner(
            agent_spec=spec,
            target=PythonFunctionAgent(lambda conversation: "Happy to help."),
            sim_provider=ShapedFakeLLM(),
            store=store,
            tactics=["symptom-minimizer"],
            sample_n=1,
        )

        result = runner.run(provider_name="fake", budget=1)

    [node] = result.tree.nodes()
    assert node.tactic == "symptom-minimizer"


def test_build_runner_defaults_to_bundled_plus_profile_personas_with_no_explicit_tactics(
    sample_agent_spec_path,
):
    spec = load_agent_spec(sample_agent_spec_path)
    profile = StressProfile(
        agent_spec_name=spec.name,
        personas=[ProfilePersona(name="custom-persona", scenario="s", user_description="u")],
    )
    with SqliteStore() as store:
        store.save_stress_profile(profile)
        runner = build_runner(
            agent_spec=spec,
            target=PythonFunctionAgent(lambda conversation: "Happy to help."),
            sim_provider=ShapedFakeLLM(),
            store=store,
            sample_n=1,
        )

        result = runner.run(provider_name="fake", budget=1)

    tactics_run = {node.tactic for node in result.tree.nodes()}
    assert tactics_run == set(default_registry().names()) | {"custom-persona"}


def test_build_runner_without_a_profile_only_runs_the_bundled_tactics(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    with SqliteStore() as store:  # no profile ever saved for this spec
        runner = build_runner(
            agent_spec=spec,
            target=PythonFunctionAgent(lambda conversation: "Happy to help."),
            sim_provider=ShapedFakeLLM(),
            store=store,
            sample_n=1,
        )

        result = runner.run(provider_name="fake", budget=1)

    tactics_run = {node.tactic for node in result.tree.nodes()}
    assert tactics_run == set(default_registry().names())


def test_build_runner_without_a_store_ignores_profiles_entirely(sample_agent_spec_path):
    # No store at all (store=None, the parameter's default) must behave
    # exactly as before this wiring existed — no crash, bundled tactics only.
    spec = load_agent_spec(sample_agent_spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(lambda conversation: "Happy to help."),
        sim_provider=ShapedFakeLLM(),
        sample_n=1,
    )

    result = runner.run(provider_name="fake", budget=1)

    tactics_run = {node.tactic for node in result.tree.nodes()}
    assert tactics_run == set(default_registry().names())


# --- Node-level metric wiring in build_runner (C1) -------------------------


def test_build_runner_runs_argument_correctness_by_default(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(tool_calling_verification_agent),
        sim_provider=ShapedFakeLLM(),
        sample_n=1,
    )

    result = runner.run(provider_name="fake", budget=1)

    verdicts = result.tree.all_verdicts()
    # The A4 target always makes a tool call, so every node gets a tool-scoped
    # verdict; task-completion stays off by default (cost-gated until C3).
    assert any(v.scope == "tool" for v in verdicts)
    assert not any(v.scope == "task" for v in verdicts)


def test_build_runner_task_completion_is_opt_in(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(tool_calling_verification_agent),
        sim_provider=ShapedFakeLLM(),
        sample_n=1,
        task_completion=True,
    )

    result = runner.run(provider_name="fake", budget=1)

    assert any(v.scope == "task" for v in result.tree.all_verdicts())


def test_build_runner_can_disable_argument_correctness(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(tool_calling_verification_agent),
        sim_provider=ShapedFakeLLM(),
        sample_n=1,
        argument_correctness=False,
    )

    result = runner.run(provider_name="fake", budget=1)

    assert not any(v.scope == "tool" for v in result.tree.all_verdicts())


# --- Whole-conversation metric wiring in build_runner (C2) ------------------


def test_build_runner_conversation_metrics_are_off_by_default(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(lambda conversation: "Happy to help."),
        sim_provider=ShapedFakeLLM(),
        sample_n=1,
        tactics=["hostile"],
    )

    result = runner.run(provider_name="fake", budget=1)

    assert not any(v.scope == "conversation" for v in result.tree.all_verdicts())


def test_build_runner_conversation_metrics_is_opt_in(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(lambda conversation: "Happy to help."),
        sim_provider=ShapedFakeLLM(),
        sample_n=1,
        tactics=["hostile"],
        conversation_metrics=True,
    )

    result = runner.run(provider_name="fake", budget=1)

    conversation_verdicts = [v for v in result.tree.all_verdicts() if v.scope == "conversation"]
    assert conversation_verdicts
    # One node — the persona's leaf — carries them all (path-keyed, not
    # scattered across every node in the chain).
    assert {v.node_id for v in conversation_verdicts} == {
        node.id for node in result.tree.nodes() if not result.tree.children(node.id)
    }


# --- Usage metering (A5) ---------------------------------------------------


def test_offline_run_populates_run_usage_token_counts_at_zero_cost(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    llm = FakeLLMProvider()
    sim_llm = ShapedFakeLLM()
    runner = build_runner(
        agent_spec=spec,
        target=SampleAgent(spec, llm),
        sim_provider=sim_llm,
        llm=llm,
        sample_n=1,  # no consistency scorer needed to exercise the meters
    )

    result = runner.run(provider_name="fake", budget=2)

    assert result.run.usage.primary.total_tokens > 0
    assert result.run.usage.primary.cost_usd == 0.0
    assert result.run.usage.primary.pricing_unavailable is False
    assert result.run.usage.adversary.total_tokens > 0
    assert result.run.usage.adversary.cost_usd == 0.0


def test_run_without_an_llm_kwarg_leaves_primary_usage_at_zero(sample_agent_spec_path):
    # A target that isn't LLM-backed at all (a scripted Python callable, an
    # HTTP endpoint, ...) has no meaningful "primary" provider to meter.
    spec = load_agent_spec(sample_agent_spec_path)
    runner = build_runner(
        agent_spec=spec,
        target=PythonFunctionAgent(lambda conversation: "a scripted reply"),
        sim_provider=ShapedFakeLLM(),
        sample_n=1,
    )

    result = runner.run(provider_name="fake", budget=1)

    assert result.run.usage.primary.total_tokens == 0
    assert result.run.usage.adversary.total_tokens > 0


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
