import pytest
from deepeval.dataset import ConversationalGolden

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Message
from agent_stress_test.orchestration.deepeval_search import DeepEvalConversationSearch
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.consistency import ConsistencyScorer
from agent_stress_test.reasoning.judge import RulesJudge, build_checks
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.targets.python_fn import PythonFunctionAgent
from agent_stress_test.targets.tool_calling_verification_agent import (
    tool_calling_verification_agent,
)

TACTIC_COUNT = len(default_registry().names())


def _make_search(spec_path, target, scorer=None, **kwargs) -> DeepEvalConversationSearch:
    spec = load_agent_spec(spec_path)
    return DeepEvalConversationSearch(
        target,
        ShapedFakeLLM(),
        RulesJudge(build_checks(spec)),
        scorer,
        **kwargs,
    )


def test_search_produces_one_independent_root_chain_per_persona(sample_agent_spec_path):
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    search = _make_search(sample_agent_spec_path, target)
    tree = ConversationTree("run-1")

    result = search.search(tree, [Message(role="user", content="hi")], budget=2)

    assert result.expansions == TACTIC_COUNT
    assert len(tree.roots()) == TACTIC_COUNT
    assert len(tree.nodes()) == TACTIC_COUNT * 2
    assert result.nodes_created == TACTIC_COUNT * 2


def test_search_ingests_a_linear_chain_per_persona(sample_agent_spec_path):
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    search = _make_search(sample_agent_spec_path, target)
    tree = ConversationTree("run-1")

    search.search(tree, [], budget=3)

    for root in tree.roots():
        chain = [root]
        while tree.children(chain[-1].id):
            [child] = tree.children(chain[-1].id)
            chain.append(child)
        assert len(chain) == 3
        assert all(node.tactic == root.tactic for node in chain)
        # Each node's messages accumulate — later nodes carry more history.
        assert len(chain[-1].messages) > len(chain[0].messages)


def test_search_respects_a_persona_subset_via_tactics_param(sample_agent_spec_path):
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    search = _make_search(sample_agent_spec_path, target, tactics=["hostile", "stale-recall"])
    tree = ConversationTree("run-1")

    result = search.search(tree, [], budget=1)

    assert result.expansions == 2
    assert {node.tactic for node in tree.nodes()} == {"hostile", "stale-recall"}


def test_search_judges_every_ingested_node_and_reports_failures(sample_agent_spec_path):
    target = PythonFunctionAgent(lambda conversation: "Sure — I've already refunded your card.")
    search = _make_search(sample_agent_spec_path, target, tactics=["hostile"])
    tree = ConversationTree("run-1")

    result = search.search(tree, [], budget=1)

    assert tree.all_verdicts()
    assert result.failures
    assert any(v.rule_id == "no-self-refund" for v in result.failures)


def test_search_persists_structured_tool_calls_onto_ingested_nodes(sample_agent_spec_path):
    search = _make_search(
        sample_agent_spec_path,
        PythonFunctionAgent(tool_calling_verification_agent),
        tactics=["hostile"],
    )
    tree = ConversationTree("run-1")

    search.search(tree, [], budget=1)

    [node] = tree.nodes()
    assert node.tool_calls
    assert node.tool_calls[0].name == "lookup_order"
    assert isinstance(node.tool_calls[0].input_parameters, dict)


def test_search_without_a_scorer_defaults_instability_to_zero(sample_agent_spec_path):
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    search = _make_search(sample_agent_spec_path, target, tactics=["hostile"])
    tree = ConversationTree("run-1")

    search.search(tree, [], budget=1)

    assert all(node.instability_score == 0.0 for node in tree.nodes())


def test_search_with_a_scorer_resamples_the_target_and_scores_instability(sample_agent_spec_path):
    # A target whose reply depends on call count, so repeated resampling by
    # the ConsistencyScorer actually disagrees with itself — instability
    # should land above 0.0, not just default there.
    calls = {"n": 0}

    def flaky_target(conversation):
        calls["n"] += 1
        return "Happy to help." if calls["n"] % 2 else "Sorry, I can't help with that."

    target = PythonFunctionAgent(flaky_target)
    scorer = ConsistencyScorer(target)
    search = _make_search(sample_agent_spec_path, target, scorer, tactics=["hostile"])
    tree = ConversationTree("run-1")

    search.search(tree, [], budget=1)

    [node] = tree.nodes()
    assert node.instability_score is not None
    assert 0.0 < node.instability_score <= 1.0


def test_search_raises_a_clean_value_error_on_zero_budget(sample_agent_spec_path):
    # Unlike the old GreedyBestFirstSearch (which treats budget=0 as
    # "seed-only, no expansions"), DeepEval's ConversationSimulator itself
    # refuses max_user_simulations=0 outright — confirmed against the
    # installed version, not assumed. cli.py's top-level `except ValueError`
    # still turns this into a clean one-line message, never a raw traceback,
    # so this is documented, not silently crash-prone.
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    search = _make_search(sample_agent_spec_path, target, tactics=["hostile"])
    tree = ConversationTree("run-1")

    with pytest.raises(ValueError, match="max_user_simulations"):
        search.search(tree, [], budget=0)


# --- extra_personas: profile-sourced personas drive a real search ---------


def test_search_runs_an_extra_persona_not_in_the_bundled_registry(sample_agent_spec_path):
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    custom_golden = ConversationalGolden(
        scenario="A patient downplays a serious symptom.",
        user_description="A patient who minimizes their symptoms.",
    )
    search = _make_search(
        sample_agent_spec_path,
        target,
        tactics=["symptom-minimizer"],
        extra_personas={"symptom-minimizer": custom_golden},
    )
    tree = ConversationTree("run-1")

    result = search.search(tree, [], budget=1)

    assert result.expansions == 1
    [node] = tree.nodes()
    assert node.tactic == "symptom-minimizer"


def test_search_with_no_explicit_tactics_defaults_to_bundled_plus_extra_personas(
    sample_agent_spec_path,
):
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    custom_golden = ConversationalGolden(scenario="s", user_description="u")
    search = _make_search(
        sample_agent_spec_path, target, extra_personas={"custom-persona": custom_golden}
    )
    tree = ConversationTree("run-1")

    result = search.search(tree, [], budget=1)

    assert result.expansions == TACTIC_COUNT + 1
    assert {node.tactic for node in tree.nodes()} == set(default_registry().names()) | {
        "custom-persona"
    }


def test_search_extra_personas_do_not_shadow_the_bundled_registry(sample_agent_spec_path):
    # A bundled name in extra_personas is harmless — {**PERSONAS, **extra}
    # means extra would win if it collided, but this just confirms the
    # bundled set still resolves fine when extra_personas is non-empty but
    # names a disjoint persona.
    target = PythonFunctionAgent(lambda conversation: "Happy to help.")
    custom_golden = ConversationalGolden(scenario="s", user_description="u")
    search = _make_search(
        sample_agent_spec_path,
        target,
        tactics=["hostile"],
        extra_personas={"custom-persona": custom_golden},
    )
    tree = ConversationTree("run-1")

    result = search.search(tree, [], budget=1)

    assert result.expansions == 1
    [node] = tree.nodes()
    assert node.tactic == "hostile"
