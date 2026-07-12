"""Shared wiring helpers reused by every composition root.

``cli.py`` and ``report/dashboard/server.py`` both need to turn a bag of
user-supplied parameters (provider name, agent spec, tactics, ...) into the
concrete adapters ``build_runner()`` expects, and both need to reload a
finished run's tree from the ``Store`` for reporting. That logic lives here,
once, so neither composition root duplicates it — a composition root should
only translate its own input shape (argv vs. an HTTP request) into calls on
these functions, never re-implement the decisions they make.
"""

from __future__ import annotations

import argparse

from agent_stress_test.models import AgentSpec, Cluster, Node, Run, Verdict
from agent_stress_test.orchestration.runner import RunResult
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import LLMProvider, Store, TargetAgent
from agent_stress_test.providers.embedder import HashingEmbedder
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.litellm_provider import LiteLLMProvider
from agent_stress_test.reasoning.clusterer import FailureClusterer
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.targets.http_agent import HttpAgent
from agent_stress_test.targets.sample_agent import SampleAgent

# The simulator only has to write one plausible adversarial customer line, not
# solve the task under test — a cheap, fast model does that job as well as the
# target-tier one, at a fraction of the cost. Used only when the run's main
# provider is a real (non-fake) model and no explicit override was given.
_DEFAULT_SIM_MODEL = "anthropic/claude-haiku-4-5-20251001"


def _build_provider(name: str) -> LLMProvider:
    if name == "fake":
        return FakeLLMProvider()
    return LiteLLMProvider(model=name)


def _resolve_sim_provider_name(args: argparse.Namespace) -> str:
    """The model name to drive the simulator: explicit override, else a cheap
    default — unless the main provider is "fake", which stays fake so offline
    runs never silently reach out to a real API."""
    if args.sim_provider is not None:
        return args.sim_provider
    if args.provider == "fake":
        return "fake"
    return _DEFAULT_SIM_MODEL


def _build_target(args: argparse.Namespace, spec: AgentSpec, llm: LLMProvider) -> TargetAgent:
    if args.target_url:
        return HttpAgent(args.target_url)
    return SampleAgent(spec, llm)


def _rebuild_tree(run_id: str, nodes: list[Node], verdicts: list[Verdict]) -> ConversationTree:
    """Rebuild a ConversationTree from flat, order-independent storage rows."""
    tree = ConversationTree(run_id)
    remaining = list(nodes)
    while remaining:
        added_any = False
        still_remaining = []
        for node in remaining:
            if node.parent_id is None or node.parent_id in {n.id for n in tree.nodes()}:
                tree.add(node)
                added_any = True
            else:
                still_remaining.append(node)
        if not added_any:
            raise ValueError("Cannot rebuild tree: orphaned node(s) with unknown parent.")
        remaining = still_remaining

    verdicts_by_node: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        verdicts_by_node.setdefault(verdict.node_id, []).append(verdict)
    for node_id, node_verdicts in verdicts_by_node.items():
        tree.attach_verdicts(node_id, node_verdicts)
    return tree


def _load_bundle(
    store: Store, run_id: str
) -> tuple[Run, ConversationTree, list[Verdict], list[Cluster]]:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"No run found with id '{run_id}'.")
    nodes = store.get_nodes(run_id)
    verdicts = store.get_verdicts(run_id)
    clusters = store.get_clusters(run_id)
    tree = _rebuild_tree(run_id, nodes, verdicts)
    return run, tree, verdicts, clusters


def _resolve_tactics(spec_arg: str | None) -> list[str]:
    """A validated tactic subset from a comma-separated arg (default: all)."""
    available = default_registry().names()
    if not spec_arg:
        return available
    chosen = [name.strip() for name in spec_arg.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown tactic(s): {', '.join(unknown)}. Available: {', '.join(available)}"
        )
    return chosen


def cluster_and_persist(result: RunResult, store: Store) -> list[Cluster]:
    """Cluster a finished run's failures and persist them.

    The one clustering step every composition root must call identically —
    ``Runner.run()`` doesn't do this itself (clustering isn't part of running
    a search, it's a reporting concern), so whichever front end triggered the
    run (CLI or dashboard) calls this the same way, and a given run always
    gets the same cluster labels no matter which one produced it.
    """
    clusterer = FailureClusterer(HashingEmbedder())
    clusters = clusterer.cluster(
        result.tree.nodes(), result.tree.all_verdicts(), run_id=result.run.id
    )
    for cluster in clusters:
        store.save_cluster(cluster)
    return clusters
