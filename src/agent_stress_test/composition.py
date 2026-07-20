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
import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from agent_stress_test.models import (
    AgentSpec,
    Cluster,
    Node,
    ProfilePersona,
    Rule,
    Run,
    StressProfile,
    Verdict,
)
from agent_stress_test.orchestration.cross_run import (
    RulePassRate,
    RunDiff,
    TrendPoint,
    diff_against_previous,
    previous_completed_run,
    reliability_trend,
    rule_pass_rate_history,
)
from agent_stress_test.orchestration.runner import RunResult
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import LLMProvider, Store, TargetAgent
from agent_stress_test.providers.embedder import HashingEmbedder
from agent_stress_test.providers.litellm_provider import LiteLLMProvider
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.reasoning.clusterer import FailureClusterer
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.targets.http_agent import HttpAgent
from agent_stress_test.targets.provider_agent import ProviderAgent
from agent_stress_test.targets.python_fn import PythonFunctionAgent
from agent_stress_test.targets.sample_agent import SampleAgent
from agent_stress_test.targets.sample_agent_advanced import AdvancedSampleAgent
from agent_stress_test.targets.subprocess_agent import SubprocessAgent
from agent_stress_test.targets.tool_backends import build_northwind_tool_backend

# The simulator only has to write one plausible adversarial customer line, not
# solve the task under test — a cheap, fast model does that job as well as the
# target-tier one, at a fraction of the cost. Used only when the run's main
# provider is a real (non-fake) model and no explicit override was given.
DEFAULT_SIM_MODEL = "anthropic/claude-haiku-4-5-20251001"


def build_provider(name: str) -> LLMProvider:
    if name == "fake":
        # ShapedFakeLLM is a strict superset of the plain FakeLLMProvider
        # (identical "fake-reply: ..." behavior whenever no DeepEval schema
        # was requested) — "fake" means the fully-capable offline stand-in
        # everywhere, since the adversarial simulator now always drives
        # sim_provider through DeepEval's schema-validated turn generation
        # (see orchestration/deepeval_search.py), even offline.
        return ShapedFakeLLM()
    return LiteLLMProvider(model=name)


def resolve_sim_provider_name(args: argparse.Namespace) -> str:
    """The model name to drive the simulator: explicit override, else a cheap
    default — unless the main provider is "fake", which stays fake so offline
    runs never silently reach out to a real API."""
    if args.sim_provider is not None:
        return args.sim_provider
    if args.provider == "fake":
        return "fake"
    return DEFAULT_SIM_MODEL


def build_target(args: argparse.Namespace, agent_spec: AgentSpec, llm: LLMProvider) -> TargetAgent:
    if args.target_url:
        return HttpAgent(args.target_url)
    return _build_target_from_spec(agent_spec, llm)


def _build_target_from_spec(agent_spec: AgentSpec, llm: LLMProvider) -> TargetAgent:
    """Build whatever ``TargetAgent`` a spec's ``target:`` block declares —
    the one place that reads it generically, so a new kind is a new branch
    here, not a new special-case threaded through cli.py/server.py. No
    ``target`` block (the common case) falls back to the bundled
    SampleAgent, driven by the run's own provider, exactly as before this
    field existed.
    """
    target = agent_spec.target
    if target is None:
        return SampleAgent(agent_spec, llm)
    if target.kind == "http":
        return HttpAgent(target.url, timeout=target.timeout, headers=target.headers)
    if target.kind == "python":
        return _load_python_target(target.import_path)
    if target.kind == "subprocess":
        return SubprocessAgent(target.command, timeout=target.timeout, cwd=target.cwd)
    if target.kind == "provider":
        return ProviderAgent(LiteLLMProvider(model=target.model), agent_spec)
    if target.kind == "sample_advanced":
        return AdvancedSampleAgent(agent_spec, llm, build_northwind_tool_backend())
    raise ValueError(f"Unknown target kind: {target.kind!r}")  # pragma: no cover - exhaustive union


def _load_python_target(import_path: str) -> TargetAgent:
    """Import ``"module.path:attribute"`` and wrap the callable it names as a
    ``PythonFunctionAgent`` — the shape any bring-your-own Python function
    target takes (``Callable[[list[Message]], str | AgentResponse]``)."""
    module_name, _, attr_name = import_path.partition(":")
    if not attr_name:
        raise ValueError(
            f"Invalid python target import_path '{import_path}' — expected 'module:attribute'."
        )
    module = importlib.import_module(module_name)
    try:
        fn = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"'{import_path}' has no attribute '{attr_name}'.") from exc
    if not callable(fn):
        raise ValueError(f"'{import_path}' is not callable.")
    return PythonFunctionAgent(fn)


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


def load_bundle(
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


@dataclass(frozen=True)
class CrossRunBundle:
    """Everything Phase RE1's cross-run panels need for one run — see
    ``orchestration/cross_run.py`` for how each piece is actually computed."""

    trend: list[TrendPoint]
    diff: RunDiff
    rule_pass_rates: list[RulePassRate]


def load_cross_run_bundle(
    store: Store, run: Run, current_clusters: list[Cluster], current_verdicts: list[Verdict]
) -> CrossRunBundle:
    """Fetch this agent's run history from ``store`` and fold it into a
    ``CrossRunBundle`` — the one place that decides what "historical" means
    (every other completed run for this ``agent_spec.name``, see
    ``Store.list_runs_for_agent``) so the dashboard route stays a plain
    translation of this into template context.
    """
    agent_runs = store.list_runs_for_agent(run.agent_spec.name)
    trend = reliability_trend(agent_runs)

    previous_run = previous_completed_run(run, agent_runs)
    previous_clusters = store.get_clusters(previous_run.id) if previous_run else []
    diff = diff_against_previous(run, current_clusters, previous_run, previous_clusters)

    historical_runs = [
        other for other in agent_runs if other.id != run.id and other.status == "completed"
    ]
    historical_verdicts = [
        verdict for other in historical_runs for verdict in store.get_verdicts(other.id)
    ]
    rule_pass_rates = rule_pass_rate_history(current_verdicts, historical_verdicts)

    return CrossRunBundle(trend=trend, diff=diff, rule_pass_rates=rule_pass_rates)


def resolve_tactics(tactics_arg: str | None, *, extra_valid: Iterable[str] = ()) -> list[str]:
    """A validated tactic subset from a comma-separated arg (default: all).

    ``extra_valid`` widens what counts as a valid name beyond the bundled
    tactic registry — the caller passes in an agent's own approved
    ``StressProfile`` persona names (if any) here, so ``build_runner()``
    (which merges those personas in automatically — see ``runner.py``'s
    ``_profile_extra_personas``) never rejects a name it would actually be
    able to run.
    """
    bundled = default_registry().names()
    extra = [name for name in extra_valid if name not in bundled]
    available = [*bundled, *extra]
    if not tactics_arg:
        return available
    chosen = [name.strip() for name in tactics_arg.split(",") if name.strip()]
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


def resolve_cluster_remediation_target(
    tree: ConversationTree, clusters: list[Cluster], cluster_id: str, agent_spec: AgentSpec
) -> tuple[Node, Rule, Verdict]:
    """The (node, rule, failing verdict) a "suggest a fix" request needs for
    one cluster: the cluster must name a representative node, that node must
    actually carry a failing verdict, and the verdict's rule must still exist
    on the current spec. Raises ``ValueError`` (the same contract every other
    lookup helper here uses — see ``_find_agent_spec_path_by_name`` and
    friends in ``report/dashboard/server.py``) describing exactly which
    precondition failed, so the one caller (the dashboard's "suggest a fix"
    route) only translates that into an HTTP 404, never re-derives the chain
    itself.
    """
    cluster = next((c for c in clusters if c.id == cluster_id), None)
    if cluster is None or cluster.representative_node_id is None:
        raise ValueError(f"Cluster '{cluster_id}' has no representative node.")
    node = tree.get(cluster.representative_node_id)
    failing = [v for v in tree.verdicts(cluster.representative_node_id) if not v.passed]
    if not failing:
        raise ValueError("Representative node has no failing verdict.")
    verdict = failing[0]
    rule = next((r for r in agent_spec.rules if r.id == verdict.rule_id), None)
    if rule is None:
        raise ValueError(f"Rule '{verdict.rule_id}' not found.")
    return node, rule, verdict


def apply_profile_edits(
    existing: StressProfile,
    *,
    names: list[str],
    scenarios: list[str],
    user_descriptions: list[str],
    rule_ids: list[str],
    rule_texts: list[str],
    rule_severities: list[str],
) -> StressProfile:
    """Fold a submitted profile-editor form into an updated StressProfile.

    What counts as a valid edited row is a decision, not a translation, so it
    lives here rather than in the dashboard route: a persona row needs all
    three fields non-blank; a rule row needs non-blank text (a blank
    ``rule_id`` mints a fresh one, for a newly-added row the form has no id
    for yet).
    """
    return existing.model_copy(
        update={
            "personas": [
                ProfilePersona(name=n, scenario=s, user_description=u)
                for n, s, u in zip(names, scenarios, user_descriptions)
                if n.strip() and s.strip() and u.strip()
            ],
            "candidate_rules": [
                Rule(id=rid or str(uuid4()), text=t, severity=sev)
                for rid, t, sev in zip(rule_ids, rule_texts, rule_severities)
                if t.strip()
            ],
        }
    )


def remove_candidate_rule(profile: StressProfile, rule_id: str) -> StressProfile:
    """Drop one candidate rule from a profile, e.g. once it's been written
    into the agent spec's own ``rules:`` list (see
    ``config_writer.apply_candidate_rule``) — it's no longer a *candidate*
    at that point, it's a real rule, so it shouldn't keep showing up here
    asking for review."""
    return profile.model_copy(
        update={
            "candidate_rules": [r for r in profile.candidate_rules if r.id != rule_id],
        }
    )


_INTERRUPTED_ERROR = "Interrupted — the process stopped before this run finished."


def reconcile_interrupted_runs(store: Store) -> int:
    """Close out any run still ``pending``/``running`` at process startup.

    Both composition roots execute a run on a thread that can be killed
    outright -- the dashboard's is a daemon thread (see
    ``report/dashboard/server.py``'s ``post_run``), and the CLI's own
    process can just as easily be Ctrl+C'd (``KeyboardInterrupt`` isn't an
    ``Exception``, so it skips right past the existing "mark this run
    failed" handling around the run itself). Either way, nothing is left
    running to ever flip that row out of "running". But a run can only
    legitimately BE "running" while the process that started it is still
    alive: neither composition root keeps any in-memory record of a run
    that could survive its own restart, so the moment a *new* process
    starts up, any row still claiming to be "running" can only be a
    leftover from one that no longer exists -- safe to close out here,
    every time, before that process does anything else with the store.
    Returns how many rows were fixed, for a caller that wants to log it.
    """
    fixed = 0
    for run in store.list_runs(limit=10_000):
        if run.status in ("pending", "running"):
            store.save_run(
                run.model_copy(
                    update={
                        "status": "failed",
                        "error": _INTERRUPTED_ERROR,
                        "completed_at": datetime.now(timezone.utc),
                    }
                )
            )
            fixed += 1
    return fixed
