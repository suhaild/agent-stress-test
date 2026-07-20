"""The dashboard's live-run registry and SSE event scheduler."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Iterator

from fastapi.templating import Jinja2Templates

from agent_stress_test.composition import CrossRunBundle, load_bundle, load_cross_run_bundle
from agent_stress_test.models import Run, Verdict
from agent_stress_test.orchestration.reliability import near_miss_ranking, score_run
from agent_stress_test.orchestration.rule_coverage import rule_coverage
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.orchestration.tree_viz import build_tree_viz
from agent_stress_test.report.shared import (
    conversation_verdicts_by_leaf,
    executive_summary_context,
    ranked_clusters,
    trend_chart_points,
)
from agent_stress_test.store.sqlite_store import SqliteStore

# Registered before the background thread is spawned, so the SSE endpoint
# never has to guard against "not registered yet". The run's own thread
# mutates each tree in place while this module reads it from the request
# threadpool; ConversationTree's own lock (tree.py) makes that race safe.
live_trees: dict[str, ConversationTree] = {}
live_trees_lock = threading.Lock()

# Kept per run_id, not per SSE connection: the failure feed only appends, so
# a dropped/reconnected EventSource with a fresh `seen` set would replay the
# whole backlog as duplicate cards. Popped on terminal in stream_run_events.
_seen_failures_lock = threading.Lock()
_seen_failures_by_run: dict[str, set[str]] = {}


def _seen_failures_for(run_id: str) -> set[str]:
    with _seen_failures_lock:
        return _seen_failures_by_run.setdefault(run_id, set())


def locked_cluster_ids(store: SqliteStore, agent_spec_name: str) -> set[str]:
    """Cluster ids already promoted into the regression corpus."""
    return {case.source_cluster_id for case in store.get_regression_cases(agent_spec_name)}


def _sse(event: str, html: str) -> str:
    data = "\n".join(f"data: {line}" for line in html.splitlines() or [""])
    return f"event: {event}\n{data}\n\n"


@dataclass
class _EventTick:
    """One poll's state, built once and shared by every panel so they all see
    a consistent snapshot instead of a tree that grew mid-iteration."""

    tree: ConversationTree | None
    node_count: int
    run: Run | None
    status: str
    is_terminal: bool
    # Populated only when is_terminal.
    final_tree: ConversationTree | None = None
    final_verdicts: list[Verdict] = field(default_factory=list)
    ranked_clusters: list[dict] = field(default_factory=list)
    run_provider: str = ""
    locked_cluster_ids: set[str] = field(default_factory=set)
    final_run: Run | None = None
    cross_run: CrossRunBundle | None = None


@dataclass
class _LivePanel:
    """One descriptor per pushed event: template, firing condition
    (``cadence``), and its render context (``context_builder``)."""

    event: str
    template: str
    cadence: Callable[[_EventTick], bool]
    context_builder: Callable[[_EventTick], dict[str, Any] | list[dict[str, Any]]]
    # Failure panel only: a beforeend swap never clears the "No failures yet"
    # placeholder on its own, so this prefixes an out-of-band delete of it.
    decorate: Callable[[str], str] | None = None


def _build_event_tick(run_id: str, db_path: str) -> _EventTick:
    tree = live_trees.get(run_id)
    node_count = len(tree.nodes()) if tree is not None else 0
    with SqliteStore(db_path) as store:
        run = store.get_run(run_id)
    status = run.status if run is not None else "pending"
    tick = _EventTick(
        tree=tree,
        node_count=node_count,
        run=run,
        status=status,
        is_terminal=status in ("completed", "failed"),
    )
    if tick.is_terminal:
        with SqliteStore(db_path) as store:
            final_run, final_tree, final_verdicts, final_clusters = load_bundle(store, run_id)
            locked = locked_cluster_ids(store, final_run.agent_spec.name)
            cross_run = load_cross_run_bundle(store, final_run, final_clusters, final_verdicts)
        tick.final_tree = final_tree
        tick.final_verdicts = final_verdicts
        tick.ranked_clusters = ranked_clusters(final_clusters, final_verdicts)
        tick.run_provider = final_run.provider
        tick.locked_cluster_ids = locked
        tick.final_run = final_run
        tick.cross_run = cross_run
    return tick


def _make_live_panels(run_id: str) -> list[_LivePanel]:
    """Build this run's panel registry, closing over per-connection trackers
    (``last_node_count``/``last_status``, safe to replay on reconnect since
    those panels swap via innerHTML) and the cross-reconnect ``seen`` set."""
    seen = _seen_failures_for(run_id)
    last_node_count = 0
    last_status: str | None = None

    def reliability_live_cadence(tick: _EventTick) -> bool:
        nonlocal last_node_count
        # Re-score on every new node so the gauge tracks the run instead of
        # freezing at its first-paint snapshot until the terminal push.
        if tick.tree is None or tick.node_count == last_node_count:
            return False
        last_node_count = tick.node_count
        return True

    def reliability_live_context(tick: _EventTick) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "reliability": score_run(tick.tree.nodes(), tick.tree.all_verdicts()),
        }

    def failure_cadence(tick: _EventTick) -> bool:
        return tick.tree is not None and any(v.id not in seen for v in tick.tree.failures())

    def failure_contexts(tick: _EventTick) -> list[dict[str, Any]]:
        contexts = []
        for verdict in tick.tree.failures():
            if verdict.id in seen:
                continue
            seen.add(verdict.id)
            contexts.append({"verdict": verdict, "node": tick.tree.get(verdict.node_id)})
        return contexts

    def status_cadence(tick: _EventTick) -> bool:
        nonlocal last_status
        if tick.status == last_status:
            return False
        last_status = tick.status
        return True

    def status_context(tick: _EventTick) -> dict[str, Any]:
        return {"run": tick.run}

    def terminal_cadence(tick: _EventTick) -> bool:
        return tick.is_terminal

    def reliability_final_context(tick: _EventTick) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "reliability": score_run(tick.final_tree.nodes(), tick.final_verdicts),
        }

    def clusters_context(tick: _EventTick) -> dict[str, Any]:
        return {
            "ranked_clusters": tick.ranked_clusters,
            "run_id": run_id,
            "run_provider": tick.run_provider,
            "locked_cluster_ids": tick.locked_cluster_ids,
        }

    def transcripts_context(tick: _EventTick) -> dict[str, Any]:
        return {
            "ranked_clusters": tick.ranked_clusters,
            "tree": tick.final_tree,
            "failures": [v for v in tick.final_verdicts if not v.passed],
        }

    def near_misses_context(tick: _EventTick) -> dict[str, Any]:
        return {"near_misses": near_miss_ranking(tick.final_tree.nodes(), tick.final_verdicts)}

    def conversation_verdicts_context(tick: _EventTick) -> dict[str, Any]:
        return {
            "tree": tick.final_tree,
            "conversation_groups": conversation_verdicts_by_leaf(tick.final_verdicts),
        }

    def cross_run_context(tick: _EventTick) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "cross_run": tick.cross_run,
            "trend_points": trend_chart_points(tick.cross_run.trend) if tick.cross_run else [],
        }

    def rule_coverage_context(tick: _EventTick) -> dict[str, Any]:
        return {
            "rule_coverage": rule_coverage(tick.final_run.agent_spec.rules, tick.final_verdicts),
        }

    def tree_viz_context(tick: _EventTick) -> dict[str, Any]:
        return {"tree_viz": build_tree_viz(tick.final_tree, tick.final_verdicts)}

    def summary_context(tick: _EventTick) -> dict[str, Any]:
        return executive_summary_context(
            tick.final_tree.nodes(),
            tick.final_verdicts,
            [entry["cluster"] for entry in tick.ranked_clusters],
            score_run(tick.final_tree.nodes(), tick.final_verdicts),
            near_miss_ranking(tick.final_tree.nodes(), tick.final_verdicts),
        )

    return [
        _LivePanel(
            "reliability", "fragments/reliability_gauge.html",
            reliability_live_cadence, reliability_live_context,
        ),
        _LivePanel(
            "failure", "fragments/failure_row.html",
            failure_cadence, failure_contexts,
            decorate=lambda html: '<div id="no-failures" hx-swap-oob="delete"></div>' + html,
        ),
        _LivePanel("status", "fragments/status_badge.html", status_cadence, status_context),
        _LivePanel(
            "reliability", "fragments/reliability_gauge.html",
            terminal_cadence, reliability_final_context,
        ),
        _LivePanel("clusters", "fragments/cluster_table.html", terminal_cadence, clusters_context),
        # A top-level block, not swapped by id like the gauge/cluster table —
        # needs an explicit push or it stays absent from the DOM until reload.
        _LivePanel(
            "transcripts", "fragments/transcripts_section.html",
            terminal_cadence, transcripts_context,
        ),
        _LivePanel(
            "near-misses", "fragments/near_miss_panel.html",
            terminal_cadence, near_misses_context,
        ),
        _LivePanel(
            "conversation-verdicts", "fragments/conversation_verdicts_section.html",
            terminal_cadence, conversation_verdicts_context,
        ),
        _LivePanel(
            "cross-run",
            "fragments/cross_run_section.html",
            terminal_cadence,
            cross_run_context,
        ),
        _LivePanel(
            "summary",
            "fragments/summary_panel.html",
            terminal_cadence,
            summary_context,
        ),
        _LivePanel(
            "rule-coverage",
            "fragments/rule_coverage_section.html",
            terminal_cadence,
            rule_coverage_context,
        ),
        _LivePanel(
            "tree-viz",
            "fragments/tree_viz_section.html",
            terminal_cadence,
            tree_viz_context,
        ),
    ]


def stream_run_events(run_id: str, db_path: str, templates: Jinja2Templates) -> Iterator[str]:
    panels = _make_live_panels(run_id)
    while True:
        tick = _build_event_tick(run_id, db_path)
        for panel in panels:
            if not panel.cadence(tick):
                continue
            contexts = panel.context_builder(tick)
            if isinstance(contexts, dict):
                contexts = [contexts]
            for context in contexts:
                html = templates.get_template(panel.template).render(**context)
                if panel.decorate is not None:
                    html = panel.decorate(html)
                yield _sse(panel.event, html)
        if tick.is_terminal:
            _seen_failures_by_run.pop(run_id, None)
            return
        time.sleep(0.3)
