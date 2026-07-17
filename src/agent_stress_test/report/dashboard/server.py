"""The web dashboard's FastAPI app — a second composition root, alongside
``cli.py``.

Routes here only translate HTTP <-> the same ``build_runner()``/``Runner``
wiring ``cli.py`` uses; the shared decision logic (which provider, which
tactics, how to reload a finished run) lives in ``composition.py`` so neither
front end duplicates it. This module owns two things nothing else needs: the
in-process registry of in-progress runs (so the live failure feed has
something to read before anything reaches the store) and the HTTP/template
glue itself.
"""

from __future__ import annotations

import difflib
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from uuid import uuid4

import pysbd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from agent_stress_test.composition import (
    _build_provider,
    _build_target,
    _load_bundle,
    _resolve_sim_provider_name,
    _resolve_tactics,
    cluster_and_persist,
)
from agent_stress_test.config import apply_system_prompt, load_agent_spec, load_settings
from agent_stress_test.models import AgentSpec, Cluster, Run, SystemPromptVersion, Verdict
from agent_stress_test.orchestration.regression import (
    RegressionRunner,
    promote_clusters_to_cases,
)
from agent_stress_test.orchestration.reliability import ReliabilityReport, score_run
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.orchestration.search import SEVERITY_WEIGHT
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.reasoning.judge import build_two_tier_judge
from agent_stress_test.reasoning.remediation import RemediationSuggester
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.store.sqlite_store import SqliteStore

_DEFAULT_DB = "runs.sqlite"
# server.py -> dashboard -> report -> agent_stress_test -> src -> repo root
_CONFIG_AGENTS_DIR = Path(__file__).resolve().parents[4] / "config" / "agents"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Registered by the request thread the instant a run starts, before the
# background thread is even spawned, so the SSE endpoint never has to guard
# against "not registered yet". ``GreedyBestFirstSearch.search()`` mutates
# each tree in place from the run's own thread while this module reads it
# from the request-serving threadpool; ``ConversationTree`` guards every
# access with its own lock (see tree.py) so that read/write race is safe.
_live_trees: dict[str, ConversationTree] = {}
_live_trees_lock = threading.Lock()

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def list_agent_specs() -> list[dict[str, str]]:
    """The configured agent specs, as ``[{"id": <filename>, "name": <spec.name>}]``."""
    return [
        {"id": path.name, "name": load_agent_spec(path).name}
        for path in sorted(_CONFIG_AGENTS_DIR.glob("*.yaml"))
    ]


def list_tactics() -> list[dict[str, str]]:
    """The built-in tactic library, as ``[{"id": <name>, "description": ...}]`` —
    read from the same registry the simulator uses, so the dashboard's tactic
    picker can never drift from what actually runs."""
    registry = default_registry()
    return [
        {"id": name, "description": registry.get(name).description}
        for name in registry.names()
    ]


def list_models() -> list[dict[str, str]]:
    """A short, curated list of commonly-used LLM ids for the model pickers.

    Not exhaustive — litellm accepts any provider/model string, so the picker
    is a native ``<datalist>`` suggestion list, not a closed enum. Typing a
    different id (a self-hosted model, a different provider) still works."""
    return [
        {"id": "fake", "label": "Offline Test Double — no API key needed"},
        {"id": "anthropic/claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 — cheap & fast"},
        {"id": "anthropic/claude-sonnet-5", "label": "Claude Sonnet 5 — balanced"},
        {"id": "anthropic/claude-opus-4-8", "label": "Claude Opus 4.8 — most capable"},
    ]


def _resolve_agent_spec_path(agent_spec_id: str) -> Path:
    """Map a client-submitted id back to a real file, refusing anything not in
    the enumerated list — the dashboard never accepts a raw filesystem path
    from the browser."""
    if any(entry["id"] == agent_spec_id for entry in list_agent_specs()):
        return _CONFIG_AGENTS_DIR / agent_spec_id
    raise ValueError(f"Unknown agent spec '{agent_spec_id}'.")


def _find_agent_spec_path_by_name(agent_spec_name: str) -> Path:
    """Resolve a spec's own ``.name`` (what RegressionCase is keyed on, and
    all a run remembers) back to its current, live YAML file — a regression
    replay (or an applied fix) must act on whatever the file says *now*, not
    a frozen copy embedded in an old Run. Scans the same enumerated
    ``config/agents/*.yaml`` list every other agent-spec lookup here uses,
    never a client-supplied path."""
    for entry in list_agent_specs():
        path = _CONFIG_AGENTS_DIR / entry["id"]
        if load_agent_spec(path).name == agent_spec_name:
            return path
    raise ValueError(f"No configured agent spec named '{agent_spec_name}'.")


def _find_agent_spec_by_name(agent_spec_name: str) -> AgentSpec:
    return load_agent_spec(_find_agent_spec_path_by_name(agent_spec_name))


def _locked_cluster_ids(store: SqliteStore, agent_spec_name: str) -> set[str]:
    """Cluster ids already promoted into the regression corpus — so the Lock
    button can show "locked" instead of silently minting duplicate cases."""
    return {case.source_cluster_id for case in store.get_regression_cases(agent_spec_name)}


_SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def _normalize_for_diff(text: str) -> list[str]:
    """Split text into sentences for diffing, ignoring incidental line-wrap
    width. A raw line-based diff is misleading here: the YAML's system_prompt
    is hard-wrapped at a fixed column width, but an LLM's suggested
    replacement rarely reproduces that exact wrap point — so a plain
    ``str.splitlines()`` diff shows the whole paragraph as removed-and-re-added
    even when only one sentence actually changed. Collapsing whitespace first
    (so wrapping can't matter) and segmenting by sentence gives a diff that
    tracks meaning, not incidental formatting.
    """
    collapsed = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SEGMENTER.segment(collapsed) if s.strip()]


def _diff_blocks(old_text: str, new_text: str) -> list[dict]:
    """Groups a unified diff into template-ready blocks for a browser
    audience, not a `git diff` reader: each contiguous run of unchanged
    sentences becomes one ``{"kind": "context", "text": ...}`` row (labeled
    "unchanged" in the template — shown only for orientation, so it's clear
    it's surrounding prompt text, not part of the edit), and each run of
    removed/added sentences becomes one ``{"kind": "change", "previous":
    [...], "suggested": [...]}`` pair, labeled "previous"/"suggested"
    explicitly rather than relying on red/green coloring alone to say which
    side is which. The raw ``---``/``+++``/``@@ -3,4 +3,4 @@`` unified-diff
    header lines (meaningful line positions to a `git diff` reader, noise to
    everyone else) are dropped entirely.
    """
    lines = difflib.unified_diff(_normalize_for_diff(old_text), _normalize_for_diff(new_text), lineterm="")
    blocks: list[dict] = []
    previous: list[str] = []
    suggested: list[str] = []
    context: list[str] = []

    def flush_change() -> None:
        if previous or suggested:
            blocks.append({"kind": "change", "previous": list(previous), "suggested": list(suggested)})
            previous.clear()
            suggested.clear()

    def flush_context() -> None:
        if context:
            blocks.append({"kind": "context", "text": " ".join(context)})
            context.clear()

    for line in lines:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            flush_context()
            previous.append(line[1:])
        elif line.startswith("+"):
            flush_context()
            suggested.append(line[1:])
        else:
            flush_change()
            context.append(line[1:])
    flush_change()
    flush_context()
    return blocks


def _record_prompt_version(
    store: SqliteStore, agent_spec_name: str, system_prompt: str
) -> None:
    """Content-addressed history: records ``system_prompt`` as a version only
    if an identical one isn't already on file for this agent. Without this,
    restoring an old version (or re-applying one just undone) would log a
    fresh duplicate row every time — the history would grow on every click
    instead of being a genuine list of distinct versions. Restoring content
    that's already recorded just moves which row counts as "current"; it
    never mints a new one.
    """
    already_recorded = any(
        v.system_prompt == system_prompt for v in store.get_system_prompt_versions(agent_spec_name)
    )
    if not already_recorded:
        store.save_system_prompt_version(
            SystemPromptVersion(agent_spec_name=agent_spec_name, system_prompt=system_prompt)
        )


def _prompt_version_history(
    current_system_prompt: str, prompt_versions: list[SystemPromptVersion]
) -> list[dict]:
    """One row per distinct version ever recorded for this agent
    (most-recent-created first), each diffed against whichever version came
    immediately before it in time — the oldest entry is the prompt as it
    stood before any fix was ever applied, shown as-is with no diff. Because
    the version list is content-addressed (see ``_record_prompt_version``),
    every row — including ones already superseded and then brought back —
    is restorable via the same action, and "current" is whichever row's text
    matches what's live right now, not necessarily the newest row.
    """
    rows = []
    total = len(prompt_versions)
    for i, version in enumerate(prompt_versions):
        predecessor = prompt_versions[i + 1] if i + 1 < total else None
        rows.append(
            {
                "version": version,
                "ordinal": total - i,
                "is_current": version.system_prompt == current_system_prompt,
                "diff_blocks": (
                    _diff_blocks(predecessor.system_prompt, version.system_prompt)
                    if predecessor is not None
                    else None
                ),
            }
        )
    return rows


def _worst_severity(cluster: Cluster, verdicts: list[Verdict]) -> str:
    """Mirrors ``report/terminal.py``'s ``_worst_severity`` — same ranking, so
    the CLI report and the dashboard never disagree on cluster ordering."""
    weights = [
        SEVERITY_WEIGHT[v.severity]
        for v in verdicts
        if not v.passed and v.node_id in cluster.member_node_ids
    ]
    if not weights:
        return "minor"
    best = max(weights)
    return next(sev for sev, weight in SEVERITY_WEIGHT.items() if weight == best)


def _ranked_clusters(clusters: list[Cluster], verdicts: list[Verdict]) -> list[dict]:
    """Clusters worst-severity-first then largest-first, each paired with its
    severity — pre-computed here so the template just iterates, no logic."""
    ranked = sorted(
        clusters,
        key=lambda c: (SEVERITY_WEIGHT[_worst_severity(c, verdicts)], len(c.member_node_ids)),
        reverse=True,
    )
    return [{"cluster": c, "severity": _worst_severity(c, verdicts)} for c in ranked]


def _execute_run(
    *,
    db_path: str,
    run_id: str,
    tree: ConversationTree,
    agent_spec_path: Path,
    provider: str,
    sim_provider: str | None,
    target_url: str | None,
    budget: int,
    sample_n: int,
    tactics_arg: str | None,
) -> None:
    """The background-thread target: mirrors ``cli.py``'s ``_cmd_run`` wiring,
    but reports failure via the store instead of letting an exception crash a
    daemon thread silently."""
    try:
        load_settings()
        spec = load_agent_spec(agent_spec_path)
        args = SimpleNamespace(
            provider=provider,
            sim_provider=sim_provider,
            target_url=target_url,
        )
        llm = _build_provider(provider)
        target = _build_target(args, spec, llm)
        tactics = _resolve_tactics(tactics_arg)
        sim_provider_name = _resolve_sim_provider_name(args)
        sim_llm = llm if sim_provider_name == provider else _build_provider(sim_provider_name)

        with SqliteStore(db_path) as store:
            runner = build_runner(
                agent_spec=spec,
                target=target,
                sim_provider=sim_llm,
                store=store,
                tactics=tactics,
                sample_n=sample_n,
            )
            result = runner.run(
                provider_name=provider, budget=budget, run_id=run_id, tree=tree
            )
            cluster_and_persist(result, store)
    except Exception as exc:
        traceback.print_exc()
        with SqliteStore(db_path) as store:
            for node in tree.nodes():
                store.save_node(node)
            for verdict in tree.all_verdicts():
                store.save_verdict(verdict)
            existing = store.get_run(run_id)
            store.save_run(
                Run(
                    id=run_id,
                    agent_spec=existing.agent_spec if existing else load_agent_spec(agent_spec_path),
                    provider=provider,
                    budget=budget,
                    status="failed",
                    started_at=existing.started_at if existing else None,
                    completed_at=datetime.now(timezone.utc),
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
            )
    finally:
        with _live_trees_lock:
            _live_trees.pop(run_id, None)


def _sse(event: str, html: str) -> str:
    data = "\n".join(f"data: {line}" for line in html.splitlines() or [""])
    return f"event: {event}\n{data}\n\n"


@dataclass
class _EventTick:
    """One poll of the live loop's state, built once per iteration and handed
    to every panel's cadence/context callables so they all see a consistent
    snapshot (the tree can otherwise keep growing mid-iteration)."""

    tree: ConversationTree | None
    node_count: int
    run: Run | None
    status: str
    is_terminal: bool
    # Populated only when is_terminal — the persisted, final version of the
    # same data the terminal panels (reliability/clusters/transcripts) render.
    final_tree: ConversationTree | None = None
    final_verdicts: list[Verdict] = field(default_factory=list)
    ranked_clusters: list[dict] = field(default_factory=list)
    run_provider: str = ""
    locked_cluster_ids: set[str] = field(default_factory=set)


@dataclass
class _LivePanel:
    """One entry in the live loop's registry: what event to push, which
    template renders it, when it fires (``cadence``), and what to render it
    with (``context_builder``). Adding a new live panel later is exactly
    "append one descriptor here" — the loop below has no per-panel branches,
    so a panel left out of this list can never accidentally end up in the
    live loop."""

    event: str
    template: str
    cadence: Callable[[_EventTick], bool]
    context_builder: Callable[[_EventTick], dict[str, Any] | list[dict[str, Any]]]
    # Only the failure panel needs this: each new failure_row.html render is
    # prefixed with an out-of-band delete of the feed's "No failures yet."
    # placeholder (a beforeend swap only ever appends, so it never clears
    # that placeholder on its own).
    decorate: Callable[[str], str] | None = None


def _build_event_tick(run_id: str, db_path: str) -> _EventTick:
    tree = _live_trees.get(run_id)
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
            final_run, final_tree, final_verdicts, final_clusters = _load_bundle(store, run_id)
            locked = _locked_cluster_ids(store, final_run.agent_spec.name)
        tick.final_tree = final_tree
        tick.final_verdicts = final_verdicts
        tick.ranked_clusters = _ranked_clusters(final_clusters, final_verdicts)
        tick.run_provider = final_run.provider
        tick.locked_cluster_ids = locked
    return tick


def _make_live_panels(run_id: str) -> list[_LivePanel]:
    """Build this run's panel registry. Bound to closures over a few
    per-connection trackers (``seen``/``last_node_count``/``last_status``) so
    each SSE connection gets its own bookkeeping."""
    seen: set[str] = set()
    last_node_count = 0
    last_status: str | None = None

    def reliability_live_cadence(tick: _EventTick) -> bool:
        nonlocal last_node_count
        # The gauge's initial render is a snapshot from whenever the page was
        # loaded; without this, it never moves again until the one terminal
        # push below, so a run can sit there reading 100% (or whatever the
        # very first node scored) for its entire duration. Re-score and push
        # every time the live tree gains a node, so it tracks the run instead
        # of a first-paint snapshot.
        if tick.tree is None or tick.node_count == last_node_count:
            return False
        last_node_count = tick.node_count
        return True

    def reliability_live_context(tick: _EventTick) -> dict[str, Any]:
        return {"reliability": score_run(tick.tree.nodes(), tick.tree.all_verdicts())}

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
        return {"reliability": score_run(tick.final_tree.nodes(), tick.final_verdicts)}

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
        # Terminal-only: fire once, together, the instant the run finishes.
        _LivePanel(
            "reliability", "fragments/reliability_gauge.html",
            terminal_cadence, reliability_final_context,
        ),
        _LivePanel("clusters", "fragments/cluster_table.html", terminal_cadence, clusters_context),
        # Representative Transcripts is a top-level block, not swapped by id
        # like the gauge/cluster table above — clustering (and thus which
        # nodes are "representative") only exists once the run is done, so
        # without this push the section stays absent from the DOM until a
        # full page reload re-renders it from scratch.
        _LivePanel(
            "transcripts", "fragments/transcripts_section.html",
            terminal_cadence, transcripts_context,
        ),
    ]


def _run_events(run_id: str, db_path: str) -> Iterator[str]:
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
            return
        time.sleep(0.3)
        time.sleep(0.3)


def create_app(db_path: str = _DEFAULT_DB) -> FastAPI:
    """Build the dashboard app. ``db_path`` is fixed here, at process startup —
    never accepted from a client request (see module docstring on trust)."""
    app = FastAPI(title="Agent Stress-Test Dashboard")
    app.state.db_path = db_path

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        with SqliteStore(app.state.db_path) as store:
            recent_runs = store.list_runs()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "agent_specs": list_agent_specs(),
                "recent_runs": recent_runs,
                "tactics": list_tactics(),
                "models": list_models(),
            },
            # The recent-runs list is a point-in-time snapshot; without this,
            # navigating back to "/" (browser back/forward) can restore a
            # cached copy from before a run finished instead of refetching,
            # so newly completed runs silently don't show up until a manual
            # reload. no-store also makes the page ineligible for the
            # back/forward cache in the browsers that honor it (e.g. Chrome).
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/agent-specs")
    def get_agent_specs() -> list[dict[str, str]]:
        return list_agent_specs()

    @app.post("/runs")
    def post_run(
        agent_spec_id: str = Form(...),
        provider: str = Form("fake"),
        sim_provider: str = Form(""),
        target_url: str = Form(""),
        budget: int = Form(6),
        sample_n: int = Form(1),
        tactics: str = Form(""),
    ) -> JSONResponse:
        try:
            agent_spec_path = _resolve_agent_spec_path(agent_spec_id)
            spec = load_agent_spec(agent_spec_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = str(uuid4())
        with SqliteStore(app.state.db_path) as store:
            store.save_run(
                Run(
                    id=run_id,
                    agent_spec=spec,
                    provider=provider,
                    budget=budget,
                    status="running",
                    started_at=datetime.now(timezone.utc),
                )
            )

        tree = ConversationTree(run_id)
        with _live_trees_lock:
            _live_trees[run_id] = tree

        thread = threading.Thread(
            target=_execute_run,
            kwargs=dict(
                db_path=app.state.db_path,
                run_id=run_id,
                tree=tree,
                agent_spec_path=agent_spec_path,
                provider=provider,
                sim_provider=sim_provider or None,
                target_url=target_url or None,
                budget=budget,
                sample_n=sample_n,
                tactics_arg=tactics or None,
            ),
            daemon=True,
        )
        thread.start()

        return JSONResponse(
            {"run_id": run_id, "status": "running"},
            status_code=202,
            headers={"HX-Redirect": f"/runs/{run_id}"},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def get_run(request: Request, run_id: str) -> HTMLResponse:
        with SqliteStore(app.state.db_path) as store:
            run = store.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"No run found with id '{run_id}'.")

            live_tree = _live_trees.get(run_id)
            if run.status in ("pending", "running") and live_tree is not None:
                tree = live_tree
                verdicts = tree.all_verdicts()
                clusters = []
                reliability: ReliabilityReport | None = (
                    score_run(tree.nodes(), verdicts) if tree.nodes() else None
                )
            else:
                run, tree, verdicts, clusters = _load_bundle(store, run_id)
                reliability = score_run(tree.nodes(), verdicts)

            locked = _locked_cluster_ids(store, run.agent_spec.name)

        failures: list[Verdict] = [v for v in verdicts if not v.passed]
        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "run": run,
                "run_id": run_id,
                "run_provider": run.provider,
                "reliability": reliability,
                "ranked_clusters": _ranked_clusters(clusters, verdicts),
                "locked_cluster_ids": locked,
                "failures": failures,
                "tree": tree,
            },
        )

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        with SqliteStore(app.state.db_path) as store:
            if store.get_run(run_id) is None:
                raise HTTPException(status_code=404, detail=f"No run found with id '{run_id}'.")
        return StreamingResponse(
            _run_events(run_id, app.state.db_path), media_type="text/event-stream"
        )

    @app.post("/runs/{run_id}/clusters/{cluster_id}/lock", response_class=HTMLResponse)
    def post_lock_cluster(request: Request, run_id: str, cluster_id: str) -> HTMLResponse:
        with SqliteStore(app.state.db_path) as store:
            run, tree, verdicts, clusters = _load_bundle(store, run_id)
            for case in promote_clusters_to_cases(run, tree, clusters, cluster_ids={cluster_id}):
                store.save_regression_case(case)
            locked = _locked_cluster_ids(store, run.agent_spec.name)

        return templates.TemplateResponse(
            request,
            "fragments/cluster_table.html",
            {
                "ranked_clusters": _ranked_clusters(clusters, verdicts),
                "run_id": run_id,
                "run_provider": run.provider,
                "locked_cluster_ids": locked,
            },
        )

    @app.post("/runs/{run_id}/clusters/{cluster_id}/suggest-fix", response_class=HTMLResponse)
    def post_suggest_fix(
        request: Request, run_id: str, cluster_id: str, provider: str = Form("fake")
    ) -> HTMLResponse:
        load_settings()
        with SqliteStore(app.state.db_path) as store:
            run, tree, _verdicts, clusters = _load_bundle(store, run_id)
        cluster = next((c for c in clusters if c.id == cluster_id), None)
        if cluster is None or cluster.representative_node_id is None:
            raise HTTPException(
                status_code=404, detail=f"Cluster '{cluster_id}' has no representative node."
            )
        node = tree.get(cluster.representative_node_id)
        failing = [v for v in tree.verdicts(cluster.representative_node_id) if not v.passed]
        if not failing:
            raise HTTPException(
                status_code=404, detail="Representative node has no failing verdict."
            )
        verdict = failing[0]
        rule = next((r for r in run.agent_spec.rules if r.id == verdict.rule_id), None)
        if rule is None:
            raise HTTPException(status_code=404, detail=f"Rule '{verdict.rule_id}' not found.")

        suggestion = RemediationSuggester(_build_provider(provider)).suggest(
            run.agent_spec, rule, node.target_reply, verdict.reason
        )
        return templates.TemplateResponse(
            request,
            "fragments/suggestion_panel.html",
            {
                "rule": rule,
                "suggestion": suggestion,
                "agent_spec_name": run.agent_spec.name,
                "old_system_prompt": run.agent_spec.system_prompt,
                "diff_blocks": _diff_blocks(run.agent_spec.system_prompt, suggestion.suggested_system_prompt),
            },
        )

    @app.post("/agent-specs/{agent_spec_name}/system-prompt/apply", response_class=HTMLResponse)
    def post_apply_system_prompt(
        request: Request, agent_spec_name: str, suggested_system_prompt: str = Form(...)
    ) -> HTMLResponse:
        try:
            path = _find_agent_spec_path_by_name(agent_spec_name)
            previous_prompt = load_agent_spec(path).system_prompt
            new_spec = apply_system_prompt(path, suggested_system_prompt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Record both ends of this write as distinct versions (content-
        # addressed — see `_record_prompt_version`). The first-ever apply for
        # an agent captures its original prompt as the baseline "Revision 1"
        # in the same stroke; every later apply is a no-op on whichever side
        # is already on file. A hosted deployment may give the person
        # clicking this button no shell/git access to the server, so "git
        # checkout" isn't a real safety net there — this is what actually
        # lets them restore any past version, regardless of how it's
        # deployed, including reapplying one they just undid.
        with SqliteStore(app.state.db_path) as store:
            _record_prompt_version(store, agent_spec_name, previous_prompt)
            _record_prompt_version(store, agent_spec_name, new_spec.system_prompt)

        if not request.headers.get("hx-request"):
            # A plain (non-htmx) form post — the "Revert to this version"
            # button on the regression page. Redirect rather than render a
            # fragment: this is a full navigation, and redirecting means a
            # page refresh afterward re-GETs instead of re-submitting the
            # write, and the corpus page re-renders the history fresh from
            # the store instead of an in-memory snapshot from before it.
            return RedirectResponse(
                url=f"/agent-specs/{agent_spec_name}/regression", status_code=303
            )

        return templates.TemplateResponse(
            request,
            "fragments/applied_fix.html",
            {
                "agent_spec_filename": path.name,
                "agent_spec_name": agent_spec_name,
                "new_system_prompt": new_spec.system_prompt,
            },
        )

    @app.get("/agent-specs/{agent_spec_name}/regression", response_class=HTMLResponse)
    def get_regression(request: Request, agent_spec_name: str) -> HTMLResponse:
        try:
            spec = _find_agent_spec_by_name(agent_spec_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with SqliteStore(app.state.db_path) as store:
            cases = store.get_regression_cases(spec.name)
            prompt_versions = store.get_system_prompt_versions(spec.name)
        return templates.TemplateResponse(
            request,
            "regression.html",
            {
                "agent_spec_name": spec.name,
                "cases": cases,
                "results": {},
                "models": list_models(),
                "prompt_history": _prompt_version_history(spec.system_prompt, prompt_versions),
            },
        )

    @app.post("/agent-specs/{agent_spec_name}/regression/run", response_class=HTMLResponse)
    def post_run_regression(
        request: Request,
        agent_spec_name: str,
        provider: str = Form("fake"),
        target_url: str = Form(""),
    ) -> HTMLResponse:
        load_settings()
        try:
            spec = _find_agent_spec_by_name(agent_spec_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with SqliteStore(app.state.db_path) as store:
            cases = store.get_regression_cases(spec.name)

        results = {}
        if cases:
            llm = _build_provider(provider)
            target = _build_target(SimpleNamespace(target_url=target_url or None), spec, llm)
            judge = build_two_tier_judge(spec, llm)
            results = {
                result.case_id: result
                for result in RegressionRunner(target, judge).replay_all(cases)
            }

        return templates.TemplateResponse(
            request, "fragments/regression_table.html", {"cases": cases, "results": results}
        )

    @app.post("/regression-cases/{case_id}/resolve", response_class=HTMLResponse)
    def post_resolve_case(request: Request, case_id: str) -> HTMLResponse:
        with SqliteStore(app.state.db_path) as store:
            case = store.get_regression_case(case_id)
            if case is None:
                raise HTTPException(status_code=404, detail=f"No regression case '{case_id}'.")
            resolved = case.model_copy(update={"status": "resolved"})
            store.save_regression_case(resolved)

        return templates.TemplateResponse(
            request, "fragments/regression_row.html", {"case": resolved, "result": None}
        )

    return app


app = create_app()
