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

import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from agent_stress_test.composition import (
    _build_provider,
    _build_target,
    _load_bundle,
    _resolve_sim_provider_name,
    _resolve_tactics,
    cluster_and_persist,
)
from agent_stress_test.config import load_agent_spec, load_settings
from agent_stress_test.models import Cluster, Run, Verdict
from agent_stress_test.orchestration.reliability import ReliabilityReport, score_run
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.orchestration.search import SEVERITY_WEIGHT
from agent_stress_test.orchestration.tree import ConversationTree
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
        use_scorer = sample_n >= 2

        with SqliteStore(db_path) as store:
            runner = build_runner(
                agent_spec=spec,
                target=target,
                sim_provider=sim_llm,
                scorer_provider=llm if use_scorer else None,
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


def _run_events(run_id: str, db_path: str) -> Iterator[str]:
    seen: set[str] = set()
    last_status: str | None = None
    last_node_count = 0
    while True:
        tree = _live_trees.get(run_id)

        # The gauge's initial render is a snapshot from whenever the page was
        # loaded; without this, it never moves again until the one terminal
        # push below, so a run can sit there reading 100% (or whatever the
        # very first node scored) for its entire duration. Re-score and push
        # every time the live tree gains a node, so it tracks the run instead
        # of a first-paint snapshot.
        if tree is not None:
            node_count = len(tree.nodes())
            if node_count != last_node_count:
                last_node_count = node_count
                reliability = score_run(tree.nodes(), tree.all_verdicts())
                yield _sse(
                    "reliability",
                    templates.get_template("fragments/reliability_gauge.html").render(
                        reliability=reliability
                    ),
                )

        for verdict in tree.failures() if tree else []:
            if verdict.id not in seen:
                seen.add(verdict.id)
                node = tree.get(verdict.node_id)
                html = templates.get_template("fragments/failure_row.html").render(
                    verdict=verdict, node=node
                )
                # The feed's "No failures yet." placeholder was rendered
                # server-side before any failure existed; a beforeend swap
                # only ever appends, so it never removes that placeholder on
                # its own — delete it out-of-band on the first failure.
                html = '<div id="no-failures" hx-swap-oob="delete"></div>' + html
                yield _sse("failure", html)

        with SqliteStore(db_path) as store:
            run = store.get_run(run_id)
        status = run.status if run is not None else "pending"
        if status != last_status:
            last_status = status
            html = templates.get_template("fragments/status_badge.html").render(run=run)
            yield _sse("status", html)
        if status in ("completed", "failed"):
            with SqliteStore(db_path) as store:
                _run, final_tree, final_verdicts, final_clusters = _load_bundle(store, run_id)
            reliability = score_run(final_tree.nodes(), final_verdicts)
            ranked = _ranked_clusters(final_clusters, final_verdicts)
            yield _sse(
                "reliability",
                templates.get_template("fragments/reliability_gauge.html").render(
                    reliability=reliability
                ),
            )
            yield _sse(
                "clusters",
                templates.get_template("fragments/cluster_table.html").render(
                    ranked_clusters=ranked
                ),
            )
            # Representative Transcripts is a top-level block, not swapped by
            # id like the gauge/cluster table above — clustering (and thus
            # which nodes are "representative") only exists once the run is
            # done, so without this push the section stays absent from the
            # DOM until a full page reload re-renders it from scratch.
            yield _sse(
                "transcripts",
                templates.get_template("fragments/transcripts_section.html").render(
                    ranked_clusters=ranked,
                    tree=final_tree,
                    failures=[v for v in final_verdicts if not v.passed],
                ),
            )
            return
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

        failures: list[Verdict] = [v for v in verdicts if not v.passed]
        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "run": run,
                "reliability": reliability,
                "ranked_clusters": _ranked_clusters(clusters, verdicts),
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

    return app


app = create_app()
