"""The web dashboard's FastAPI app — a second composition root, alongside
``cli.py``.

Routes here only translate HTTP <-> the same ``build_runner()``/``Runner``
wiring ``cli.py`` uses; the shared decision logic (which provider, which
tactics, how to reload a finished run) lives in ``composition.py`` so neither
front end duplicates it. The live-run registry and SSE scheduler live in
``live_events.py``; system-prompt diffing and version history live in
``prompt_diff.py``; cluster/conversation-verdict ranking shared with the CLI
report lives in ``report/shared.py``. This module owns what's left: the HTTP
routes themselves, plus the one background-thread job (``_execute_run``)
they launch.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from agent_stress_test.composition import (
    apply_profile_edits,
    build_provider,
    build_target,
    cluster_and_persist,
    load_bundle,
    resolve_cluster_remediation_target,
    resolve_sim_provider_name,
    resolve_tactics,
)
from agent_stress_test.config import load_agent_spec, load_settings
from agent_stress_test.config_writer import apply_system_prompt
from agent_stress_test.models import AgentSpec, Run, Verdict
from agent_stress_test.orchestration.regression import (
    RegressionRunner,
    promote_clusters_to_cases,
)
from agent_stress_test.orchestration.reliability import (
    ReliabilityReport,
    SeverityWeightedModel,
    TaskSuccessModel,
    UnweightedFailureModel,
    near_miss_ranking,
    score_run,
)
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.reasoning.judge import build_two_tier_judge
from agent_stress_test.reasoning.profiler import AgentProfiler
from agent_stress_test.reasoning.remediation import RemediationSuggester
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.report.dashboard.live_events import (
    live_trees,
    live_trees_lock,
    locked_cluster_ids,
    stream_run_events,
)
from agent_stress_test.report.dashboard.prompt_diff import (
    diff_blocks,
    prompt_version_history,
    record_prompt_version,
)
from agent_stress_test.report.shared import conversation_verdicts_by_leaf, ranked_clusters
from agent_stress_test.store.migrations import ensure_current_or_raise
from agent_stress_test.store.sqlite_store import SqliteStore

_DEFAULT_DB = "runs.sqlite"

# Phase C6's scoring-model picker: maps the <select>'s value to a ScoringModel
# class (see orchestration/reliability.py) — re-scores a run's already-loaded
# nodes/verdicts on demand, never touches what was actually persisted as
# Run.final_score.
_SCORING_MODELS = {
    SeverityWeightedModel.name: SeverityWeightedModel,
    UnweightedFailureModel.name: UnweightedFailureModel,
    TaskSuccessModel.name: TaskSuccessModel,
}
# server.py -> dashboard -> report -> agent_stress_test -> src -> repo root
_CONFIG_AGENTS_DIR = Path(__file__).resolve().parents[4] / "config" / "agents"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def list_agent_specs() -> list[dict[str, str]]:
    """The configured agent specs, as ``[{"id": <filename>, "name": <agent_spec.name>}]``."""
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


def _persona_options_for_spec(agent_spec: AgentSpec, store: SqliteStore) -> list[dict[str, str]]:
    """The run form's per-agent persona picker options: this agent's own
    stress profile personas if one has been generated, else the bundled
    tactic library.

    Fully consumable by a real run: ``build_runner()`` (see
    ``orchestration/runner.py``'s ``_profile_extra_personas``) merges this
    same spec's approved profile personas in automatically, and
    ``resolve_tactics``'s ``extra_valid`` (see ``_execute_run`` below)
    accepts a profile-sourced name explicitly selected here.
    """
    profile = store.get_stress_profile(agent_spec.name)
    if profile is not None and profile.personas:
        return [{"id": p.name, "description": p.scenario} for p in profile.personas]
    return list_tactics()


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
        agent_spec = load_agent_spec(agent_spec_path)
        args = SimpleNamespace(
            provider=provider,
            sim_provider=sim_provider,
            target_url=target_url,
        )
        llm = build_provider(provider)
        target = build_target(args, agent_spec, llm)
        with SqliteStore(db_path) as store:
            profile = store.get_stress_profile(agent_spec.name)
        extra_valid = [persona.name for persona in profile.personas] if profile else []
        tactics = resolve_tactics(tactics_arg, extra_valid=extra_valid)
        sim_provider_name = resolve_sim_provider_name(args)
        sim_llm = llm if sim_provider_name == provider else build_provider(sim_provider_name)

        with SqliteStore(db_path) as store:
            runner = build_runner(
                agent_spec=agent_spec,
                target=target,
                sim_provider=sim_llm,
                llm=llm,
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
        with live_trees_lock:
            live_trees.pop(run_id, None)


def create_app(db_path: str = _DEFAULT_DB) -> FastAPI:
    """Build the dashboard app. ``db_path`` is fixed here, at process startup —
    never accepted from a client request (see module docstring on trust)."""
    ensure_current_or_raise(db_path)
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

    @app.get("/agent-specs/personas", response_class=HTMLResponse)
    def get_personas_picker(request: Request, agent_spec_id: str) -> HTMLResponse:
        try:
            path = _resolve_agent_spec_path(agent_spec_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        agent_spec = load_agent_spec(path)
        with SqliteStore(app.state.db_path) as store:
            options = _persona_options_for_spec(agent_spec, store)
        return templates.TemplateResponse(
            request, "fragments/personas_picker.html", {"tactics": options}
        )

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
            agent_spec = load_agent_spec(agent_spec_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = str(uuid4())
        with SqliteStore(app.state.db_path) as store:
            store.save_run(
                Run(
                    id=run_id,
                    agent_spec=agent_spec,
                    provider=provider,
                    budget=budget,
                    status="running",
                    started_at=datetime.now(timezone.utc),
                )
            )

        tree = ConversationTree(run_id)
        with live_trees_lock:
            live_trees[run_id] = tree

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

            live_tree = live_trees.get(run_id)
            if run.status in ("pending", "running") and live_tree is not None:
                tree = live_tree
                verdicts = tree.all_verdicts()
                clusters = []
                reliability: ReliabilityReport | None = (
                    score_run(tree.nodes(), verdicts) if tree.nodes() else None
                )
            else:
                run, tree, verdicts, clusters = load_bundle(store, run_id)
                reliability = score_run(tree.nodes(), verdicts)

            locked = locked_cluster_ids(store, run.agent_spec.name)

        failures: list[Verdict] = [v for v in verdicts if not v.passed]
        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "run": run,
                "run_id": run_id,
                "run_provider": run.provider,
                "reliability": reliability,
                "ranked_clusters": ranked_clusters(clusters, verdicts),
                "locked_cluster_ids": locked,
                "failures": failures,
                "tree": tree,
                "near_misses": near_miss_ranking(tree.nodes(), verdicts),
                "conversation_groups": conversation_verdicts_by_leaf(verdicts),
            },
        )

    @app.get("/runs/{run_id}/reliability", response_class=HTMLResponse)
    def get_run_reliability(
        request: Request, run_id: str, model: str = "severity_weighted"
    ) -> HTMLResponse:
        """Phase C6's scoring-model picker: re-score this run's already-loaded
        nodes/verdicts under a different ScoringModel and return the gauge
        fragment re-rendered — never touches the persisted Run.final_score."""
        model_cls = _SCORING_MODELS.get(model)
        if model_cls is None:
            raise HTTPException(status_code=400, detail=f"Unknown scoring model '{model}'.")

        with SqliteStore(app.state.db_path) as store:
            run = store.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"No run found with id '{run_id}'.")

            live_tree = live_trees.get(run_id)
            if run.status in ("pending", "running") and live_tree is not None:
                nodes, verdicts = live_tree.nodes(), live_tree.all_verdicts()
            else:
                _run, tree, verdicts, _clusters = load_bundle(store, run_id)
                nodes = tree.nodes()

        reliability = score_run(nodes, verdicts, model=model_cls())
        return templates.TemplateResponse(
            request,
            "fragments/reliability_gauge.html",
            {"run_id": run_id, "reliability": reliability},
        )

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        with SqliteStore(app.state.db_path) as store:
            if store.get_run(run_id) is None:
                raise HTTPException(status_code=404, detail=f"No run found with id '{run_id}'.")
        return StreamingResponse(
            stream_run_events(run_id, app.state.db_path, templates), media_type="text/event-stream"
        )

    @app.post("/runs/{run_id}/clusters/{cluster_id}/lock", response_class=HTMLResponse)
    def post_lock_cluster(request: Request, run_id: str, cluster_id: str) -> HTMLResponse:
        with SqliteStore(app.state.db_path) as store:
            run, tree, verdicts, clusters = load_bundle(store, run_id)
            for case in promote_clusters_to_cases(run, tree, clusters, cluster_ids={cluster_id}):
                store.save_regression_case(case)
            locked = locked_cluster_ids(store, run.agent_spec.name)

        return templates.TemplateResponse(
            request,
            "fragments/cluster_table.html",
            {
                "ranked_clusters": ranked_clusters(clusters, verdicts),
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
            run, tree, _verdicts, clusters = load_bundle(store, run_id)
        try:
            node, rule, verdict = resolve_cluster_remediation_target(
                tree, clusters, cluster_id, run.agent_spec
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        suggestion = RemediationSuggester(build_provider(provider)).suggest(
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
                "diff_blocks": diff_blocks(run.agent_spec.system_prompt, suggestion.suggested_system_prompt),
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
        # addressed — see `record_prompt_version`). The first-ever apply for
        # an agent captures its original prompt as the baseline "Revision 1"
        # in the same stroke; every later apply is a no-op on whichever side
        # is already on file. A hosted deployment may give the person
        # clicking this button no shell/git access to the server, so "git
        # checkout" isn't a real safety net there — this is what actually
        # lets them restore any past version, regardless of how it's
        # deployed, including reapplying one they just undid.
        with SqliteStore(app.state.db_path) as store:
            record_prompt_version(store, agent_spec_name, previous_prompt)
            record_prompt_version(store, agent_spec_name, new_spec.system_prompt)

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

    @app.get("/agent-specs/{agent_spec_name}/profile", response_class=HTMLResponse)
    def get_profile(request: Request, agent_spec_name: str) -> HTMLResponse:
        try:
            _find_agent_spec_by_name(agent_spec_name)  # 404s on an unknown spec
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with SqliteStore(app.state.db_path) as store:
            profile = store.get_stress_profile(agent_spec_name)
        return templates.TemplateResponse(
            request,
            "profile.html",
            {"agent_spec_name": agent_spec_name, "profile": profile, "models": list_models()},
        )

    @app.post("/agent-specs/{agent_spec_name}/profile/generate", response_class=HTMLResponse)
    def post_generate_profile(
        request: Request, agent_spec_name: str, provider: str = Form("fake")
    ) -> HTMLResponse:
        load_settings()
        try:
            agent_spec = _find_agent_spec_by_name(agent_spec_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            profile = AgentProfiler(build_provider(provider)).profile(agent_spec)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with SqliteStore(app.state.db_path) as store:
            store.save_stress_profile(profile)

        return templates.TemplateResponse(
            request,
            "fragments/profile_editor.html",
            {"agent_spec_name": agent_spec_name, "profile": profile},
        )

    @app.post("/agent-specs/{agent_spec_name}/profile/save", response_class=HTMLResponse)
    async def post_save_profile(request: Request, agent_spec_name: str) -> HTMLResponse:
        form = await request.form()

        with SqliteStore(app.state.db_path) as store:
            existing = store.get_stress_profile(agent_spec_name)
            if existing is None:
                raise HTTPException(
                    status_code=404, detail=f"No stress profile for '{agent_spec_name}' yet."
                )

            updated = apply_profile_edits(
                existing,
                names=form.getlist("persona_name"),
                scenarios=form.getlist("persona_scenario"),
                user_descriptions=form.getlist("persona_user_description"),
                rule_ids=form.getlist("rule_id"),
                rule_texts=form.getlist("rule_text"),
                rule_severities=form.getlist("rule_severity"),
            )
            store.save_stress_profile(updated)

        return templates.TemplateResponse(
            request,
            "fragments/profile_editor.html",
            {"agent_spec_name": agent_spec_name, "profile": updated},
        )

    @app.get("/agent-specs/{agent_spec_name}/regression", response_class=HTMLResponse)
    def get_regression(request: Request, agent_spec_name: str) -> HTMLResponse:
        try:
            agent_spec = _find_agent_spec_by_name(agent_spec_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with SqliteStore(app.state.db_path) as store:
            cases = store.get_regression_cases(agent_spec.name)
            prompt_versions = store.get_system_prompt_versions(agent_spec.name)
        return templates.TemplateResponse(
            request,
            "regression.html",
            {
                "agent_spec_name": agent_spec.name,
                "cases": cases,
                "results": {},
                "models": list_models(),
                "prompt_history": prompt_version_history(agent_spec.system_prompt, prompt_versions),
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
            agent_spec = _find_agent_spec_by_name(agent_spec_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with SqliteStore(app.state.db_path) as store:
            cases = store.get_regression_cases(agent_spec.name)

        results = {}
        if cases:
            llm = build_provider(provider)
            target = build_target(
                SimpleNamespace(target_url=target_url or None), agent_spec, llm
            )
            judge = build_two_tier_judge(agent_spec, llm)
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
