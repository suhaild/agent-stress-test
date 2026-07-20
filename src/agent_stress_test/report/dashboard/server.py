"""The web dashboard's FastAPI app — a second composition root, alongside
``cli.py``. Routes translate HTTP <-> the same ``build_runner()``/``Runner``
wiring ``cli.py`` uses; shared decision logic lives in ``composition.py``,
the live-run registry/SSE scheduler in ``live_events.py``, prompt diffing in
``prompt_diff.py``, and cluster/verdict ranking in ``report/shared.py``.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates

from agent_stress_test.composition import (
    apply_profile_edits,
    build_provider,
    build_target,
    cluster_and_persist,
    load_bundle,
    load_cross_run_bundle,
    reconcile_interrupted_runs,
    remove_candidate_rule,
    resolve_cluster_remediation_target,
    resolve_sim_provider_name,
    resolve_tactics,
)
from agent_stress_test.config import load_agent_spec, load_settings
from agent_stress_test.config_writer import apply_candidate_rule, apply_system_prompt
from agent_stress_test.models import AgentSpec, Rule, Run, Verdict
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
from agent_stress_test.orchestration.rule_coverage import rule_coverage
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.orchestration.tree_viz import build_tree_viz
from agent_stress_test.reasoning.judge import build_two_tier_judge
from agent_stress_test.report.export import build_report_bundle, to_json, to_markdown
from agent_stress_test.reasoning.profiler import AgentProfiler
from agent_stress_test.reasoning.remediation import RemediationSuggester
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.reasoning.summary import RunSummarizer
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
from agent_stress_test.report.shared import (
    conversation_verdicts_by_leaf,
    executive_summary_context,
    ranked_clusters,
    trend_chart_points,
)
from agent_stress_test.store.migrations import ensure_current_or_raise
from agent_stress_test.store.sqlite_store import SqliteStore

_DEFAULT_DB = "runs.sqlite"

# Maps the scoring-model <select>'s value to a ScoringModel class; re-scores
# on demand, never touches what was persisted as Run.final_score.
_SCORING_MODELS = {
    SeverityWeightedModel.name: SeverityWeightedModel,
    UnweightedFailureModel.name: UnweightedFailureModel,
    TaskSuccessModel.name: TaskSuccessModel,
}
# server.py -> dashboard -> report -> agent_stress_test -> src -> repo root
_CONFIG_AGENTS_DIR = Path(__file__).resolve().parents[4] / "config" / "agents"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Display-only label/detail pairs for the New Run form's Target Agent card;
# never read back from the client.
_TARGET_KIND_INFO = {
    "sample": (
        "Bundled Agent · Narrated Tools",
        "The bundled demo agent — narrates tool use as free-text ReAct reasoning; nothing is actually executed.",
    ),
    "sample_advanced": (
        "Bundled Agent · Real Tools",
        "The bundled demo agent's harder sibling — every declared tool call actually executes against an in-memory fake backend.",
    ),
    "http": (
        "HTTP Endpoint",
        "A bring-your-own agent reached over HTTP/JSON (see targets/http_agent.py).",
    ),
    "python": (
        "Python Function",
        "A bring-your-own agent wired in-process as a plain Python callable (see targets/python_fn.py).",
    ),
    "subprocess": (
        "Subprocess",
        "A bring-your-own agent driven over stdin/stdout JSON framing — can be written in any language (see targets/subprocess_agent.py).",
    ),
    "provider": (
        "Direct Model · Native Tools",
        "A bare model id driven directly through litellm's own tool-calling, using this spec's tools/system prompt (see targets/provider_agent.py).",
    ),
}

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def list_agent_specs() -> list[dict[str, Any]]:
    """The configured agent specs as ``[{"id": <filename>, "name": <display
    name>, ...}]``; "id" is always the real filename, used by
    ``_resolve_agent_spec_path``. The remaining fields are static, file-only
    metadata embedded so the New Run form's Target Agent card can update
    client-side without a round trip."""
    specs = [
        (path.name, load_agent_spec(path)) for path in sorted(_CONFIG_AGENTS_DIR.glob("*.yaml"))
    ]
    return [
        {
            "id": filename,
            "name": spec.display_name or spec.name,
            "domain": spec.domain,
            "purpose": spec.purpose,
            "tools_count": len(spec.tools),
            "tool_names": [tool.name for tool in spec.tools],
            "rules_count": len(spec.rules),
            "target_kind_label": _TARGET_KIND_INFO.get(
                spec.target.kind if spec.target else "sample", ("Custom Target", "")
            )[0],
            "target_kind_detail": _TARGET_KIND_INFO.get(
                spec.target.kind if spec.target else "sample", ("Custom Target", "")
            )[1],
        }
        for filename, spec in specs
    ]


def list_tactics() -> list[dict[str, str]]:
    """The built-in tactic library, read from the same registry the
    simulator uses so the picker can't drift from what actually runs."""
    registry = default_registry()
    return [
        {"id": name, "description": registry.get(name).description}
        for name in registry.names()
    ]


def _personas_picker_context(agent_spec: AgentSpec, store: SqliteStore) -> dict:
    """The run form's Attack Tactics picker context: this agent's own stress
    profile personas if one exists, else the bundled tactic library."""
    profile = store.get_stress_profile(agent_spec.name)
    has_profile = profile is not None and bool(profile.personas)
    tactics = (
        [{"id": p.name, "description": p.scenario} for p in profile.personas]
        if has_profile
        else list_tactics()
    )
    candidate_rule_count = len(profile.candidate_rules) if profile else 0
    return {
        "tactics": tactics,
        "agent_spec_name": agent_spec.name,
        "has_profile": has_profile,
        "candidate_rule_count": candidate_rule_count,
    }


def list_models() -> list[dict[str, str]]:
    """Curated LLM ids for the model pickers. Not exhaustive — the picker is
    a ``<datalist>`` suggestion list, not a closed enum; any litellm
    provider/model string still works."""
    return [
        {"id": "fake", "label": "Offline Test Double — no API key needed"},
        {"id": "anthropic/claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 — cheap & fast"},
        {"id": "anthropic/claude-sonnet-5", "label": "Claude Sonnet 5 — balanced"},
        {"id": "anthropic/claude-opus-4-8", "label": "Claude Opus 4.8 — most capable"},
    ]


def _resolve_agent_spec_path(agent_spec_id: str) -> Path:
    """Map a client-submitted id back to a real file, refusing anything not
    in the enumerated list — never a raw filesystem path from the browser."""
    if any(entry["id"] == agent_spec_id for entry in list_agent_specs()):
        return _CONFIG_AGENTS_DIR / agent_spec_id
    raise ValueError(f"Unknown agent spec '{agent_spec_id}'.")


def _find_agent_spec_path_by_name(agent_spec_name: str) -> Path:
    """Resolve a spec's ``.name`` back to its current YAML file, so a
    regression replay or applied fix acts on the live file, not a copy
    frozen in an old Run."""
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
    """The background-thread target: mirrors ``cli.py``'s run wiring, but
    reports failure via the store instead of letting an exception crash a
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


def _require_terminal_run(store: SqliteStore, run_id: str) -> Run:
    """Nodes/verdicts/clusters are only persisted once a run finishes, so
    export/lock/suggest-fix against a still-running run would otherwise
    silently return a near-empty bundle instead of a clear error."""
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No run found with id '{run_id}'.")
    if run.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run '{run_id[:8]}' is still {run.status} -- its report isn't ready yet. "
                "Wait for the run to finish before exporting, locking a cluster, or "
                "requesting a suggested fix."
            ),
        )
    return run


def create_app(db_path: str = _DEFAULT_DB) -> FastAPI:
    """Build the dashboard app. ``db_path`` is fixed at process startup,
    never accepted from a client request."""
    ensure_current_or_raise(db_path)
    with SqliteStore(db_path) as store:
        reconcile_interrupted_runs(store)
    app = FastAPI(title="Agent Stress-Test Dashboard")
    app.state.db_path = db_path

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        agent_specs = list_agent_specs()
        with SqliteStore(app.state.db_path) as store:
            recent_runs = store.list_runs()
            # Must match what get_personas_picker returns for the <select>'s
            # default option, or the picker is for the wrong agent on load.
            picker_context = (
                _personas_picker_context(
                    load_agent_spec(_CONFIG_AGENTS_DIR / agent_specs[0]["id"]), store
                )
                if agent_specs
                else {"tactics": list_tactics(), "agent_spec_name": "", "has_profile": False}
            )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "agent_specs": agent_specs,
                "recent_runs": recent_runs,
                "models": list_models(),
                **picker_context,
            },
            # Without no-store, browser back/forward can restore a cached
            # snapshot from before a run finished, hiding newly completed runs.
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/agent-specs")
    def get_agent_specs() -> list[dict[str, Any]]:
        return list_agent_specs()

    @app.get("/agent-specs/personas", response_class=HTMLResponse)
    def get_personas_picker(request: Request, agent_spec_id: str) -> HTMLResponse:
        try:
            path = _resolve_agent_spec_path(agent_spec_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        agent_spec = load_agent_spec(path)
        with SqliteStore(app.state.db_path) as store:
            context = _personas_picker_context(agent_spec, store)
        return templates.TemplateResponse(request, "fragments/personas_picker.html", context)

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
                cross_run = None
            else:
                run, tree, verdicts, clusters = load_bundle(store, run_id)
                reliability = score_run(tree.nodes(), verdicts)
                cross_run = load_cross_run_bundle(store, run, clusters, verdicts)

            locked = locked_cluster_ids(store, run.agent_spec.name)

        failures: list[Verdict] = [v for v in verdicts if not v.passed]
        near_misses = near_miss_ranking(tree.nodes(), verdicts)
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
                "near_misses": near_misses,
                "conversation_groups": conversation_verdicts_by_leaf(verdicts),
                "cross_run": cross_run,
                "trend_points": trend_chart_points(cross_run.trend) if cross_run else [],
                "rule_coverage": rule_coverage(run.agent_spec.rules, verdicts),
                "tree_viz": build_tree_viz(tree, verdicts),
                **(
                    executive_summary_context(
                        tree.nodes(), verdicts, clusters, reliability, near_misses
                    )
                    if reliability is not None
                    else {"summary": None, "fix_first": []}
                ),
            },
        )

    @app.get("/runs/{run_id}/reliability", response_class=HTMLResponse)
    def get_run_reliability(
        request: Request, run_id: str, model: str = "severity_weighted"
    ) -> HTMLResponse:
        """Re-score this run's already-loaded nodes/verdicts under a
        different ScoringModel; never touches the persisted Run.final_score."""
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

    @app.post("/runs/{run_id}/summary/llm", response_class=HTMLResponse)
    def post_llm_summary(
        request: Request, run_id: str, provider: str = Form("fake")
    ) -> HTMLResponse:
        """Opt-in LLM rephrasing of the deterministic summary — only spends a
        real call when the user explicitly clicks for it."""
        with SqliteStore(app.state.db_path) as store:
            _require_terminal_run(store, run_id)
            run, tree, verdicts, clusters = load_bundle(store, run_id)
        reliability = score_run(tree.nodes(), verdicts)
        near_misses = near_miss_ranking(tree.nodes(), verdicts)
        summary = executive_summary_context(
            tree.nodes(), verdicts, clusters, reliability, near_misses
        )["summary"]

        llm_text = RunSummarizer(build_provider(provider)).summarize(summary.text)
        return templates.TemplateResponse(
            request, "fragments/llm_summary.html", {"llm_text": llm_text}
        )

    @app.get("/runs/{run_id}/export.html", response_class=HTMLResponse)
    def get_export_html(request: Request, run_id: str) -> HTMLResponse:
        """A static, self-contained HTML export — ``run_id=None`` suppresses
        every fragment's interactive controls (Lock/Suggest-Fix, the
        scoring-model picker) so the page is safe to save or forward
        standalone. "PDF export" is just the browser's own Print-to-PDF on
        this page, not a server-side PDF library."""
        with SqliteStore(app.state.db_path) as store:
            _require_terminal_run(store, run_id)
            run, tree, verdicts, clusters = load_bundle(store, run_id)
        bundle = build_report_bundle(run, tree, verdicts, clusters)
        return templates.TemplateResponse(
            request,
            "export.html",
            {
                "run": run,
                "run_id": None,
                "run_provider": run.provider,
                "reliability": bundle.reliability,
                "ranked_clusters": bundle.ranked_clusters,
                "locked_cluster_ids": set(),
                "failures": [v for v in verdicts if not v.passed],
                "tree": tree,
                "near_misses": bundle.near_misses,
                "conversation_groups": bundle.conversation_groups,
                "rule_coverage": bundle.rule_coverage,
                "summary": bundle.summary,
                "fix_first": bundle.fix_first,
                "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            },
        )

    @app.get("/runs/{run_id}/export.json")
    def get_export_json(run_id: str) -> Response:
        with SqliteStore(app.state.db_path) as store:
            _require_terminal_run(store, run_id)
            run, tree, verdicts, clusters = load_bundle(store, run_id)
        bundle = build_report_bundle(run, tree, verdicts, clusters)
        return Response(
            content=to_json(bundle),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="report-{run_id[:8]}.json"'},
        )

    @app.get("/runs/{run_id}/export.md")
    def get_export_markdown(run_id: str) -> PlainTextResponse:
        with SqliteStore(app.state.db_path) as store:
            _require_terminal_run(store, run_id)
            run, tree, verdicts, clusters = load_bundle(store, run_id)
        bundle = build_report_bundle(run, tree, verdicts, clusters)
        return PlainTextResponse(
            content=to_markdown(bundle),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="report-{run_id[:8]}.md"'},
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
            _require_terminal_run(store, run_id)
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
            _require_terminal_run(store, run_id)
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

        # Records both ends of this write as distinct versions so any past
        # version (including one just undone) stays restorable later.
        with SqliteStore(app.state.db_path) as store:
            record_prompt_version(store, agent_spec_name, previous_prompt)
            record_prompt_version(store, agent_spec_name, new_spec.system_prompt)

        if not request.headers.get("hx-request"):
            # A plain (non-htmx) form post from the regression page's
            # "Revert to this version" button — redirect so a refresh re-GETs
            # instead of resubmitting the write.
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
            picker_context = _personas_picker_context(agent_spec, store)

        # Also renders an out-of-band personas_picker.html, updating the New
        # Run form's tactics picker if it's present in the requesting page.
        return templates.TemplateResponse(
            request,
            "fragments/profile_generate_result.html",
            {"agent_spec_name": agent_spec_name, "profile": profile, **picker_context},
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

    @app.post("/agent-specs/{agent_spec_name}/candidate-rules/apply", response_class=HTMLResponse)
    def post_apply_candidate_rule(
        request: Request,
        agent_spec_name: str,
        rule_id: str = Form(...),
        rule_text: str = Form(...),
        rule_severity: str = Form(...),
    ) -> HTMLResponse:
        """Promotes one candidate rule from this agent's stress profile into
        a real, enforced rule on the spec's own YAML."""
        if not rule_id.strip():
            raise HTTPException(status_code=400, detail="Give this rule an id before applying it.")
        try:
            agent_spec_path = _find_agent_spec_path_by_name(agent_spec_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            rule = Rule(id=rule_id, text=rule_text, severity=rule_severity)
            apply_candidate_rule(agent_spec_path, rule)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        remaining_count = None
        with SqliteStore(app.state.db_path) as store:
            profile = store.get_stress_profile(agent_spec_name)
            if profile is not None:
                updated = remove_candidate_rule(profile, rule_id)
                store.save_stress_profile(updated)
                remaining_count = len(updated.candidate_rules)

        return templates.TemplateResponse(
            request,
            "fragments/candidate_rule_applied.html",
            {"rule": rule, "remaining_count": remaining_count},
        )

    @app.post(
        "/agent-specs/{agent_spec_name}/candidate-rules/apply-all", response_class=HTMLResponse
    )
    async def post_apply_all_candidate_rules(
        request: Request, agent_spec_name: str
    ) -> HTMLResponse:
        """Bulk sibling of ``post_apply_candidate_rule``: writes every
        candidate rule into the spec's YAML in one pass. A collision (id
        already on the spec) is skipped and reported rather than aborting
        the whole batch."""
        form = await request.form()
        rule_ids = form.getlist("rule_id")
        rule_texts = form.getlist("rule_text")
        rule_severities = form.getlist("rule_severity")

        try:
            agent_spec_path = _find_agent_spec_path_by_name(agent_spec_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        applied: list[Rule] = []
        skipped: list[tuple[str, str]] = []
        for rule_id, rule_text, rule_severity in zip(rule_ids, rule_texts, rule_severities):
            if not rule_id.strip():
                continue  # a brand-new, never-saved "+Add Rule" row has no id yet
            rule = Rule(id=rule_id, text=rule_text, severity=rule_severity)
            try:
                apply_candidate_rule(agent_spec_path, rule)
                applied.append(rule)
            except ValueError as exc:
                skipped.append((rule_id, str(exc)))

        remaining_rules: list[Rule] = []
        with SqliteStore(app.state.db_path) as store:
            profile = store.get_stress_profile(agent_spec_name)
            if profile is not None:
                for rule in applied:
                    profile = remove_candidate_rule(profile, rule.id)
                if applied:
                    store.save_stress_profile(profile)
                remaining_rules = profile.candidate_rules

        return templates.TemplateResponse(
            request,
            "fragments/rules_section.html",
            {
                "agent_spec_name": agent_spec_name,
                "rules": remaining_rules,
                "bulk_result": {"applied": applied, "skipped": skipped},
            },
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
