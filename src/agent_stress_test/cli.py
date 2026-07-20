"""Command-line entry point — a composition root alongside ``orchestration/runner.py``.

Deliberately thin: the dashboard is the primary front end. Only ``run``
(scriptable, no-browser runs) and ``serve`` (launches the dashboard) live here.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from agent_stress_test.composition import (
    DEFAULT_SIM_MODEL,  # noqa: F401 (re-exported: tests import this from cli)
    build_provider,
    build_target,
    cluster_and_persist,
    reconcile_interrupted_runs,
    resolve_sim_provider_name,
    resolve_tactics,
)
from agent_stress_test.config import load_agent_spec, load_settings
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.ports import ProviderError
from agent_stress_test.report.export import build_report_bundle, to_json, to_markdown
from agent_stress_test.report.terminal import render_full_report
from agent_stress_test.store.migrations import ensure_current_or_raise
from agent_stress_test.store.sqlite_store import SqliteStore

_DEFAULT_AGENT_SPEC = (
    Path(__file__).resolve().parents[2] / "config" / "agents" / "sample_support_advanced.yaml"
)
_DEFAULT_DB = "runs.sqlite"


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    load_settings()  # loads .env so litellm sees the API key
    spec = load_agent_spec(args.agent_spec)
    llm = build_provider(args.provider)
    target = build_target(args, spec, llm)
    budget = args.budget
    sample_n = args.sample_n

    # Widen --tactics validation to include this agent's own approved
    # StressProfile personas, which build_runner() merges in automatically.
    with SqliteStore(args.db) as store:
        reconcile_interrupted_runs(store)
        profile = store.get_stress_profile(spec.name)
    extra_valid = [persona.name for persona in profile.personas] if profile else []
    tactics = resolve_tactics(args.tactics, extra_valid=extra_valid)

    # The simulator only writes one adversarial line per turn, so it defaults
    # to a cheaper model than the target tier.
    sim_provider_name = resolve_sim_provider_name(args)
    sim_llm = llm if sim_provider_name == args.provider else build_provider(sim_provider_name)

    # A single sample can only ever score 0.0, so skip the scorer below n=2.
    use_scorer = sample_n >= 2
    n_personas = len(tactics)
    # Rough estimate, not exact: DeepEvalConversationSearch runs one
    # conversation per persona up to `budget` turns each.
    nodes = n_personas * budget
    est_calls = nodes + (nodes * sample_n if use_scorer else 0)
    consistency = f"sample-n={sample_n}" if use_scorer else "off"
    sim_note = f", simulator={sim_provider_name}" if sim_provider_name != args.provider else ""
    # json/markdown feed CI gating via stdout, so suppress progress chatter.
    quiet = args.format != "rich"
    if not quiet:
        console.print(
            f"[dim]Running against [bold]{args.provider}[/bold] "
            f"(budget={budget} turns/persona, {n_personas} personas, "
            f"consistency={consistency}{sim_note}) - "
            f"up to ~{est_calls} model calls. This can take a while.[/dim]"
        )

    with SqliteStore(args.db) as store:
        runner = build_runner(
            agent_spec=spec,
            target=target,
            sim_provider=sim_llm,
            llm=llm,
            store=store,
            tactics=tactics,
            sample_n=sample_n,
        )
        if quiet:
            result = runner.run(provider_name=args.provider, budget=budget)
        else:
            with console.status("[bold]Stress-testing agent...[/bold]", spinner="dots"):
                result = runner.run(provider_name=args.provider, budget=budget)

        clusters = cluster_and_persist(result, store)
        verdicts = result.tree.all_verdicts()

        if args.format == "json":
            print(to_json(build_report_bundle(result.run, result.tree, verdicts, clusters)))
        elif args.format == "markdown":
            print(to_markdown(build_report_bundle(result.run, result.tree, verdicts, clusters)))
        else:
            console.print(f"Run ID: {result.run.id}")
            render_full_report(
                console,
                run=result.run,
                reliability=result.reliability,
                clusters=clusters,
                tree=result.tree,
                verdicts=verdicts,
            )
    return 0


def _cmd_serve(args: argparse.Namespace, console: Console) -> int:
    # Lazy import: `run` shouldn't pay for FastAPI/uvicorn/Jinja2 setup.
    import uvicorn

    from agent_stress_test.report.dashboard.server import create_app

    console.print(
        f"[dim]Serving dashboard at [bold]http://{args.host}:{args.port}[/bold] "
        f"(db={args.db}).[/dim]"
    )
    uvicorn.run(create_app(db_path=args.db), host=args.host, port=args.port)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-stress-test")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a stress test against a target agent.")
    run_parser.add_argument("--agent-spec", type=Path, default=_DEFAULT_AGENT_SPEC)
    run_parser.add_argument("--provider", default="fake")
    run_parser.add_argument(
        "--sim-provider",
        default=None,
        help=(
            "Model for the adversarial simulator (default: a cheap model, since "
            "it doesn't need target-tier quality; stays 'fake' if --provider is 'fake')."
        ),
    )
    run_parser.add_argument("--target-url", default=None)
    run_parser.add_argument("--db", default=_DEFAULT_DB)
    run_parser.add_argument(
        "--budget",
        type=int,
        default=6,
        help="Turns per persona conversation (default: 6).",
    )
    run_parser.add_argument(
        "--sample-n",
        type=int,
        default=1,
        help="Self-consistency samples per node; <2 disables the scorer (default: 1).",
    )
    run_parser.add_argument(
        "--tactics",
        default=None,
        help="Comma-separated subset of tactics to use (default: all).",
    )
    run_parser.add_argument(
        "--format",
        choices=["rich", "json", "markdown"],
        default="rich",
        help=(
            "Report format (default: rich, the colorful terminal report). "
            "'json'/'markdown' print a parseable report to stdout with no other "
            "output, for CI gating."
        ),
    )
    run_parser.set_defaults(func=_cmd_run)

    serve_parser = subparsers.add_parser("serve", help="Serve the web dashboard.")
    serve_parser.add_argument("--db", default=_DEFAULT_DB)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()
    try:
        ensure_current_or_raise(args.db)
        return args.func(args, console)
    except (ValueError, ProviderError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}", soft_wrap=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
