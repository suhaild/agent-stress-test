"""Command-line entry point — the composition root (with runner.py).

This is the one place, alongside ``orchestration/runner.py``, allowed to
construct concrete adapters (``SqliteStore``, ``FakeLLMProvider``,
``LiteLLMProvider``, ``SampleAgent``, ``HttpAgent``) and wire them together.
Everything it renders goes through the pure ``report/terminal.py`` layer.

Deliberately thin: the dashboard (``report/dashboard/server.py``) is the one
real front end — every control and report surface lives there. This module
keeps only what the dashboard can't do for itself: ``run`` (a scriptable,
no-browser way to kick off a stress test, e.g. from CI) and ``serve`` (the
thing that actually launches the dashboard — also how ``pyproject.toml``'s
``agent-stress-test`` console script boots it). Every other command that
used to live here (``report``, ``replay``, ``lock``, ``resolve``,
``suggest-fix``, ``regress``, ``profile``) now has a dashboard equivalent
with no CLI-only functionality left behind.
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
    load_settings()  # side effect: load .env so litellm sees the API key
    spec = load_agent_spec(args.agent_spec)
    llm = build_provider(args.provider)
    target = build_target(args, spec, llm)
    budget = args.budget
    sample_n = args.sample_n

    # A short-lived peek at this agent's own approved StressProfile (if any),
    # so an explicit --tactics naming one of its personas validates here
    # instead of being rejected before build_runner() (which merges profile
    # personas in automatically) ever sees it.
    with SqliteStore(args.db) as store:
        reconcile_interrupted_runs(store)
        profile = store.get_stress_profile(spec.name)
    extra_valid = [persona.name for persona in profile.personas] if profile else []
    tactics = resolve_tactics(args.tactics, extra_valid=extra_valid)

    # The simulator's job (write one adversarial customer line) doesn't need
    # the target-tier model, so it defaults to a cheaper one.
    sim_provider_name = resolve_sim_provider_name(args)
    sim_llm = llm if sim_provider_name == args.provider else build_provider(sim_provider_name)

    # Self-consistency needs >= 2 samples to detect any disagreement; a single
    # sample can only ever score 0.0, so skip the scorer (and every one of its
    # extra target calls) entirely below that threshold. build_runner()
    # resamples the target itself, so this applies to any target, not just
    # the bundled SampleAgent.
    use_scorer = sample_n >= 2
    n_personas = len(tactics)
    # build_runner() wires DeepEvalConversationSearch by default: one
    # independent conversation per persona, up to `budget` turns each (see
    # its own docstring) — not GreedyBestFirstSearch's expansion-tree shape,
    # so this is a rough per-turn estimate, not an exact call count.
    nodes = n_personas * budget
    est_calls = nodes + (nodes * sample_n if use_scorer else 0)
    consistency = f"sample-n={sample_n}" if use_scorer else "off"
    sim_note = f", simulator={sim_provider_name}" if sim_provider_name != args.provider else ""
    # Phase RE4: --format json/markdown is for CI gating, which redirects
    # stdout and expects it to parse cleanly — no progress chatter or a Rich
    # spinner mixed in, unlike the default --format rich (a human at a
    # terminal, watching a live run).
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
    # Imported lazily so `run` never pays for importing FastAPI/uvicorn or
    # building the Jinja2 environment.
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
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
