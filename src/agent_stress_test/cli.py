"""Command-line entry point — the composition root (with runner.py).

This is the one place, alongside ``orchestration/runner.py``, allowed to
construct concrete adapters (``SqliteStore``, ``FakeLLMProvider``,
``LiteLLMProvider``, ``SampleAgent``, ``HttpAgent``) and wire them together.
Everything it renders goes through the pure ``report/terminal.py`` layer.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from agent_stress_test.composition import (
    _DEFAULT_SIM_MODEL,  # noqa: F401 (re-exported: tests import this from cli)
    _build_provider,
    _build_target,
    _load_bundle,
    _resolve_sim_provider_name,
    _resolve_tactics,
    cluster_and_persist,
)
from agent_stress_test.config import load_agent_spec, load_settings
from agent_stress_test.orchestration.reliability import score_run
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.report.terminal import render_full_report, render_replay
from agent_stress_test.store.sqlite_store import SqliteStore

_DEFAULT_AGENT_SPEC = (
    Path(__file__).resolve().parents[2] / "config" / "agents" / "sample_support.yaml"
)
_DEFAULT_DB = "runs.sqlite"


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    load_settings()  # side effect: load .env so litellm sees the API key
    spec = load_agent_spec(args.agent_spec)
    llm = _build_provider(args.provider)
    target = _build_target(args, spec, llm)
    budget = args.budget
    sample_n = args.sample_n
    tactics = _resolve_tactics(args.tactics)

    # The simulator's job (write one adversarial customer line) doesn't need
    # the target-tier model, so it defaults to a cheaper one; the scorer stays
    # on the target's own model, since it measures *that* model's instability
    # by resampling it — a different model there would score the wrong thing.
    sim_provider_name = _resolve_sim_provider_name(args)
    sim_llm = llm if sim_provider_name == args.provider else _build_provider(sim_provider_name)

    # Self-consistency needs >= 2 samples to detect any disagreement; a single
    # sample can only ever score 0.0, so skip the scorer (and every one of its
    # calls) entirely below that threshold.
    use_scorer = sample_n >= 2
    n_tactics = len(tactics)
    nodes = 1 + budget * n_tactics
    est_calls = nodes + (nodes * sample_n if use_scorer else 0) + budget * n_tactics
    consistency = f"sample-n={sample_n}" if use_scorer else "off"
    sim_note = f", simulator={sim_provider_name}" if sim_provider_name != args.provider else ""
    console.print(
        f"[dim]Running against [bold]{args.provider}[/bold] "
        f"(budget={budget}, {n_tactics} tactics, consistency={consistency}{sim_note}) - "
        f"up to ~{est_calls} model calls. This can take a while.[/dim]"
    )

    with SqliteStore(args.db) as store:
        runner = build_runner(
            agent_spec=spec,
            target=target,
            sim_provider=sim_llm,
            scorer_provider=llm if use_scorer else None,
            store=store,
            tactics=tactics,
            sample_n=sample_n,
        )
        with console.status("[bold]Stress-testing agent...[/bold]", spinner="dots"):
            result = runner.run(provider_name=args.provider, budget=budget)

        clusters = cluster_and_persist(result, store)

        console.print(f"Run ID: {result.run.id}")
        render_full_report(
            console,
            run=result.run,
            reliability=result.reliability,
            clusters=clusters,
            tree=result.tree,
            verdicts=result.tree.all_verdicts(),
        )
    return 0


def _cmd_report(args: argparse.Namespace, console: Console) -> int:
    with SqliteStore(args.db) as store:
        run, tree, verdicts, clusters = _load_bundle(store, args.run_id)
        reliability = score_run(tree.nodes(), verdicts)
        render_full_report(
            console, run=run, reliability=reliability, clusters=clusters, tree=tree, verdicts=verdicts
        )
    return 0


def _cmd_replay(args: argparse.Namespace, console: Console) -> int:
    with SqliteStore(args.db) as store:
        _run, tree, verdicts, _clusters = _load_bundle(store, args.run_id)
        node_ids = [v.node_id for v in tree.failures()]
        render_replay(console, tree=tree, node_ids=node_ids, verdicts=verdicts)
    return 0


def _cmd_serve(args: argparse.Namespace, console: Console) -> int:
    # Imported lazily so `run`/`report`/`replay` never pay for importing
    # FastAPI/uvicorn or building the Jinja2 environment.
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
        "--budget", type=int, default=6, help="Search expansions (default: 6)."
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
    run_parser.set_defaults(func=_cmd_run)

    report_parser = subparsers.add_parser("report", help="Show the report for a stored run.")
    report_parser.add_argument("run_id")
    report_parser.add_argument("--db", default=_DEFAULT_DB)
    report_parser.set_defaults(func=_cmd_report)

    replay_parser = subparsers.add_parser("replay", help="Replay a stored run's failing transcripts.")
    replay_parser.add_argument("run_id")
    replay_parser.add_argument("--db", default=_DEFAULT_DB)
    replay_parser.set_defaults(func=_cmd_replay)

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
        return args.func(args, console)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
