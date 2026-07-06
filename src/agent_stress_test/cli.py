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

from agent_stress_test.config import load_agent_spec, load_settings
from agent_stress_test.models import Cluster, Node, Run, Verdict
from agent_stress_test.orchestration.reliability import score_run
from agent_stress_test.orchestration.runner import build_runner
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import LLMProvider, Store, TargetAgent
from agent_stress_test.providers.embedder import HashingEmbedder
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.providers.litellm_provider import LiteLLMProvider
from agent_stress_test.reasoning.clusterer import FailureClusterer
from agent_stress_test.reasoning.simulator import default_registry
from agent_stress_test.report.terminal import render_full_report, render_replay
from agent_stress_test.store.sqlite_store import SqliteStore
from agent_stress_test.targets.http_agent import HttpAgent
from agent_stress_test.targets.sample_agent import SampleAgent

_DEFAULT_AGENT_SPEC = (
    Path(__file__).resolve().parents[2] / "config" / "agents" / "sample_support.yaml"
)
_DEFAULT_DB = "runs.sqlite"


def _build_provider(name: str) -> LLMProvider:
    if name == "fake":
        return FakeLLMProvider()
    return LiteLLMProvider(model=name)


def _build_target(args: argparse.Namespace, spec, llm: LLMProvider) -> TargetAgent:
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


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    load_settings()  # side effect: load .env so litellm sees the API key
    spec = load_agent_spec(args.agent_spec)
    llm = _build_provider(args.provider)
    target = _build_target(args, spec, llm)
    budget = args.budget
    sample_n = args.sample_n
    tactics = _resolve_tactics(args.tactics)

    # Self-consistency needs >= 2 samples to detect any disagreement; a single
    # sample can only ever score 0.0, so skip the scorer (and every one of its
    # calls) entirely below that threshold.
    use_scorer = sample_n >= 2
    n_tactics = len(tactics)
    nodes = 1 + budget * n_tactics
    est_calls = nodes + (nodes * sample_n if use_scorer else 0) + budget * n_tactics
    consistency = f"sample-n={sample_n}" if use_scorer else "off"
    console.print(
        f"[dim]Running against [bold]{args.provider}[/bold] "
        f"(budget={budget}, {n_tactics} tactics, consistency={consistency}) - "
        f"up to ~{est_calls} model calls. This can take a while.[/dim]"
    )

    with SqliteStore(args.db) as store:
        runner = build_runner(
            agent_spec=spec,
            target=target,
            sim_provider=llm,
            scorer_provider=llm if use_scorer else None,
            store=store,
            tactics=tactics,
            sample_n=sample_n,
        )
        with console.status("[bold]Stress-testing agent...[/bold]", spinner="dots"):
            result = runner.run(provider_name=args.provider, budget=budget)

        clusterer = FailureClusterer(HashingEmbedder())
        clusters = clusterer.cluster(
            result.tree.nodes(), result.tree.all_verdicts(), run_id=result.run.id
        )
        for cluster in clusters:
            store.save_cluster(cluster)

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-stress-test")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a stress test against a target agent.")
    run_parser.add_argument("--agent-spec", type=Path, default=_DEFAULT_AGENT_SPEC)
    run_parser.add_argument("--provider", default="fake")
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
