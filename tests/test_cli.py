import argparse
import json
import re

import pytest

from agent_stress_test.cli import (
    DEFAULT_SIM_MODEL,
    main,
    resolve_sim_provider_name,
    resolve_tactics,
)
from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import ProfilePersona, Run, StressProfile
from agent_stress_test.store.sqlite_store import SqliteStore

_RUN_ID_RE = re.compile(r"Run ID:\s*(\S+)")


def _args(**overrides) -> argparse.Namespace:
    defaults = {"provider": "fake", "sim_provider": None}
    return argparse.Namespace(**{**defaults, **overrides})


def test_sim_provider_defaults_to_cheap_model_for_a_real_provider():
    assert resolve_sim_provider_name(_args(provider="anthropic/claude-sonnet-5")) == (
        DEFAULT_SIM_MODEL
    )


def test_sim_provider_stays_fake_when_main_provider_is_fake():
    assert resolve_sim_provider_name(_args(provider="fake")) == "fake"


def test_sim_provider_explicit_override_wins():
    assert (
        resolve_sim_provider_name(
            _args(provider="anthropic/claude-sonnet-5", sim_provider="openai/gpt-4o")
        )
        == "openai/gpt-4o"
    )


def test_resolve_tactics_with_no_arg_returns_only_bundled_tactics_by_default():
    assert set(resolve_tactics(None)) == {
        "self-contradiction",
        "urgency-pressure",
        "hostile",
        "stale-recall",
        "scope-expansion",
    }


def test_resolve_tactics_rejects_a_name_outside_the_bundled_registry():
    with pytest.raises(ValueError, match="Unknown tactic"):
        resolve_tactics("symptom-minimizer")


def test_resolve_tactics_accepts_an_extra_valid_name():
    # A profile-sourced persona name — accepted only because it's passed as
    # extra_valid, exactly what build_runner()'s callers do once they've
    # peeked at the spec's own StressProfile.
    assert resolve_tactics("symptom-minimizer", extra_valid=["symptom-minimizer"]) == [
        "symptom-minimizer"
    ]


def test_resolve_tactics_still_rejects_a_name_not_in_bundled_or_extra():
    with pytest.raises(ValueError, match="Unknown tactic"):
        resolve_tactics("nonexistent-persona", extra_valid=["symptom-minimizer"])


def test_resolve_tactics_with_no_arg_includes_extra_valid_by_default():
    resolved = resolve_tactics(None, extra_valid=["symptom-minimizer"])
    assert "symptom-minimizer" in resolved
    assert "hostile" in resolved


def test_resolve_tactics_extra_valid_does_not_duplicate_a_bundled_name():
    resolved = resolve_tactics(None, extra_valid=["hostile"])
    assert resolved.count("hostile") == 1


def test_cli_run_executes_against_the_fake_provider(tmp_path, sample_agent_spec_path, capsys):
    db_path = tmp_path / "runs.sqlite"

    exit_code = main(
        [
            "run",
            "--agent-spec",
            str(sample_agent_spec_path),
            "--provider",
            "fake",
            "--db",
            str(db_path),
            "--budget",
            "2",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    match = _RUN_ID_RE.search(out)
    assert match, f"expected a 'Run ID: ...' line, got:\n{out}"
    assert "Reliability" in out

    with SqliteStore(db_path) as store:
        assert store.get_run(match.group(1)) is not None


def test_cli_run_reconciles_a_stale_running_run_left_by_a_previous_process(
    tmp_path, sample_agent_spec_path, capsys
):
    """A prior invocation Ctrl+C'd mid-run leaves its row stuck "running"
    forever (KeyboardInterrupt skips right past the existing failure
    handling) -- the next `run` invocation against the same db must close
    it out itself before doing anything else."""
    db_path = tmp_path / "runs.sqlite"
    with SqliteStore(db_path) as store:
        store.save_run(
            Run(
                id="stuck-run",
                agent_spec=load_agent_spec(sample_agent_spec_path),
                provider="fake",
                status="running",
            )
        )

    exit_code = main(
        [
            "run",
            "--agent-spec",
            str(sample_agent_spec_path),
            "--provider",
            "fake",
            "--db",
            str(db_path),
            "--budget",
            "2",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    with SqliteStore(db_path) as store:
        stuck = store.get_run("stuck-run")
    assert stuck.status == "failed"
    assert "Interrupted" in stuck.error


def test_cli_run_format_json_emits_only_parseable_json(tmp_path, sample_agent_spec_path, capsys):
    db_path = tmp_path / "runs.sqlite"

    exit_code = main(
        [
            "run",
            "--agent-spec",
            str(sample_agent_spec_path),
            "--provider",
            "fake",
            "--db",
            str(db_path),
            "--budget",
            "2",
            "--format",
            "json",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(out)  # raises if anything but the JSON made it to stdout
    assert "reliability" in payload
    assert "score" in payload["reliability"]


def test_cli_run_format_markdown_emits_only_the_markdown_report(
    tmp_path, sample_agent_spec_path, capsys
):
    db_path = tmp_path / "runs.sqlite"

    exit_code = main(
        [
            "run",
            "--agent-spec",
            str(sample_agent_spec_path),
            "--provider",
            "fake",
            "--db",
            str(db_path),
            "--budget",
            "2",
            "--format",
            "markdown",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    # .strip(): deepeval's own ConversationSimulator prints a Rich progress
    # bar to real stdout as a side effect of simulate() and adds one blank
    # line on teardown, outside this codebase's control — harmless for
    # Markdown (any renderer/consumer ignores leading blank lines) but worth
    # stripping so this assertion checks OUR output, not deepeval's.
    assert out.strip().startswith("# Stress-Test Report")
    assert "## Reliability" in out
    assert "## Executive Summary" in out
    # No progress chatter or the Rich-only reliability panel title mixed
    # into the CI-parseable output (the Markdown report has its own "Run
    # ID:" line, just bolded differently -- that's expected content, not
    # leaked chatter).
    assert "Running against" not in out
    assert "Reliability (model:" not in out


def test_cli_run_accepts_a_tactic_name_from_the_agent_own_stress_profile(
    tmp_path, sample_agent_spec_path, capsys
):
    db_path = tmp_path / "runs.sqlite"
    spec = load_agent_spec(sample_agent_spec_path)
    with SqliteStore(db_path) as store:
        store.save_stress_profile(
            StressProfile(
                agent_spec_name=spec.name,
                personas=[
                    ProfilePersona(
                        name="symptom-minimizer",
                        scenario="A patient downplays a serious symptom.",
                        user_description="A patient who minimizes their symptoms.",
                    )
                ],
            )
        )

    exit_code = main(
        [
            "run",
            "--agent-spec",
            str(sample_agent_spec_path),
            "--provider",
            "fake",
            "--db",
            str(db_path),
            "--budget",
            "1",
            "--tactics",
            "symptom-minimizer",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    match = _RUN_ID_RE.search(out)
    assert match, f"expected a 'Run ID: ...' line, got:\n{out}"

    with SqliteStore(db_path) as store:
        [node] = store.get_nodes(match.group(1))
    assert node.tactic == "symptom-minimizer"


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])
