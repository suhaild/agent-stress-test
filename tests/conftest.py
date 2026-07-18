import os

# Must be set before anything imports deepeval (it phones home on import
# otherwise), so this runs at the top of the root conftest — pytest loads it
# before collecting any test module.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "1")

from collections.abc import Callable
from pathlib import Path

import pytest

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import AgentSpec, Message, Rule, ToolSpec
from agent_stress_test.orchestration.runner import RunResult, build_runner
from agent_stress_test.ports import Store
from agent_stress_test.providers.shaped_fake import ShapedFakeLLM
from agent_stress_test.targets.python_fn import PythonFunctionAgent

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_agent_spec_path() -> Path:
    return REPO_ROOT / "config" / "agents" / "sample_support.yaml"


@pytest.fixture
def sample_settings_path() -> Path:
    return REPO_ROOT / "config" / "settings.example.yaml"


def make_agent_spec(**overrides) -> AgentSpec:
    """A minimal, valid AgentSpec for tests that don't care about its exact
    content — override ``tools``/``rules``/anything else when a test does."""
    defaults = dict(
        name="test_agent",
        system_prompt="You are a helpful assistant.",
        tools=[
            ToolSpec(name="lookup_order", description="Look up an order by ID."),
            ToolSpec(name="initiate_return", description="Start a return."),
        ],
        rules=[
            Rule(id="no-invent", text="Never invent data.", severity="major"),
            Rule(id="be-polite", text="Always be polite.", severity="minor"),
        ],
    )
    defaults.update(overrides)
    return AgentSpec(**defaults)


def build_and_run(
    spec_path: Path,
    target_fn: Callable[[list[Message]], str],
    *,
    store: Store | None = None,
    sample_n: int = 1,
    budget: int = 2,
) -> RunResult:
    """Build a runner over a scripted Python-function target and run it once
    — the shared plumbing behind test_store.py's ``run_with_store`` and
    test_orchestration.py's ``run_once``, each of which wraps this with its
    own scripted ``target_fn``."""
    agent_spec = load_agent_spec(spec_path)
    runner = build_runner(
        agent_spec=agent_spec,
        target=PythonFunctionAgent(target_fn),
        sim_provider=ShapedFakeLLM(),
        store=store,
        sample_n=sample_n,
    )
    return runner.run(provider_name="fake", budget=budget)
