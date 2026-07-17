import os

# Must be set before anything imports deepeval (it phones home on import
# otherwise), so this runs at the top of the root conftest — pytest loads it
# before collecting any test module.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "1")

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_agent_spec_path() -> Path:
    return REPO_ROOT / "config" / "agents" / "sample_support.yaml"


@pytest.fixture
def sample_settings_path() -> Path:
    return REPO_ROOT / "config" / "settings.example.yaml"
