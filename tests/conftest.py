from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_agent_spec_path() -> Path:
    return REPO_ROOT / "config" / "agents" / "sample_support.yaml"


@pytest.fixture
def sample_settings_path() -> Path:
    return REPO_ROOT / "config" / "settings.example.yaml"
