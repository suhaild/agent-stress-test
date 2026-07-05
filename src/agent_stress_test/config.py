"""Pydantic settings + YAML loading."""

from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from agent_stress_test.models import AgentSpec


class Settings(BaseModel):
    """Run-time settings. Never holds API keys — those come from the environment."""

    model_config = ConfigDict(extra="forbid")

    default_model: str = "claude-3-5-sonnet-20241022"
    max_steps: int = 20
    max_samples: int = 5


def load_settings(path: str | Path | None = None, *, env_file: str | Path = ".env") -> Settings:
    """Load .env into the environment as a side effect, then build Settings from YAML."""
    load_dotenv(env_file, override=False)
    data = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings(**data)


def load_agent_spec(path: str | Path) -> AgentSpec:
    """Load an AgentSpec from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AgentSpec.model_validate(data)
