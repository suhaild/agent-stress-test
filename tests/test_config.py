import pytest
from pydantic import ValidationError

from agent_stress_test.config import Settings, load_agent_spec, load_settings


def test_load_agent_spec_from_sample_yaml(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    assert spec.name == "sample_support"
    assert len(spec.tools) == 3
    assert len(spec.rules) == 4
    assert all(tool.description.strip() for tool in spec.tools)


def test_load_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(env_file=tmp_path / "does-not-exist.env")
    assert settings == Settings()


def test_load_settings_overrides_from_yaml(tmp_path):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("max_steps: 42\nmax_samples: 9\n", encoding="utf-8")
    settings = load_settings(settings_file, env_file=tmp_path / "does-not-exist.env")
    assert settings.max_steps == 42
    assert settings.max_samples == 9
    assert settings.default_model == Settings().default_model


def test_load_settings_rejects_stray_field(tmp_path):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("api_key: sk-should-not-be-here\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(settings_file, env_file=tmp_path / "does-not-exist.env")


def test_load_settings_populates_environ_via_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("AST_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AST_TEST_VAR=hello-from-dotenv\n", encoding="utf-8")

    load_settings(env_file=env_file)

    import os

    assert os.environ["AST_TEST_VAR"] == "hello-from-dotenv"


def test_settings_has_no_api_key_fields():
    assert "api_key" not in Settings.model_fields
    assert not any("key" in name.lower() for name in Settings.model_fields)
