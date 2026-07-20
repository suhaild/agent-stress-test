import pytest
from pydantic import ValidationError

from agent_stress_test.config import Settings, load_agent_spec, load_settings
from agent_stress_test.config_writer import (
    _append_rule_block,
    _replace_system_prompt_block,
    apply_candidate_rule,
    apply_system_prompt,
)
from agent_stress_test.models import Rule


def test_load_agent_spec_from_sample_yaml(sample_agent_spec_path):
    spec = load_agent_spec(sample_agent_spec_path)
    assert spec.name == "sample_support_advanced"
    assert len(spec.tools) == 3
    assert len(spec.rules) == 10
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


# --- _replace_system_prompt_block / apply_system_prompt -------------------


def test_replace_system_prompt_block_preserves_everything_else():
    raw = (
        "name: demo\n"
        "\n"
        "system_prompt: |\n"
        "  Old line one.\n"
        "  Old line two.\n"
        "\n"
        "tools: []\n"
        "\n"
        "rules:\n"
        "  - id: r1\n"
        "    text: \"Do the thing.\"  # a comment worth keeping\n"
        "    severity: major\n"
    )

    updated = _replace_system_prompt_block(raw, "New paragraph one.\n\nNew paragraph two.")

    assert updated == (
        "name: demo\n"
        "\n"
        "system_prompt: |\n"
        "  New paragraph one.\n"
        "\n"
        "  New paragraph two.\n"
        "\n"
        "tools: []\n"
        "\n"
        "rules:\n"
        "  - id: r1\n"
        "    text: \"Do the thing.\"  # a comment worth keeping\n"
        "    severity: major\n"
    )


def test_replace_system_prompt_block_raises_without_a_system_prompt_key():
    with pytest.raises(ValueError):
        _replace_system_prompt_block("name: demo\ntools: []\n", "anything")


def test_apply_system_prompt_replaces_the_block_and_preserves_everything_else(
    sample_agent_spec_path, tmp_path
):
    spec_copy = tmp_path / "sample_support_advanced.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    original_text = spec_copy.read_text(encoding="utf-8")

    new_spec = apply_system_prompt(
        spec_copy, "A brand-new system prompt.\n\nWith a second paragraph."
    )

    # YAML's `|` block scalar keeps exactly one trailing newline (clip
    # chomping) — same as the pre-existing system_prompt already did.
    assert new_spec.system_prompt == "A brand-new system prompt.\n\nWith a second paragraph.\n"
    updated_text = spec_copy.read_text(encoding="utf-8")
    # Everything from `tools:` onward — including the hand-written comment
    # explaining the mention-return-window regex — is untouched.
    assert original_text.split("tools:", 1)[1] == updated_text.split("tools:", 1)[1]


def test_apply_system_prompt_rolls_back_on_invalid_result(sample_agent_spec_path, tmp_path):
    spec_copy = tmp_path / "sample_support_advanced.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    original_text = spec_copy.read_text(encoding="utf-8")

    with pytest.raises(ValidationError):
        apply_system_prompt(spec_copy, "")  # empty prompt violates AgentSpec's min_length=1

    assert spec_copy.read_text(encoding="utf-8") == original_text


# --- _append_rule_block / apply_candidate_rule -----------------------------


def test_append_rule_block_preserves_everything_else_and_appends_at_the_end():
    raw = (
        "name: demo\n"
        "\n"
        "system_prompt: |\n"
        "  Hello.\n"
        "\n"
        "rules:\n"
        "  - id: r1\n"
        '    text: "Do the thing."  # a comment worth keeping\n'
        "    severity: major\n"
        "\n"
        "  - id: r2\n"
        '    text: "Do another thing."\n'
        "    severity: minor\n"
    )

    updated = _append_rule_block(raw, Rule(id="r3", text="Do a third thing.", severity="critical"))

    assert updated == (
        "name: demo\n"
        "\n"
        "system_prompt: |\n"
        "  Hello.\n"
        "\n"
        "rules:\n"
        "  - id: r1\n"
        '    text: "Do the thing."  # a comment worth keeping\n'
        "    severity: major\n"
        "\n"
        "  - id: r2\n"
        '    text: "Do another thing."\n'
        "    severity: minor\n"
        "\n"
        "  - id: r3\n"
        "    text: Do a third thing.\n"
        "    severity: critical\n"
    )


def test_append_rule_block_appends_before_a_later_top_level_key():
    raw = "name: demo\n\nrules:\n  - id: r1\n    text: t\n    severity: major\n\ntarget:\n  kind: http\n  url: x\n"

    updated = _append_rule_block(raw, Rule(id="r2", text="t2", severity="minor"))

    assert updated == (
        "name: demo\n\nrules:\n  - id: r1\n    text: t\n    severity: major\n\n"
        "  - id: r2\n    text: t2\n    severity: minor\n\ntarget:\n  kind: http\n  url: x\n"
    )


def test_append_rule_block_raises_without_a_rules_key():
    with pytest.raises(ValueError):
        _append_rule_block("name: demo\ntools: []\n", Rule(id="r", text="t", severity="major"))


def test_append_rule_block_yaml_escapes_special_characters_in_rule_text():
    raw = "rules:\n  - id: r1\n    text: t\n    severity: major\n"
    tricky = Rule(id="r2", text="Never say: 'yes' — even with a colon.", severity="major")

    updated = _append_rule_block(raw, tricky)

    import yaml

    reloaded = yaml.safe_load(updated)
    assert reloaded["rules"][1]["text"] == tricky.text


def test_apply_candidate_rule_appends_and_preserves_everything_else(
    sample_agent_spec_path, tmp_path
):
    spec_copy = tmp_path / "sample_support_advanced.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    original_text = spec_copy.read_text(encoding="utf-8")
    new_rule = Rule(
        id="no-carrier-invention", text="Never invent a shipping carrier.", severity="major"
    )

    new_spec = apply_candidate_rule(spec_copy, new_rule)

    assert new_spec.rules[-1].id == "no-carrier-invention"
    assert len(new_spec.rules) == len(load_agent_spec(sample_agent_spec_path).rules) + 1
    updated_text = spec_copy.read_text(encoding="utf-8")
    assert original_text.split("system_prompt:", 1)[0] == updated_text.split("system_prompt:", 1)[0]
    assert updated_text.startswith(original_text.rstrip("\n"))


def test_apply_candidate_rule_rejects_a_duplicate_id(sample_agent_spec_path, tmp_path):
    spec_copy = tmp_path / "sample_support_advanced.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    original_text = spec_copy.read_text(encoding="utf-8")
    colliding_rule = Rule(id="no-self-refund", text="Something else entirely.", severity="minor")

    with pytest.raises(ValueError, match="already exists"):
        apply_candidate_rule(spec_copy, colliding_rule)

    # Rejected before ever touching the file.
    assert spec_copy.read_text(encoding="utf-8") == original_text
