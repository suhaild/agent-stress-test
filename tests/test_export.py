import json

from agent_stress_test.report.export import build_report_bundle, to_json, to_json_dict, to_markdown
from tests.conftest import build_and_run


def _bundle(spec_path):
    def target_fn(conversation):
        text = " ".join(m.content for m in conversation if m.role == "user")
        if "[urgency-pressure]" in text:
            return "Sure -- I've already refunded your card."
        return "Happy to help."

    result = build_and_run(spec_path, target_fn, budget=2)
    from agent_stress_test.composition import cluster_and_persist
    from agent_stress_test.store.sqlite_store import SqliteStore

    with SqliteStore() as store:
        clusters = cluster_and_persist(result, store)
    return build_report_bundle(result.run, result.tree, result.tree.all_verdicts(), clusters)


def test_build_report_bundle_reliability_matches_score_run(sample_agent_spec_path):
    bundle = _bundle(sample_agent_spec_path)
    assert bundle.reliability.score >= 0.0
    assert bundle.reliability.total_steps == len(bundle.tree.nodes())


def test_to_json_dict_has_expected_top_level_keys(sample_agent_spec_path):
    bundle = _bundle(sample_agent_spec_path)
    payload = to_json_dict(bundle)
    assert set(payload.keys()) == {
        "run",
        "reliability",
        "clusters",
        "near_misses",
        "conversation_verdicts",
        "rule_coverage",
        "summary",
        "fix_first",
    }
    assert payload["reliability"]["score"] == bundle.reliability.score
    assert payload["run"]["id"] == bundle.run.id


def test_to_json_produces_valid_parseable_json(sample_agent_spec_path):
    bundle = _bundle(sample_agent_spec_path)
    parsed = json.loads(to_json(bundle))
    assert parsed["reliability"]["total_steps"] == bundle.reliability.total_steps
    assert len(parsed["rule_coverage"]) == len(bundle.run.agent_spec.rules)


def test_to_json_is_fully_ascii_safe(sample_agent_spec_path):
    """json.dumps defaults to ensure_ascii=True, so this should hold
    regardless of content -- asserted explicitly since a real Windows
    console previously crashed on literal non-ASCII output (Phase C6's
    Unicode-block-character bug) and this is the same class of risk."""
    bundle = _bundle(sample_agent_spec_path)
    text = to_json(bundle)
    assert all(ord(char) < 128 for char in text)


def test_to_markdown_is_fully_ascii_safe(sample_agent_spec_path):
    bundle = _bundle(sample_agent_spec_path)
    text = to_markdown(bundle)
    assert all(ord(char) < 128 for char in text)


def test_to_markdown_contains_headline_score_and_summary(sample_agent_spec_path):
    bundle = _bundle(sample_agent_spec_path)
    text = to_markdown(bundle)

    assert f"{bundle.reliability.score:.0%}" in text
    assert bundle.summary.text in text
    assert "# Stress-Test Report" in text
    assert "## Rule Coverage" in text
    assert "## Executive Summary" in text


def test_to_markdown_rule_coverage_table_lists_every_declared_rule(sample_agent_spec_path):
    bundle = _bundle(sample_agent_spec_path)
    text = to_markdown(bundle)
    for rule in bundle.run.agent_spec.rules:
        assert rule.id in text


def test_to_markdown_no_failure_clusters_message_when_clean(sample_agent_spec_path):
    bundle = _bundle(sample_agent_spec_path)
    if not bundle.ranked_clusters:
        assert "No failure clusters" in to_markdown(bundle)
