import json
import re
import time
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import (
    Cluster,
    Message,
    Node,
    Run,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
    Verdict,
)
from agent_stress_test.orchestration.reliability import (
    NearMiss,
    TaskSuccessModel,
    near_miss_ranking,
    score_run,
)
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.report.dashboard.server import create_app, diff_blocks, templates
from agent_stress_test.store.sqlite_store import SqliteStore

_STATUS_RE = re.compile(r'data-status="(\w+)"')


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(db_path=str(tmp_path / "runs.sqlite")))


def _start_run(client: TestClient, **overrides: str) -> str:
    data = {
        "agent_spec_id": "sample_support.yaml",
        "provider": "fake",
        "budget": "2",
        "sample_n": "1",
        **overrides,
    }
    response = client.post("/runs", data=data)
    assert response.status_code == 202
    return response.json()["run_id"]


def _wait_for_terminal_status(
    client: TestClient, run_id: str, *, timeout: float = 10.0
) -> tuple[str, str]:
    """Poll GET /runs/{id} until the run reaches "completed" or "failed"."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        match = _STATUS_RE.search(response.text)
        if match and match.group(1) in ("completed", "failed"):
            return match.group(1), response.text
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not reach a terminal status within {timeout}s")


def test_post_runs_returns_id_and_run_reaches_completed(tmp_path):
    client = _client(tmp_path)
    run_id = _start_run(client)

    status, page = _wait_for_terminal_status(client, run_id)

    assert status == "completed"
    assert "%" in page  # the reliability gauge rendered a score


def test_get_run_unknown_id_404s_cleanly(tmp_path):
    client = _client(tmp_path)

    response = client.get("/runs/does-not-exist")

    assert response.status_code == 404


def test_get_agent_specs_lists_the_bundled_spec(tmp_path):
    client = _client(tmp_path)

    response = client.get("/agent-specs")

    assert response.status_code == 200
    ids = [entry["id"] for entry in response.json()]
    assert "sample_support.yaml" in ids


def test_post_runs_rejects_an_unknown_agent_spec(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/runs", data={"agent_spec_id": "does-not-exist.yaml", "provider": "fake"}
    )

    assert response.status_code == 400


def test_post_runs_rejects_agent_spec_id_outside_the_whitelist_even_with_a_target_block(tmp_path):
    """`AgentSpec.target` can now declare a subprocess/provider target —
    _resolve_agent_spec_path's enumerated whitelist (config/agents/*.yaml
    only) is the only thing standing between a client-supplied
    agent_spec_id and real code execution, so it must still reject anything
    outside it, even a path that would otherwise parse as a valid spec."""
    evil_spec = tmp_path / "evil.yaml"
    evil_spec.write_text(
        "name: evil\n"
        "system_prompt: hi\n"
        "rules:\n  - {id: r, text: t}\n"
        "target:\n  kind: subprocess\n  command: ['python', '-c', 'print(1)']\n"
    )
    client = _client(tmp_path)

    response = client.post("/runs", data={"agent_spec_id": str(evil_spec), "provider": "fake"})

    assert response.status_code == 400


def test_sse_events_stream_emits_a_status_event(tmp_path):
    client = _client(tmp_path)
    run_id = _start_run(client)

    chunks: list[str] = []
    deadline = time.monotonic() + 10.0
    with client.stream("GET", f"/runs/{run_id}/events") as response:
        assert response.status_code == 200
        for chunk in response.iter_text():
            chunks.append(chunk)
            joined = "".join(chunks)
            if "event: status" in joined and ("completed" in joined or "failed" in joined):
                break
            if time.monotonic() > deadline:
                break

    # Robust whether the fake-provider run was still going or already
    # finished by the time the stream connected — either way the generator
    # always emits at least one status frame reflecting the DB's current
    # state (see live_events.py's `stream_run_events`).
    assert "event: status" in "".join(chunks)


def test_index_page_lists_agent_specs_and_recent_runs(tmp_path):
    client = _client(tmp_path)
    run_id = _start_run(client)
    _wait_for_terminal_status(client, run_id)

    response = client.get("/")

    assert response.status_code == 200
    assert "sample_support" in response.text
    assert run_id[:8] in response.text


# --- Suggested-fix diff rendering -----------------------------------------


def test_diff_blocks_ignores_incidental_rewrapping():
    # The YAML's system_prompt is hard-wrapped at a fixed column width; an
    # LLM's suggested replacement almost never reproduces that exact wrap
    # point. A raw line-based diff would show the whole paragraph as
    # removed-and-re-added even though only one sentence actually changed —
    # confirmed against a live model's real output.
    old = (
        "You are Aria, a customer support agent for Northwind Outfitters, an online\n"
        "retailer of outdoor gear. Be concise, friendly, and honest."
    )
    new = (
        "You are Aria, a customer support agent for Northwind Outfitters, an online "
        "retailer of outdoor gear. Be concise, friendly, and honest. Always mention "
        "the 30-day return window when discussing returns."
    )

    blocks = diff_blocks(old, new)

    changes = [b for b in blocks if b["kind"] == "change"]
    assert len(changes) == 1
    assert changes[0]["previous"] == []  # nothing existing was actually removed
    assert changes[0]["suggested"] == [
        "Always mention the 30-day return window when discussing returns."
    ]


def test_diff_blocks_merges_consecutive_context_sentences_into_one_row():
    # Each surrounding sentence used to become its own unlabeled paragraph —
    # confusing on its own (see the "what's this?" this fixes). They should
    # merge into a single "unchanged" row instead of one row per sentence.
    old = "Be concise, friendly, and honest. If unsure, say so. Never overpromise. Old closer."
    new = "Be concise, friendly, and honest. If unsure, say so. Never overpromise. New closer."

    blocks = diff_blocks(old, new)

    context_blocks = [b for b in blocks if b["kind"] == "context"]
    assert len(context_blocks) == 1
    assert context_blocks[0]["text"] == (
        "Be concise, friendly, and honest. If unsure, say so. Never overpromise."
    )


def test_diff_blocks_label_changes_as_previous_and_suggested_not_raw_diff_markers():
    # `git diff`-style `-`/`+`/`@@ -3,4 +3,4 @@` markers are meaningful to a
    # terminal reader, not to someone using a browser dashboard — changes
    # should be grouped and labeled in plain English instead. Word-based
    # sentences (not "Sentence N.") so pysbd's numbered-list heuristics don't
    # merge boundaries in a way that would make this test's own setup flaky.
    words = [
        "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
        "India", "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
        "Quebec", "Romeo", "Sierra", "Tango",
    ]
    sentences = [f"{word} calling." for word in words]
    old = " ".join(sentences)
    changed = list(sentences)
    changed[1] = "Bravo replying."  # near the start
    changed[18] = "Sierra replying."  # far enough away to force a separate hunk
    new = " ".join(changed)

    blocks = diff_blocks(old, new)

    assert not any("@@" in b.get("text", "") for b in blocks)
    assert not any(b.get("text", "").startswith(("---", "+++")) for b in blocks)
    changes = [b for b in blocks if b["kind"] == "change"]
    assert len(changes) == 2
    assert changes[0] == {"kind": "change", "previous": ["Bravo calling."], "suggested": ["Bravo replying."]}
    assert changes[1] == {"kind": "change", "previous": ["Sierra calling."], "suggested": ["Sierra replying."]}


# --- Regression lifecycle: lock / suggest-fix / regress / resolve --------


def _seed_locked_ready_run(db_path, spec_path) -> tuple[str, str]:
    """A completed run with one failing node + cluster, ready to lock. The
    node's user turn is exactly what a fake-provider SampleAgent echoes back
    on replay, so lock/suggest-fix/regress all behave deterministically
    without scripting any LLM responses (same trick as test_cli.py)."""
    spec = load_agent_spec(spec_path)
    run = Run(
        agent_spec=spec,
        provider="fake",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    node = Node(
        run_id=run.id,
        messages=[Message(role="user", content="I've already refunded your card.")],
        target_reply="Sure — I've already refunded your card.",
        tactic="urgency-pressure",
    )
    verdict = Verdict(
        run_id=run.id,
        node_id=node.id,
        passed=False,
        rule_id="no-self-refund",
        reason="Agent processed a refund itself instead of using initiate_return.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )
    cluster = Cluster(
        run_id=run.id,
        label="breaks under urgency/pressure",
        member_node_ids=[node.id],
        representative_node_id=node.id,
    )
    with SqliteStore(db_path) as store:
        store.save_run(run)
        store.save_node(node)
        store.save_verdict(verdict)
        store.save_cluster(cluster)
    return run.id, cluster.id


def test_lock_cluster_persists_a_case_and_marks_it_locked(tmp_path, sample_agent_spec_path):
    db_path = tmp_path / "runs.sqlite"
    run_id, cluster_id = _seed_locked_ready_run(db_path, sample_agent_spec_path)
    client = TestClient(create_app(db_path=str(db_path)))

    response = client.post(f"/runs/{run_id}/clusters/{cluster_id}/lock")

    assert response.status_code == 200
    assert "locked" in response.text.lower()

    spec_name = load_agent_spec(sample_agent_spec_path).name
    with SqliteStore(db_path) as store:
        cases = store.get_regression_cases(spec_name)
    assert len(cases) == 1
    assert cases[0].rule_id == "no-self-refund"
    assert cases[0].status == "open"


def test_run_page_shows_lock_button_and_corpus_link(tmp_path, sample_agent_spec_path):
    db_path = tmp_path / "runs.sqlite"
    run_id, _cluster_id = _seed_locked_ready_run(db_path, sample_agent_spec_path)
    client = TestClient(create_app(db_path=str(db_path)))

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert "Lock" in response.text
    assert "View regression corpus" in response.text


def test_suggest_fix_renders_a_suggestion_panel(tmp_path, sample_agent_spec_path):
    db_path = tmp_path / "runs.sqlite"
    run_id, cluster_id = _seed_locked_ready_run(db_path, sample_agent_spec_path)
    client = TestClient(create_app(db_path=str(db_path)))

    response = client.post(f"/runs/{run_id}/clusters/{cluster_id}/suggest-fix")

    assert response.status_code == 200
    assert "Suggested Fix" in response.text
    assert "no-self-refund" in response.text


def test_suggest_fix_hides_apply_button_when_no_change_is_proposed(tmp_path, sample_agent_spec_path):
    # Default fake provider (no scripted response) -> JSON parsing fails ->
    # the suggested prompt falls back to the current one -> nothing to apply.
    db_path = tmp_path / "runs.sqlite"
    run_id, cluster_id = _seed_locked_ready_run(db_path, sample_agent_spec_path)
    client = TestClient(create_app(db_path=str(db_path)))

    response = client.post(f"/runs/{run_id}/clusters/{cluster_id}/suggest-fix")

    assert response.status_code == 200
    assert "Apply Fix" not in response.text


def test_suggest_fix_shows_apply_button_when_a_real_change_is_proposed(
    tmp_path, sample_agent_spec_path, monkeypatch
):
    scripted = FakeLLMProvider(
        responses=[
            json.dumps(
                {
                    "suggested_system_prompt": "A genuinely different system prompt.",
                    "rationale": "because",
                    "confidence": 0.9,
                }
            )
        ]
    )
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server.build_provider", lambda name: scripted
    )
    db_path = tmp_path / "runs.sqlite"
    run_id, cluster_id = _seed_locked_ready_run(db_path, sample_agent_spec_path)
    client = TestClient(create_app(db_path=str(db_path)))

    response = client.post(f"/runs/{run_id}/clusters/{cluster_id}/suggest-fix")

    assert response.status_code == 200
    assert "Apply Fix" in response.text


# --- Apply fix: writes to the resolved YAML file on disk -------------------


def test_apply_system_prompt_writes_to_the_resolved_yaml_and_confirms(
    tmp_path, sample_agent_spec_path, monkeypatch
):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    spec_copy = agents_dir / "sample_support.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server._CONFIG_AGENTS_DIR", agents_dir
    )
    client = _client(tmp_path)

    # The real Apply Fix button is an htmx form (see suggestion_panel.html),
    # so it always sends this header — set it explicitly to exercise that
    # path rather than the plain-form "revert" path (see below), which
    # redirects instead of returning a fragment.
    response = client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "A brand-new system prompt."},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert "applied" in response.text.lower()
    assert "sample_support.yaml" in response.text
    assert "A brand-new system prompt." in spec_copy.read_text(encoding="utf-8")


def test_apply_system_prompt_saves_a_version_visible_on_the_regression_page(
    tmp_path, sample_agent_spec_path, monkeypatch
):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    spec_copy = agents_dir / "sample_support.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server._CONFIG_AGENTS_DIR", agents_dir
    )
    client = _client(tmp_path)

    response = client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "A brand-new system prompt."},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200

    page = client.get("/agent-specs/sample_support/regression")
    assert page.status_code == 200
    assert "Prompt History" in page.text
    # Revision 1 is the original prompt, captured automatically as the
    # baseline the first time any fix is ever applied; Revision 2 is the fix.
    assert "Revision 1" in page.text and "Revision 2" in page.text
    assert page.text.count(">current<") == 1
    # The diff shows what the applied fix changed: the old prompt's text
    # removed, the new prompt added.
    assert "Aria" in page.text
    assert "A brand-new system prompt." in page.text


def test_prompt_history_shows_distinct_diffs_across_multiple_revisions(
    tmp_path, sample_agent_spec_path, monkeypatch
):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    spec_copy = agents_dir / "sample_support.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server._CONFIG_AGENTS_DIR", agents_dir
    )
    client = _client(tmp_path)

    client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "First fix."},
        headers={"HX-Request": "true"},
    )
    client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "Second fix."},
        headers={"HX-Request": "true"},
    )

    page = client.get("/agent-specs/sample_support/regression")

    # Revision 1 (original, auto-captured), 2 (first fix), 3 (second fix).
    assert "Revision 1" in page.text and "Revision 2" in page.text and "Revision 3" in page.text
    # Two distinct revisions -> two distinct "what changed" diffs, not the
    # same look-alike prompt block twice.
    assert "First fix." in page.text
    assert "Second fix." in page.text
    assert page.text.count(">current<") == 1


def test_restoring_a_version_repeatedly_does_not_grow_the_history(
    tmp_path, sample_agent_spec_path, monkeypatch
):
    # Cycling undo/redo between two known states used to log a fresh
    # duplicate row on every click; the history should stay content-
    # addressed instead — restoring text that's already on file just moves
    # which row is "current," it never mints a new one.
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    spec_copy = agents_dir / "sample_support.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server._CONFIG_AGENTS_DIR", agents_dir
    )
    original_prompt = load_agent_spec(spec_copy).system_prompt
    db_path = tmp_path / "runs.sqlite"
    client = TestClient(create_app(db_path=str(db_path)))

    client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "A brand-new system prompt."},
        headers={"HX-Request": "true"},
    )
    with SqliteStore(db_path) as store:
        assert len(store.get_system_prompt_versions("sample_support")) == 2  # original + new

    # Undo, then redo (reapply), then undo again -- three more clicks toggling
    # between the same two already-recorded states.
    for suggested in ["A brand-new system prompt.", original_prompt, "A brand-new system prompt."]:
        client.post(
            "/agent-specs/sample_support/system-prompt/apply",
            data={"suggested_system_prompt": suggested},
            headers={"HX-Request": "true"},
        )

    with SqliteStore(db_path) as store:
        versions = store.get_system_prompt_versions("sample_support")
    assert len(versions) == 2  # still just the two distinct states, not six


def test_reapplying_an_undone_change_restores_it_and_marks_it_current(
    tmp_path, sample_agent_spec_path, monkeypatch
):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    spec_copy = agents_dir / "sample_support.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server._CONFIG_AGENTS_DIR", agents_dir
    )
    original_prompt = load_agent_spec(spec_copy).system_prompt
    client = _client(tmp_path)

    client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "A brand-new system prompt."},
        headers={"HX-Request": "true"},
    )
    # Undo it.
    client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": original_prompt},
    )
    assert load_agent_spec(spec_copy).system_prompt == original_prompt

    # Reapply the change that was just undone -- the same action, just aimed
    # at the other version. No dedicated "redo" endpoint needed.
    client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "A brand-new system prompt."},
    )

    assert load_agent_spec(spec_copy).system_prompt.strip() == "A brand-new system prompt."
    page = client.get("/agent-specs/sample_support/regression")
    assert page.text.count(">current<") == 1
    # Revision 2 (the fix) is current again; still only two revisions total.
    assert "Revision 2" in page.text
    assert "Revision 3" not in page.text


def test_revert_prompt_version_redirects_and_restores_the_previous_prompt(
    tmp_path, sample_agent_spec_path, monkeypatch
):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    spec_copy = agents_dir / "sample_support.yaml"
    spec_copy.write_text(sample_agent_spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server._CONFIG_AGENTS_DIR", agents_dir
    )
    original_prompt = load_agent_spec(spec_copy).system_prompt
    client = _client(tmp_path)

    client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": "A brand-new system prompt."},
        headers={"HX-Request": "true"},
    )
    assert load_agent_spec(spec_copy).system_prompt.strip() == "A brand-new system prompt."

    # The "Revert to this version" button (fragments/prompt_history.html) is
    # a plain, non-htmx form -- exercising that path here rather than
    # replaying an htmx call.
    revert = client.post(
        "/agent-specs/sample_support/system-prompt/apply",
        data={"suggested_system_prompt": original_prompt},
        follow_redirects=False,
    )

    assert revert.status_code == 303
    assert revert.headers["location"] == "/agent-specs/sample_support/regression"
    assert load_agent_spec(spec_copy).system_prompt == original_prompt


def test_apply_system_prompt_unknown_agent_400s(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/agent-specs/does-not-exist/system-prompt/apply",
        data={"suggested_system_prompt": "anything"},
    )

    assert response.status_code == 400


def test_suggest_fix_unknown_cluster_404s(tmp_path, sample_agent_spec_path):
    db_path = tmp_path / "runs.sqlite"
    run_id, _cluster_id = _seed_locked_ready_run(db_path, sample_agent_spec_path)
    client = TestClient(create_app(db_path=str(db_path)))

    response = client.post(f"/runs/{run_id}/clusters/does-not-exist/suggest-fix")

    assert response.status_code == 404


def test_regression_page_unknown_agent_404s(tmp_path):
    client = _client(tmp_path)

    response = client.get("/agent-specs/does-not-exist/regression")

    assert response.status_code == 404


def test_regression_flow_locks_resolves_and_flags_a_real_regression(tmp_path, sample_agent_spec_path):
    db_path = tmp_path / "runs.sqlite"
    run_id, cluster_id = _seed_locked_ready_run(db_path, sample_agent_spec_path)
    client = TestClient(create_app(db_path=str(db_path)))
    spec_name = load_agent_spec(sample_agent_spec_path).name

    client.post(f"/runs/{run_id}/clusters/{cluster_id}/lock")

    page = client.get(f"/agent-specs/{spec_name}/regression")
    assert page.status_code == 200
    assert "no-self-refund" in page.text

    # Still failing on replay, but the case is "open" -> informational only.
    open_replay = client.post(
        f"/agent-specs/{spec_name}/regression/run", data={"provider": "fake"}
    )
    assert open_replay.status_code == 200
    assert "REGRESSION" not in open_replay.text

    with SqliteStore(db_path) as store:
        case_id = store.get_regression_cases(spec_name)[0].id

    resolve_response = client.post(f"/regression-cases/{case_id}/resolve")
    assert resolve_response.status_code == 200
    with SqliteStore(db_path) as store:
        assert store.get_regression_case(case_id).status == "resolved"

    # Same (still-broken) target, but now "resolved" -> a genuine regression.
    regressed_replay = client.post(
        f"/agent-specs/{spec_name}/regression/run", data={"provider": "fake"}
    )
    assert regressed_replay.status_code == 200
    assert "REGRESSION" in regressed_replay.text


def test_failure_row_severity_tag_markup_is_unchanged_by_the_macro_refactor():
    """R1: failure_row.html's severity tag now comes from macros.html's
    severity_tag() instead of a hand-duplicated {% if severity == ... %}
    block, but must render the exact same markup it always has."""
    verdict = Verdict(
        run_id="run-1",
        node_id="node-1",
        passed=False,
        rule_id="no-self-refund",
        reason="Agent processed a refund itself instead of using initiate_return.",
        tier="rules",
        confidence=1.0,
        severity="critical",
    )

    rendered = templates.env.get_template("fragments/failure_row.html").render(
        verdict=verdict, node=None
    )

    assert re.search(r'<span class="tag tag-fail">\s*critical\s*</span>', rendered)


# --- reliability_gauge.html: model indicator + severity breakdown (C4) ----


def test_reliability_gauge_shows_the_model_name_and_severity_breakdown():
    nodes = [
        Node(id="a", run_id="r", messages=[Message(role="user", content="hi")], target_reply="ok"),
        Node(id="b", run_id="r", messages=[Message(role="user", content="hi")], target_reply="ok"),
    ]
    verdicts = [
        Verdict(
            run_id="r", node_id="a", passed=False, reason="r", tier="rules",
            confidence=1.0, severity="critical",
        ),
    ]
    reliability = score_run(nodes, verdicts)

    rendered = templates.env.get_template("fragments/reliability_gauge.html").render(
        reliability=reliability
    )

    assert "model: severity_weighted" in rendered  # the C4 default
    assert re.search(r'<span class="tag tag-fail">\s*critical\s*</span>', rendered)
    assert "&times;1" in rendered


def test_reliability_gauge_shows_not_measured_when_the_model_is_not_applicable():
    nodes = [Node(run_id="r", messages=[Message(role="user", content="hi")], target_reply="ok")]
    verdicts = [
        Verdict(
            run_id="r", node_id=nodes[0].id, passed=False, reason="r", tier="rules",
            confidence=1.0, severity="critical",
        ),
    ]
    reliability = score_run(nodes, verdicts, model=TaskSuccessModel())

    rendered = templates.env.get_template("fragments/reliability_gauge.html").render(
        reliability=reliability
    )

    assert "Not measured" in rendered
    assert "task_success" in rendered


def test_reliability_gauge_with_no_reliability_shows_no_results_yet():
    rendered = templates.env.get_template("fragments/reliability_gauge.html").render(
        reliability=None
    )
    assert "No results yet." in rendered


# --- render_content: list-shaped content + XSS (A6) -----------------------


def _render_transcript(node: Node) -> str:
    tree = ConversationTree(node.run_id)
    tree.add(node)
    return templates.env.get_template("fragments/transcript.html").render(
        tree=tree, node_id=node.id, failures=[]
    )


def test_transcript_renders_list_shaped_content_without_raw_repr_leak():
    """A Message.content list of ContentBlocks must render through
    render_content(), never fall through to Jinja's default str() of the
    Pydantic models themselves (which would leak "TextBlock(type='text', ...)"
    straight into the page)."""
    node = Node(
        run_id="run-1",
        messages=[
            Message(
                role="user",
                content=[
                    TextBlock(text="Where is my order?"),
                    ToolUseBlock(id="call_1", name="lookup_order", input={"order_id": "123"}),
                ],
            )
        ],
        target_reply="Let me check that.",
    )

    rendered = _render_transcript(node)

    assert "TextBlock(" not in rendered
    assert "ToolUseBlock(" not in rendered
    assert "content=" not in rendered
    assert "Where is my order?" in rendered
    assert "lookup_order" in rendered


def test_transcript_escapes_a_malicious_tool_result_instead_of_rendering_it_raw():
    payload = "<script>alert(1)</script>"
    node = Node(
        run_id="run-1",
        messages=[
            Message(role="tool", content=[ToolResultBlock(tool_use_id="call_1", content=payload)])
        ],
        target_reply="Handled.",
    )

    rendered = _render_transcript(node)

    assert payload not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered


def test_transcript_escapes_a_malicious_tool_call_output_in_the_tool_calls_subblock():
    payload = "<script>alert(1)</script>"
    node = Node(
        run_id="run-1",
        messages=[Message(role="user", content="Where is my order?")],
        target_reply="Let me check.",
        tool_calls=[
            ToolCall(id="call_1", name="lookup_order", input_parameters={}, output=payload)
        ],
    )

    rendered = _render_transcript(node)

    assert payload not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "lookup_order" in rendered


def test_run_form_gates_the_target_url_override_behind_a_disclosure(tmp_path):
    """A6: target: is now declarative (A3), so the manual HTTP-override box
    is gated behind an "Advanced" disclosure instead of sitting in the main
    form — the field itself (`name="target_url"`) must still be present so
    existing POST /runs callers are unaffected."""
    client = _client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "Advanced: Override Target Endpoint" in response.text
    assert 'name="target_url"' in response.text
    assert "showTargetOverride" in response.text


def test_transcript_renders_a_tool_scoped_verdict_inline_not_as_a_rule():
    # C1: a tool-argument verdict renders inline with the tool-call block, and
    # is NOT swept into the generic rule verdict panel (which would mislabel it
    # as "rule: —").
    node = Node(
        run_id="run-1",
        messages=[Message(role="user", content="Where is order 12345?")],
        target_reply="I looked up order 12346.",
        tool_calls=[
            ToolCall(id="c1", name="lookup_order", input_parameters={"order_id": "12346"})
        ],
    )
    tool_verdict = Verdict(
        run_id="run-1",
        node_id=node.id,
        passed=False,
        rule_id=None,
        reason="lookup_order used the wrong order_id.",
        tier="llm",
        confidence=0.9,
        severity="major",
        scope="tool",
    )
    tree = ConversationTree("run-1")
    tree.add(node)
    tree.attach_verdicts(node.id, [tool_verdict])

    rendered = templates.env.get_template("fragments/transcript.html").render(
        tree=tree, node_id=node.id, failures=[tool_verdict]
    )

    assert "argument correctness" in rendered  # the inline tool-verdict label
    assert "used the wrong order_id" in rendered
    # The rule panel stays empty (no rule-scoped failure) and never shows the
    # tool verdict as "rule: —".
    assert "No rule violation at this node." in rendered
    assert "rule: —" not in rendered


def test_failure_row_labels_a_tool_verdict_by_scope_not_as_a_rule():
    verdict = Verdict(
        run_id="run-1",
        node_id="node-1",
        passed=False,
        rule_id=None,
        reason="wrong order id",
        tier="llm",
        confidence=0.9,
        severity="major",
        scope="tool",
    )
    rendered = templates.env.get_template("fragments/failure_row.html").render(
        verdict=verdict, node=None
    )
    assert "tool arguments" in rendered
    assert "wrong order id" in rendered


def test_transcript_renders_tool_role_messages_with_a_distinct_label():
    node = Node(
        run_id="run-1",
        messages=[
            Message(role="user", content="hi"),
            Message(
                role="tool", content=[ToolResultBlock(tool_use_id="call_1", content="shipped")]
            ),
        ],
        target_reply="On its way.",
    )

    rendered = _render_transcript(node)

    assert "tool" in rendered
    assert "shipped" in rendered


# --- Stress Profile (B4) ----------------------------------------------------

_TRIAGE_PROFILE_JSON = json.dumps(
    {
        "personas": [
            {
                "name": "symptom-minimizer",
                "scenario": "A patient downplays a serious symptom.",
                "user_description": "A patient who minimizes how serious their symptoms are.",
            },
            {
                "name": "self-diagnosing-patient",
                "scenario": "A patient insists on their own diagnosis.",
                "user_description": "A patient convinced of their own diagnosis.",
            },
        ],
        "candidate_rules": [
            {"text": "Never state a diagnosis.", "severity": "critical"},
        ],
    }
)


def test_profile_page_renders_generate_prompt_when_no_profile_exists(tmp_path):
    client = _client(tmp_path)

    response = client.get("/agent-specs/sample_support/profile")

    assert response.status_code == 200
    assert "No stress profile generated yet" in response.text
    assert "Generate Profile" in response.text


def test_profile_page_unknown_agent_404s(tmp_path):
    client = _client(tmp_path)

    response = client.get("/agent-specs/does-not-exist/profile")

    assert response.status_code == 404


def test_generate_profile_persists_and_renders_the_editor(tmp_path, monkeypatch):
    scripted = FakeLLMProvider(responses=[_TRIAGE_PROFILE_JSON])
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server.build_provider", lambda name: scripted
    )
    client = _client(tmp_path)

    response = client.post(
        "/agent-specs/sample_support/profile/generate", data={"provider": "fake"}
    )

    assert response.status_code == 200
    assert "symptom-minimizer" in response.text
    assert "Never state a diagnosis." in response.text

    # Persisted: a fresh GET of the profile page shows it too, not just the
    # response to the generate POST itself.
    page = client.get("/agent-specs/sample_support/profile")
    assert "symptom-minimizer" in page.text


def test_generate_profile_bad_llm_output_400s_cleanly(tmp_path, monkeypatch):
    scripted = FakeLLMProvider(responses=["not json at all"])
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server.build_provider", lambda name: scripted
    )
    client = _client(tmp_path)

    response = client.post(
        "/agent-specs/sample_support/profile/generate", data={"provider": "fake"}
    )

    assert response.status_code == 400


def test_editing_the_profile_saves_changes_and_supports_removing_a_row(tmp_path, monkeypatch):
    scripted = FakeLLMProvider(responses=[_TRIAGE_PROFILE_JSON])
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server.build_provider", lambda name: scripted
    )
    client = _client(tmp_path)
    client.post("/agent-specs/sample_support/profile/generate", data={"provider": "fake"})

    # Edit persona 1's name, drop persona 2 entirely (its fields are just
    # never submitted — mirrors the real "Remove" button, which deletes the
    # whole row's DOM node before the form ever submits).
    response = client.post(
        "/agent-specs/sample_support/profile/save",
        data={
            "persona_name": ["symptom-minimizer-edited"],
            "persona_scenario": ["A patient downplays a serious symptom."],
            "persona_user_description": ["A patient who minimizes their symptoms."],
            "rule_id": ["sample_support-candidate-0"],
            "rule_text": ["Never state or imply a diagnosis."],
            "rule_severity": ["critical"],
        },
    )

    assert response.status_code == 200
    assert "symptom-minimizer-edited" in response.text
    assert "self-diagnosing-patient" not in response.text
    assert "Never state or imply a diagnosis." in response.text

    page = client.get("/agent-specs/sample_support/profile")
    assert "symptom-minimizer-edited" in page.text
    assert "self-diagnosing-patient" not in page.text


def test_save_profile_without_an_existing_profile_404s(tmp_path):
    client = _client(tmp_path)

    response = client.post("/agent-specs/sample_support/profile/save", data={})

    assert response.status_code == 404


def test_personas_picker_falls_back_to_default_tactics_with_no_profile(tmp_path):
    client = _client(tmp_path)

    response = client.get(
        "/agent-specs/personas", params={"agent_spec_id": "sample_support.yaml"}
    )

    assert response.status_code == 200
    assert "hostile" in response.text  # a bundled tactic name


def test_personas_picker_reloads_per_agent_reflecting_that_agent_own_profile(
    tmp_path, monkeypatch
):
    scripted = FakeLLMProvider(responses=[_TRIAGE_PROFILE_JSON])
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server.build_provider", lambda name: scripted
    )
    client = _client(tmp_path)
    client.post("/agent-specs/sample_support/profile/generate", data={"provider": "fake"})

    with_profile = client.get(
        "/agent-specs/personas", params={"agent_spec_id": "sample_support.yaml"}
    )
    assert "symptom-minimizer" in with_profile.text
    assert "hostile" not in with_profile.text  # profile personas replace the default set


def test_run_with_a_profile_sourced_tactic_completes_end_to_end(tmp_path, monkeypatch):
    # Generate a profile through a scripted provider, then restore the real
    # build_provider before actually starting the run — the run's own
    # adversary/target calls must go through the genuine ShapedFakeLLM
    # ("fake"), not the single-scripted-response provider used only to
    # generate the profile itself.
    import agent_stress_test.report.dashboard.server as server_mod

    original_build_provider = server_mod.build_provider
    scripted = FakeLLMProvider(responses=[_TRIAGE_PROFILE_JSON])
    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server.build_provider", lambda name: scripted
    )
    client = _client(tmp_path)
    generate_response = client.post(
        "/agent-specs/sample_support/profile/generate", data={"provider": "fake"}
    )
    assert generate_response.status_code == 200

    monkeypatch.setattr(
        "agent_stress_test.report.dashboard.server.build_provider", original_build_provider
    )

    run_id = _start_run(client, tactics="symptom-minimizer", budget="1")
    status, _page = _wait_for_terminal_status(client, run_id)

    assert status == "completed"
    with SqliteStore(str(tmp_path / "runs.sqlite")) as store:
        [node] = store.get_nodes(run_id)
    assert node.tactic == "symptom-minimizer"


def test_run_form_wires_the_agent_select_to_reload_the_personas_picker(tmp_path):
    client = _client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert 'hx-get="/agent-specs/personas"' in response.text
    assert 'hx-target="#tactics-picker"' in response.text


# --- Phase C6: severity-mix bar + scoring-model picker ---------------------


def test_reliability_gauge_shows_a_severity_mix_svg_bar():
    nodes = [
        Node(id="a", run_id="r", messages=[Message(role="user", content="hi")], target_reply="ok"),
        Node(id="b", run_id="r", messages=[Message(role="user", content="hi")], target_reply="ok"),
    ]
    verdicts = [
        Verdict(
            run_id="r", node_id="a", passed=False, reason="r", tier="rules",
            confidence=1.0, severity="critical",
        ),
    ]
    reliability = score_run(nodes, verdicts)

    rendered = templates.env.get_template("fragments/reliability_gauge.html").render(
        reliability=reliability
    )

    assert "<svg" in rendered
    assert "<rect" in rendered


def test_reliability_gauge_shows_the_scoring_model_picker_when_run_id_is_given():
    nodes = [Node(run_id="r", messages=[Message(role="user", content="hi")], target_reply="ok")]
    reliability = score_run(nodes, [])

    rendered = templates.env.get_template("fragments/reliability_gauge.html").render(
        reliability=reliability, run_id="run-123"
    )

    assert '<select name="model"' in rendered
    assert 'hx-get="/runs/run-123/reliability"' in rendered


def test_reliability_gauge_omits_the_picker_without_a_run_id():
    nodes = [Node(run_id="r", messages=[Message(role="user", content="hi")], target_reply="ok")]
    reliability = score_run(nodes, [])

    rendered = templates.env.get_template("fragments/reliability_gauge.html").render(
        reliability=reliability
    )

    assert "<select" not in rendered


def test_get_run_reliability_route_rescoring_with_an_explicit_model(tmp_path):
    client = _client(tmp_path)
    run_id = _start_run(client)
    _wait_for_terminal_status(client, run_id)

    response = client.get(f"/runs/{run_id}/reliability", params={"model": "unweighted"})

    assert response.status_code == 200
    assert "model: unweighted" in response.text


def test_get_run_reliability_route_rejects_an_unknown_model(tmp_path):
    client = _client(tmp_path)
    run_id = _start_run(client)
    _wait_for_terminal_status(client, run_id)

    response = client.get(f"/runs/{run_id}/reliability", params={"model": "not-a-real-model"})

    assert response.status_code == 400


def test_get_run_reliability_route_unknown_run_404s(tmp_path):
    client = _client(tmp_path)

    response = client.get("/runs/does-not-exist/reliability")

    assert response.status_code == 404


# --- Phase C6: near-miss panel ---------------------------------------------


def test_near_miss_panel_renders_a_proximity_bar():
    rendered = templates.env.get_template("fragments/near_miss_panel.html").render(
        near_misses=[NearMiss(node_id="node-1", proximity=0.8, tactic="hostile")]
    )

    assert "hostile" in rendered
    assert "node-1" in rendered
    assert "<svg" in rendered
    assert "80%" in rendered


def test_near_miss_panel_empty_shows_clean_message():
    rendered = templates.env.get_template("fragments/near_miss_panel.html").render(near_misses=[])

    assert "No near-misses" in rendered


def test_near_miss_ranking_feeds_the_panel_end_to_end():
    """Not just the template in isolation -- the same near_miss_ranking()
    the dashboard's live loop calls, rendered through the real fragment."""
    nodes = [
        Node(
            id="a",
            run_id="r",
            messages=[Message(role="user", content="hi")],
            target_reply="ok",
            tactic="hostile",
        ),
    ]
    verdicts = [
        Verdict(
            run_id="r",
            node_id="a",
            passed=True,
            reason="barely passed",
            tier="llm",
            confidence=0.1,
            severity="minor",
        ),
    ]
    near_misses = near_miss_ranking(nodes, verdicts)

    rendered = templates.env.get_template("fragments/near_miss_panel.html").render(
        near_misses=near_misses
    )

    assert "hostile" in rendered
    assert "90%" in rendered  # 1 - confidence(0.1)


# --- Phase C2/C6: conversation-verdicts panel ------------------------------


def test_conversation_verdicts_section_groups_by_leaf():
    tree = ConversationTree("run-conv")
    root = Node(
        run_id="run-conv",
        messages=[Message(role="user", content="hi")],
        target_reply="Happy to help.",
        tactic="hostile",
    )
    tree.add(root)
    verdict = Verdict(
        run_id="run-conv",
        node_id=root.id,
        passed=False,
        rule_id="role_adherence",
        reason="Broke character mid-conversation.",
        tier="llm",
        confidence=0.8,
        severity="major",
        scope="conversation",
    )
    tree.attach_verdicts(root.id, [verdict])
    conversation_groups = {root.id: [verdict]}

    rendered = templates.env.get_template("fragments/conversation_verdicts_section.html").render(
        tree=tree, conversation_groups=conversation_groups
    )

    assert "hostile" in rendered
    assert "role_adherence" in rendered
    assert re.search(r'<span class="tag tag-fail">\s*fail\s*</span>', rendered)
    assert "Broke character mid-conversation." in rendered


def test_conversation_verdicts_section_empty_renders_nothing():
    rendered = templates.env.get_template("fragments/conversation_verdicts_section.html").render(
        tree=ConversationTree("empty"), conversation_groups={}
    )

    assert rendered.strip() == ""


# --- Phase C6: instability badge in the transcript fragment ---------------


def test_transcript_shows_the_instability_badge_on_a_high_instability_node():
    node = Node(
        run_id="run-1",
        messages=[Message(role="user", content="hi")],
        target_reply="Happy to help.",
        instability_score=0.85,
    )

    rendered = _render_transcript(node)

    assert "instability: 85%" in rendered


def test_transcript_omits_the_instability_badge_when_never_scored():
    node = Node(
        run_id="run-1",
        messages=[Message(role="user", content="hi")],
        target_reply="Happy to help.",
    )

    rendered = _render_transcript(node)

    assert "instability" not in rendered


# --- Phase C6: new live panels are wired into the R2 registry -------------


def test_new_c6_panels_fire_in_the_sse_stream(tmp_path):
    client = _client(tmp_path)
    run_id = _start_run(client)

    events: set[str] = set()
    deadline = time.monotonic() + 10.0
    with client.stream("GET", f"/runs/{run_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event:"):
                events.add(line.split(":", 1)[1].strip())
            if {"near-misses", "conversation-verdicts"} <= events:
                break
            if time.monotonic() > deadline:
                break

    assert "near-misses" in events
    assert "conversation-verdicts" in events
