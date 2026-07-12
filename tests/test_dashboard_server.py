import json
import re
import time
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from agent_stress_test.config import load_agent_spec
from agent_stress_test.models import Cluster, Message, Node, Run, Verdict
from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.report.dashboard.server import _diff_blocks, create_app
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
    # state (see server.py's `_run_events`).
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

    blocks = _diff_blocks(old, new)

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

    blocks = _diff_blocks(old, new)

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

    blocks = _diff_blocks(old, new)

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
        "agent_stress_test.report.dashboard.server._build_provider", lambda name: scripted
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
