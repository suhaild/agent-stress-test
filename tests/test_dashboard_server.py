import re
import time

from fastapi.testclient import TestClient

from agent_stress_test.report.dashboard.server import create_app

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
