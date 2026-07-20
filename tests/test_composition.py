from agent_stress_test.composition import reconcile_interrupted_runs
from agent_stress_test.models import Run
from agent_stress_test.store.sqlite_store import SqliteStore
from tests.conftest import make_agent_spec


def _run(run_id: str, status: str) -> Run:
    return Run(id=run_id, agent_spec=make_agent_spec(), provider="fake", status=status)


def test_reconcile_interrupted_runs_fails_out_pending_and_running_rows(tmp_path):
    with SqliteStore(str(tmp_path / "runs.sqlite")) as store:
        store.save_run(_run("r-pending", "pending"))
        store.save_run(_run("r-running", "running"))
        store.save_run(_run("r-completed", "completed"))
        store.save_run(_run("r-failed", "failed"))

        fixed = reconcile_interrupted_runs(store)

        assert fixed == 2
        assert store.get_run("r-pending").status == "failed"
        assert store.get_run("r-running").status == "failed"
        assert "Interrupted" in store.get_run("r-pending").error
        assert store.get_run("r-pending").completed_at is not None
        # Already-finished runs are untouched, including their error field.
        assert store.get_run("r-completed").status == "completed"
        assert store.get_run("r-failed").status == "failed"
        assert store.get_run("r-failed").error is None


def test_reconcile_interrupted_runs_is_a_no_op_when_nothing_is_stuck(tmp_path):
    with SqliteStore(str(tmp_path / "runs.sqlite")) as store:
        store.save_run(_run("r-completed", "completed"))

        fixed = reconcile_interrupted_runs(store)

        assert fixed == 0
        assert store.get_run("r-completed").status == "completed"


def test_reconcile_interrupted_runs_handles_an_empty_store(tmp_path):
    with SqliteStore(str(tmp_path / "runs.sqlite")) as store:
        assert reconcile_interrupted_runs(store) == 0
