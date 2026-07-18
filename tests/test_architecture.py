import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "agent_stress_test"


def _import_offenders(module: str, exempt_path: Path) -> list[str]:
    """``exempt_path`` may be a single file (that file alone is allowed to
    import ``module``) or a directory (every file under it is allowed to)."""
    pattern = re.compile(rf"^\s*(import {module}\b|from {module}\b)", re.MULTILINE)
    offenders = []
    for path in SRC_ROOT.rglob("*.py"):
        if path == exempt_path or exempt_path in path.parents:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(SRC_ROOT)))
    return offenders


def test_litellm_only_imported_in_its_own_adapter():
    offenders = _import_offenders("litellm", SRC_ROOT / "providers" / "litellm_provider.py")

    assert offenders == [], (
        "litellm must only be imported in providers/litellm_provider.py, "
        f"but also found in: {offenders}"
    )


def test_httpx_only_imported_in_its_own_adapter():
    offenders = _import_offenders("httpx", SRC_ROOT / "targets" / "http_agent.py")

    assert offenders == [], (
        f"httpx must only be imported in targets/http_agent.py, but also found in: {offenders}"
    )


def test_deepeval_only_imported_in_the_reasoning_layer():
    # Not one hardcoded file: reasoning/deepeval_bridge.py (the shim),
    # reasoning/deepeval_simulator.py (ConversationSimulator/ConversationalGolden
    # — see orchestration/deepeval_search.py), and reasoning/profiler.py (B4's
    # AgentProfiler, which converts a ProfilePersona to a ConversationalGolden)
    # all legitimately need it.
    offenders = _import_offenders("deepeval", SRC_ROOT / "reasoning")

    assert offenders == [], (
        f"deepeval must only be imported inside reasoning/, but also found in: {offenders}"
    )


def test_deepeval_never_imported_in_orchestration():
    # B4: orchestration/ drives DeepEval-backed strategies (deepeval_search.py)
    # entirely through the reasoning-layer bridges/models — it must never reach
    # for `deepeval` directly, even for a type as tempting to import straight as
    # ConversationalGolden. Named and asserted explicitly (not just folded into
    # the generalized reasoning-only check above) since orchestration/ is
    # exactly the layer a future StressProfile-consuming search change would be
    # tempted to add this import to.
    offenders = _import_offenders("deepeval", SRC_ROOT / "reasoning")
    orchestration_offenders = [
        o
        for o in offenders
        if o.startswith("orchestration" + "\\") or o.startswith("orchestration/")
    ]

    assert orchestration_offenders == [], (
        f"deepeval must never be imported in orchestration/, but found in: {orchestration_offenders}"
    )
