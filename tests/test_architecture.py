import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "agent_stress_test"


def _import_offenders(module: str, exempt_path: Path) -> list[str]:
    pattern = re.compile(rf"^\s*(import {module}\b|from {module}\b)", re.MULTILINE)
    offenders = []
    for path in SRC_ROOT.rglob("*.py"):
        if path == exempt_path:
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
        "httpx must only be imported in targets/http_agent.py, "
        f"but also found in: {offenders}"
    )
