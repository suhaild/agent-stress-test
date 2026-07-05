import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "agent_stress_test"
IMPORT_PATTERN = re.compile(r"^\s*(import litellm\b|from litellm\b)", re.MULTILINE)


def test_litellm_only_imported_in_its_own_adapter():
    offenders = []
    for path in SRC_ROOT.rglob("*.py"):
        if path == SRC_ROOT / "providers" / "litellm_provider.py":
            continue
        if IMPORT_PATTERN.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(SRC_ROOT)))

    assert offenders == [], (
        "litellm must only be imported in providers/litellm_provider.py, "
        f"but also found in: {offenders}"
    )
