"""Bring-your-own: wrap a subprocess as a TargetAgent via stdin/stdout JSON framing."""

import json
import subprocess

from agent_stress_test.models import AgentResponse, Message
from agent_stress_test.ports import TargetAgent
from agent_stress_test.targets.wire_protocol import _build_wire_payload, _parse_wire_response


class SubprocessAgent(TargetAgent):
    """Wraps a command-line process as a TargetAgent.

    Spawns ``command`` fresh for each ``respond()`` call — the same
    stateless request/response contract ``HttpAgent`` uses (the conversation
    tree already owns the full history, so nothing needs to persist between
    turns) — writing ``{"messages": [...]}`` as JSON to stdin and reading one
    JSON object ``{"reply": "...", "trace": [...]}`` back from stdout. A
    missing or null ``trace`` is returned as ``trace=None``, never fabricated.
    """

    def __init__(
        self,
        command: list[str],
        *,
        timeout: float = 30.0,
        cwd: str | None = None,
    ) -> None:
        self._command = command
        self._timeout = timeout
        self._cwd = cwd

    def respond(self, conversation: list[Message]) -> AgentResponse:
        result = subprocess.run(
            self._command,
            input=json.dumps(_build_wire_payload(conversation)),
            capture_output=True,
            text=True,
            timeout=self._timeout,
            cwd=self._cwd,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess target {self._command!r} exited {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return _parse_wire_response(json.loads(result.stdout.strip()))
