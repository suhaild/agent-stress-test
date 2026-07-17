"""Bring-your-own: wrap a subprocess as a TargetAgent via stdin/stdout JSON framing."""

import json
import subprocess

from agent_stress_test.models import AgentResponse, Message, Step
from agent_stress_test.ports import TargetAgent


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
        payload = {"messages": [m.model_dump(exclude={"cache"}) for m in conversation]}
        result = subprocess.run(
            self._command,
            input=json.dumps(payload),
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
        body = json.loads(result.stdout.strip())
        trace_data = body.get("trace")
        trace = [Step(**step) for step in trace_data] if trace_data else None
        return AgentResponse(final_reply=body["reply"], trace=trace)
