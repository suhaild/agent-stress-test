#!/usr/bin/env python
"""Example subprocess target for config/agents/example_subprocess_target.yaml.

Reads one JSON object ``{"messages": [...]}`` from stdin, writes one JSON
object ``{"reply": "..."}`` to stdout — the wire shape ``SubprocessAgent``
(src/agent_stress_test/targets/subprocess_agent.py) expects. A real
bring-your-own target can be written in any language; this one is Python
only because that's what ships with this repo's demo — it has no dependency
on the agent_stress_test package itself.
"""

import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read())
    messages = payload.get("messages", [])
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    print(json.dumps({"reply": f"You said: {last_user}"}))


if __name__ == "__main__":
    main()
