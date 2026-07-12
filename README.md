# Agent Stress-Test

A tool for stress-testing AI agents before you put them in production. It runs adversarial,
multi-turn conversations against your agent to find where it breaks, then gives you a reliability
score, a catalog of failure patterns grouped by type, and full transcripts you can replay.

It's built with a hexagonal (ports-and-adapters) architecture. The core logic never touches a
provider SDK, an HTTP client, or SQLite directly. Everything external goes through three ports:
`LLMProvider`, `TargetAgent`, and `Store`.

## How it works

1. **Simulate** - an adversarial user simulator drives multi-turn conversations against your
   agent, using a library of tactics to probe for jailbreaks, policy violations, and inconsistent
   behavior.
2. **Search** - a best-first search expands the conversation tree toward whatever looks most
   likely to break the agent, within a fixed budget.
3. **Judge** - every turn gets scored by a two-tier judge (deterministic rules first, then an
   LLM judge). It always returns a reason, a confidence score, and a severity level, not just a
   pass/fail.
4. **Score** - self-consistency sampling estimates how stable the agent's behavior is, and rolls
   up into one reliability score for the run.
5. **Cluster** - confirmed failures get grouped into named clusters, using an offline hashing
   embedder and simple clustering. No vector DB, no model download.
6. **Report** - view results as a terminal report or in a web dashboard, with failure clusters and
   replayable transcripts.
7. **Regress** - promote a failure cluster into a permanent regression case, get an LLM-suggested
   fix for the system prompt, apply it, and replay the whole regression corpus to make sure
   nothing else broke. Every prompt change is versioned, so you can restore or reapply any past
   version without losing history.

## Requirements

- Python 3.11+
- An Anthropic and/or OpenAI API key, only if you want to run against a real model. The test
  suite runs fully offline against a fake provider, so you don't need a key just to develop.

## Setup

```bash
git clone <this-repo>
cd agent-stress-test
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e ".[dev]"
```

Create a `.env` file at the repo root (it's already gitignored) with whatever provider keys you
plan to use:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Copy `config/settings.example.yaml` to `config/settings.yaml` and adjust the default model or run
budgets if you want. Keys always come from `.env`, never from this file.

## Quick start

Run a stress test against the bundled sample agent, using the offline fake provider (no key
needed):

```bash
agent-stress-test run --agent-spec sample_support --provider fake --budget 6
```

Run it against a real model instead:

```bash
agent-stress-test run --agent-spec sample_support --provider anthropic/claude-3-5-sonnet-20241022
```

Then check the results:

```bash
agent-stress-test report <run_id>
agent-stress-test replay <run_id>
agent-stress-test serve --db runs.sqlite --port 8000
```

`serve` opens a dashboard at `http://127.0.0.1:8000` where you can trigger new runs, watch failures
come in live, browse clusters, and manage the regression corpus.

## CLI commands

| Command | What it does |
|---|---|
| `run` | Run a stress test against a target agent. |
| `report` | Show the report for a stored run. |
| `replay` | Replay a stored run's failing transcripts. |
| `lock` | Promote a run's failure clusters into permanent regression cases. |
| `resolve` | Mark a regression case as fixed. |
| `suggest-fix` | Suggest a system-prompt fix for one cluster's failure. |
| `regress` | Replay the regression corpus and report status. |
| `serve` | Serve the web dashboard. |

Run `agent-stress-test <command> --help` for the full options on any of these.

## Target agents

You can point this at any agent without changing anything in this codebase, through one of three
adapters:

- **Bundled sample agent** - a general tool-calling / ReAct-style demo agent
  (`config/agents/sample_support.yaml`), built on an `LLMProvider`.
- **Python-function adapter** - wrap any Python callable as a target.
- **HTTP adapter** - wrap any HTTP endpoint as a target (`--target-url`).

To test your own agent, add a new YAML spec under `config/agents/` with its system prompt, tools,
and rules, following `sample_support.yaml` as an example.

## Project layout

```
src/agent_stress_test/
  models.py               # Pydantic data models (Run, Node, Verdict, Cluster, AgentSpec, ...)
  ports.py                 # LLMProvider, TargetAgent, Store - the only way the core reaches outside
  config.py                # settings + .env loading
  composition.py           # shared wiring used by both cli.py and the dashboard
  cli.py                   # command-line entry point
  providers/                # LLMProvider adapters: litellm-backed + deterministic fake (for tests)
  targets/                  # TargetAgent adapters: sample agent, Python-fn, HTTP
  reasoning/                # simulator, self-consistency scorer, two-tier judge, clusterer, remediation
  orchestration/            # conversation tree, greedy search, reliability score, runner, regression corpus
  store/                    # SqliteStore (Store port implementation)
  report/                   # terminal report + FastAPI/htmx dashboard
tests/                      # one test module per component, fully offline
config/
  settings.example.yaml     # copy to settings.yaml
  agents/                   # AgentSpec YAML files (bundled sample_support.yaml + your own)
doc/
  GETTING_STARTED.md        # one-time setup checklist
```

## Testing

Everything runs offline against a fake provider, no key or network access needed:

```bash
pytest
```

There are also live smoke tests that hit a real provider through litellm, but they're opt-in and
not part of the default run.

## Design principles

- **Hexagonal architecture is non-negotiable.** The core only depends on the `LLMProvider`,
  `TargetAgent`, and `Store` ports. Adapters just translate; they never contain business logic.
- **Deterministic first.** Anything that can be plain rules or plain math gets built and tested
  that way before any LLM gets involved.
- **Every LLM call is mockable.** All LLM-backed components can run against the fake provider in
  tests.
- **Don't over-engineer.** No port or adapter for something that only ever has one implementation.
