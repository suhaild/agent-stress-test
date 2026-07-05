# CLAUDE.md — Agent Stress-Test

Guidance for Claude Code when working in this repository. Read this before making changes. Follow it consistently across every phase.

---

## Project Summary

Agent Stress-Test is a pre-deployment tool that stress-tests any AI agent before it goes live. It runs adversarial, multi-turn conversations against a target agent to discover how and where it breaks, then produces a reliability score, a grouped catalog of failure patterns, and replayable failure transcripts. The goal is to catch agent failures during development rather than in production.

This is a solo build, developed in sequential phases. Each phase delivers a working, tested slice and must have a passing pytest suite before the next phase begins.

---

## Golden Rules (do not violate)

1. **Hexagonal architecture is non-negotiable.** The core (reasoning + orchestration) depends only on abstract ports. It must never import a provider SDK, an HTTP client, litellm, or SQLite directly.
2. **Everything external lives behind a port.** LLM providers, target agents, and storage are all accessed only through their interfaces (`LLMProvider`, `TargetAgent`, `Store`).
3. **Deterministic-first.** Anything that can be a plain rule or plain math is built and tested deterministically before any LLM is involved.
4. **Every LLM call is mockable.** All LLM-backed components must run against the deterministic fake provider in tests. The test suite must never depend on a live API.
5. **Tests gate progress.** No phase is "done" until its pytest suite passes. Do not start a new phase on top of failing tests.
6. **Do not over-engineer.** Do not create a port or adapter for something that will only ever have one implementation. Prefer the simplest design that satisfies the phase. Clean means simple, not maximally abstracted.

---

## Architecture

Three layers, with dependencies pointing inward (outer depends on inner, never the reverse):

- **Adapter layer** (outermost) — all contact with the outside world: LLM providers, target agents, storage.
- **Reasoning layer** — the AI components: adversarial simulator, self-consistency scorer, failure judge, failure clusterer.
- **Orchestration layer** — the controller: conversation tree, search engine, reliability scorer, runner, reporting.

### Ports (the only way the core reaches outside)

- `LLMProvider` — `complete(messages)` and `sample_n(messages, n)`. Real implementation is a thin wrapper over litellm. A deterministic fake implementation backs the tests.
- `TargetAgent` — `respond(conversation) -> reply`. Implemented by the bundled sample agent and by bring-your-own adapters (Python function, HTTP).
- `Store` — persist and reload runs, nodes, verdicts, clusters. Implemented by SQLite.

### Adapters never orchestrate

Adapters only translate (outside world <-> port shape). If an adapter starts to contain business logic (more than a few lines of decision-making), that logic belongs in a reasoning or orchestration component, not the adapter.

---

## Design Patterns (use these names consistently)

Name components by their pattern so the codebase stays legible. These describe what is already being built — do not add extra machinery to "implement" them.

**Software patterns**
- **Hexagonal (Ports-and-Adapters)** — the backbone.
- **Adapter** — provider/target/store implementations.
- **Strategy** — interchangeable algorithms behind one interface: simulator tactics, and the search engine (greedy now, MCTS later).
- **Repository** — the `Store` port over runs/nodes/verdicts/clusters.
- **Composition Root / Dependency Injection** — all wiring happens in one place (`runner.py` / `cli.py`); the core never constructs its own dependencies.

**AI / agentic patterns**
- **Orchestrator-Workers** — the runner coordinates specialized workers (simulator, scorer, judge, clusterer).
- **Evaluator-Optimizer loop** — simulator generates an adversarial turn, judge evaluates it, search steers toward the most promising failures. This is the core engine cycle.
- **LLM-as-Judge (Evaluator)** — deterministic rules first, then an LLM judge; always returns a reason, a confidence score, and a severity level, not just pass/fail.
- **Self-Consistency** — sample N times, measure agreement, to estimate uncertainty.
- **Blackboard** — the conversation tree + run store is a shared knowledge space. Components collaborate through it, not by calling each other directly: the simulator writes probes, the target writes replies, the scorer/judge write scores and verdicts, the search reads scores to pick the next node, the clusterer reads confirmed failures.

---

## Tooling Constraints

**Use:**
- `litellm` — the multi-provider LLM layer (behind `LLMProvider` only).
- `pydantic` — data models and settings.
- `pytest` — testing (with mocking; `pytest-asyncio` only if async is actually used).
- `rich` / `textual` — the terminal report.
- `hdbscan` / scikit-learn + `sentence-transformers` (or provider embeddings) — failure clustering.

**Do NOT use (deliberate choices):**
- No LangChain, LlamaIndex, or any agent framework — they bury the clean architecture.
- No vector database — clustering is in-memory, not a DB.
- No MCTS library — if MCTS is built (stretch), write it directly behind the existing search interface.
- Do not add a dependency without a clear reason; keep the dependency list lean.

**Agent Skills — use actively, not just as reference:**
If the following Agent Skills are installed, invoke them at the relevant point in the build rather than treating them as background reading:
- **Hexagonal / clean-architecture skill** — use from Phase 0 onward, on every phase, to keep layering discipline (ports model capabilities not technologies; adapters only translate, never orchestrate business logic; dependencies point inward).
- For `providers/litellm_provider.py` (Phase 1), reference litellm's official docs (docs.litellm.ai) directly rather than a third-party skill.
- **LLM-evaluation skill** — use when building the failure judge, both tiers (Phases 3 and 8).
- **frontend-design skill** — use when building the dashboard (Phase 10).
- **pytest / Python-testing skill** — use throughout, for every phase's test suite.
If a relevant skill is not installed, proceed using the rules in this file and say so, rather than skipping silently.

---

## Testing Policy

- **Fast core suite:** models, rules, tree, search, reliability, store — deterministic, no network, run on every change.
- **Fake-provider suite:** simulator, consistency, LLM-judge, clustering — run against the deterministic fake provider so they are repeatable.
- **Hand-labeled sets:** small curated example sets for the judge (both tiers), asserting accuracy bars.
- **Live smoke tests:** optional, off by default, kept out of the main suite; they hit a real provider once through litellm to confirm Claude and OpenAI both work.
- The main suite must run fully offline and fast. Never require API keys to run it.

---

## Repository Layout

```
agent-stress-test/
  pyproject.toml            # deps + tool config (pytest, ruff)
  README.md                 # points to docs/GETTING_STARTED.md + the build plan
  CLAUDE.md                 # this file (root, read automatically)
  agent_stress_test_build_plan.md  # build plan (root)
  docs/
    GETTING_STARTED.md      # one-time setup checklist
  config/
    settings.example.yaml   # provider key refs, run budgets
    agents/
      sample_support.yaml   # the bundled demo agent spec
  src/ast/
    __init__.py
    config.py               # Pydantic settings + YAML loading
    models.py               # Pydantic data models (Run, Node, Verdict, Cluster, AgentSpec)
    ports.py                # the interfaces: LLMProvider, TargetAgent, Store
    providers/
      __init__.py
      fake.py               # deterministic fake provider (for tests)
      litellm_provider.py   # litellm-backed LLMProvider (Claude + OpenAI via one call)
    targets/
      __init__.py
      sample_agent.py       # bundled demo agent (built on an LLMProvider)
      python_fn.py          # bring-your-own: wrap a Python callable
      http_agent.py         # bring-your-own: wrap an HTTP endpoint
    reasoning/
      simulator.py          # adversarial user simulator + tactic library
      consistency.py        # self-consistency scorer
      judge.py              # two-tier failure judge
      clusterer.py          # failure clustering + naming
    orchestration/
      tree.py               # conversation tree structure
      search.py             # greedy best-first (MCTS later)
      reliability.py        # compounding reliability score
      runner.py             # ties everything into one run
    store/
      sqlite_store.py       # Store implementation
    report/
      terminal.py           # CLI/terminal report
      dashboard/            # (Phase 10) web dashboard
    cli.py                  # command-line entry point
  tests/
    ...                     # one test module per phase
```

Note: `src/ast/` is the package. If `ast` clashes with Python's standard-library module in practice, rename the package (e.g. `agent_stress_test`) rather than shadowing stdlib — decide this in Phase 1 and keep it consistent.

---

## Build Phases (do them in order; do not skip)

1. **Foundations** — models, ports, config, fake provider, litellm provider (litellm mocked in tests).
2. **Target agents** — bundled sample agent (general tool-calling / ReAct-style; the primary demo target) + Python-function adapter + HTTP adapter. Real-world targets for the "works on any agent" story: primary `rulyone/Simple-ReAct-Agent` (local Ollama + Llama 3.2 3B, no API key), backup `mattambrogi/agent-implementation` (OpenAI key). Wrap real agents via the Python-function or HTTP adapter — no architecture change. Verify repo license before using in the demo.
3. **Deterministic failure judge** — rules tier only.
4. **Adversarial simulator** — tactic library (Strategy pattern).
5. **Self-consistency scorer** — instability score.
6. **Conversation tree + greedy search** — first end-to-end engine (the integration point).
7. **Persistence + reliability score** — SQLite store + compounding score.
8. **LLM-as-judge (tier 2) + failure clustering.**
9. **Terminal report + CLI** — first shippable product.
10. **Visual dashboard** — built last, on representative pre-run results.
11. **MCTS search (stretch)** — only if 1-10 are green with time to spare.

At the start of each phase: state which phase, what it delivers, and its tests. At the end: confirm the pytest suite passes before moving on.

---

## Conventions

- **Python 3.11+**, type hints on all public functions and methods.
- **Pydantic** models for all structured data; no raw dicts across boundaries.
- **ruff** for linting/formatting; keep the tree clean.
- Small, focused modules matching the layout above; do not create new top-level modules without reason.
- No secrets in code. API keys come from a `.env` referenced by config; never hardcode or log them.
- Every LLM-backed component takes its `LLMProvider` via constructor/argument injection — never instantiates a provider itself.
- Prefer pure functions and immutable data where practical; keep side effects at the edges (adapters).
- Clear names over cleverness. A reader should be able to map any file to a layer and a pattern.

---

## Definition of Done (per phase)

- Code matches the layout and respects the layer boundaries.
- The phase's pytest suite passes, fully offline.
- No disallowed dependencies were added.
- Public functions are typed and named clearly.
- The work is committed to git as a clean checkpoint before the next phase.